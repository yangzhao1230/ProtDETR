"""
Train and eval functions
"""
import math
import sys
from typing import Iterable

import torch
import torch.distributed as dist
import util.misc as utils
import pandas as pd
from util.clean import get_true_labels, get_eval_metrics
import time
from util.metric import MacroAvg

def get_esm_emb(model, alphabet, esm_layer, batch_sequences, device):
    """
    Get ESM embedding from model
    """
    max_length = max(len(seq) for seq in batch_sequences)

    batch_tokens = []
    attention_mask = []

    for seq in batch_sequences:
        tokens = [alphabet.cls_idx] + [alphabet.get_idx(s) for s in seq] + [alphabet.eos_idx]
        padding = [alphabet.padding_idx] * (max_length + 2 - len(tokens))  # add 2 for CLS and EOS
        padded_tokens = tokens + padding
        batch_tokens.append(padded_tokens)
        
        # create mask, not include CLS and padding
        seq_mask = [0] + [1] * len(seq) + [0] + [0] * (max_length - len(seq)) 
        attention_mask.append(seq_mask)

    batch_tokens = torch.tensor(batch_tokens)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        # move tokens to model's device
        results = model(batch_tokens.to(device), repr_layers=[esm_layer], return_contacts=False)
        token_embeddings = results["representations"][esm_layer]  # (bs, seq_len+2, dim)

    # remove CLS and EOS token embeddings, only keep the original sequence token embeddings
    token_embeddings = token_embeddings[:, 1:-1, :]  # (bs, max_len, dim)

    # modify attention_mask to remove the influence of CLS and EOS tokens
    attention_mask = attention_mask[:, 1:-1]  # (bs, max_len)

    return token_embeddings, attention_mask

def train_one_epoch(esm_model, alphabet, esm_layer,
                    model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()

    total_loss = 0.0
    num_batches = 0.0

    total_esm_time = 0.0

    for batch, data in enumerate(data_loader):
        ids, seqs, labels, onehot_labels = data
        with torch.no_grad():
            esm_start = time.time()
            embs, masks = get_esm_emb(esm_model, alphabet, esm_layer, seqs, device)
            esm_end = time.time()
            total_esm_time += esm_end - esm_start

        # samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(embs.to(device=device), masks.to(device=device))
        labels = [label.to(device=device) for label in labels]
        targets = [{'labels': label} for label in labels]
        # outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        total_loss += loss_value
        num_batches += 1

    total_loss_tensor = torch.tensor(total_loss, device=device)
    if dist.is_initialized():
        dist.all_reduce(total_loss_tensor)
        avg_loss = total_loss_tensor.item() / (dist.get_world_size() * num_batches)
    else:
        avg_loss = total_loss / num_batches

    return avg_loss, total_esm_time

def evaluate_multi(model_name, esm_model, alphabet, esm_layer, model, test_data, test_loader, ec_to_label, label_to_ec, max_labels, device, infer_threshold):
    model.eval()
    num_labels = len(ec_to_label)

    predictions = []
    pred_label = [] 
    total_predict_ecs = 0
    for batch, data in enumerate(test_loader):
        ids, seqs, labels, onehot_labels = data
        with torch.no_grad():
            esm_start = time.time()
            embs, masks = get_esm_emb(esm_model, alphabet, esm_layer, seqs, device)
            outputs = model(embs.to(device=device), masks.to(device=device))  # (bs, max_labels, num_labels + 1)
            logits = outputs['pred_logits']  # (bs, max_labels, num_labels + 1)
            probas = logits.softmax(-1)[:,:,:-1]  # (bs, max_labels, num_labels)
            preds = torch.argmax(logits, dim=2)  # (bs, max_labels)
            
            keep = probas.max(-1).values > infer_threshold
            preds[keep == False] = num_labels

        batch_size = embs.size(0)
        for i in range(batch_size):
            pred_idxs = preds[i].cpu().numpy()
            pred_probas = probas[i].cpu().numpy()
            
            pred_ecs_with_probs = [(label_to_ec[idx], pred_probas[j, idx]) for j, idx in enumerate(pred_idxs) if idx != num_labels]
            
            if len(pred_ecs_with_probs) == 0:
                """
                If no EC is predicted, choose the one with the highest probability
                We do this because the model may not predict any EC for some sequences, but our baseline CLEAN always predicts at least one EC
                """
                max_prob, max_prob_label_idx = torch.max(probas[i], dim=1)
                pred_ecs_with_probs = [(label_to_ec[max_prob_label_idx[max_prob.argmax()].item()], max_prob.max().item())]
            
            pred_ecs_with_probs.sort(key=lambda x: x[1], reverse=True)
            
            pred_ecs = []
            seen = set()
            for ec, _ in pred_ecs_with_probs:
                if ec not in seen:
                    seen.add(ec)
                    pred_ecs.append(ec)

            pred_label.append(pred_ecs)
            predictions.append((ids[i], pred_ecs))
            total_predict_ecs += len(pred_ecs)

    # true_label, all_label = get_true_labels(f"./data/multi_func/{test_data}")
    # FIXME
    true_label, all_label = get_true_labels(f"./data/multi_func/{test_data}")
    pre, rec, f1, acc = get_eval_metrics(pred_label, true_label, all_label)
    print(f"Evaluate on {test_data} dataset")
    print(f"Precision: {pre:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}, Acc: {acc:.4f}")

def evaluate_mono_zero_level(model_name, esm_model, alphabet, esm_layer, model, test_data, test_loader, ec_to_label, label_to_ec, max_labels, device, infer_threshold=0.9):
    """
    Evaluate model on test data with only level 0 prediction, i.e., justify whether the sequence is enzyme or not.
    Keep consistent with the evaluation in the EnzBert paper.
    Threshold is infact not used in this function.
    This function has many redundant code, more optimizion should be done.
    """
    print(f"Evaluate on {test_data} dataset at level 0 (enzyme or not enzyme)")
    model.eval()
    num_labels = len(ec_to_label)
    y_true = []
    y_pred = []
    for batch, data in enumerate(test_loader):
        ids, seqs, labels, onehot_labels = data
        with torch.no_grad():
            embs, masks = get_esm_emb(esm_model, alphabet, esm_layer, seqs, device)
            outputs = model(embs.to(device=device), masks.to(device=device))  # (bs, max_labels, num_labels + 1)
            logits = outputs['pred_logits']  # (bs, max_labels, num_labels + 1)
            probas = logits.softmax(-1)[:,:,:-1]  # (bs, max_labels, num_labels)
            preds = torch.argmax(logits, dim=2)  # (bs, max_labels)
            
            keep = probas.max(-1).values > infer_threshold
            preds[keep == False] = num_labels

        batch_size = embs.size(0)
        for i in range(batch_size):

            pred_idxs = preds[i].cpu().numpy()
            pred_probas = probas[i].cpu().numpy()
        
            pred_ecs_with_probs = [(label_to_ec[idx], pred_probas[j, idx]) for j, idx in enumerate(pred_idxs) if idx != num_labels]
            if len(pred_ecs_with_probs) == 0:
                max_prob, max_prob_label_idx = torch.max(probas[i], dim=1)
                pred_ecs_with_probs = [(label_to_ec[max_prob_label_idx[max_prob.argmax()].item()], max_prob.max().item())]
            pred_ecs_with_probs.sort(key=lambda x: x[1], reverse=True)
            
            pred_ecs = []
            seen = set()
            for ec, _ in pred_ecs_with_probs:
                if ec not in seen:
                    seen.add(ec)
                    pred_ecs.append(ec)

            pred_ec = pred_ecs[0] # top1 prediction, for mono-function prediction
            true_ec = label_to_ec[labels[i].item()]
            y_true.append(true_ec)
            y_pred.append(pred_ec)
            
    metric_obj = MacroAvg(separator=".")
    metric_obj.y_pred = y_pred
    metric_obj.y_true = y_true
    metric_obj.get_all_metrics(level_max=0)

def evaluate_mono_enzyme_level(model_name, esm_model, alphabet, esm_layer, model, test_data, test_loader, ec_to_label, label_to_ec, max_labels, device, infer_threshold=0.9):
    """
    Evaluate model on test data with only enzyme level prediction.
    Keep consistent with the evaluation in the EnzBert paper.
    Threshold is infact not used in this function.
    This function has many redundant code, more optimizion should be done.
    """
    print(f"Evaluate on {test_data} dataset, remove zero label (Non-enzyme), at level 1,2,3,4")
    model.eval()
    num_labels = len(ec_to_label)
    y_true = []
    y_pred = []
    zero_label = ec_to_label["0.0.0.0"]
    for batch, data in enumerate(test_loader):
        ids, seqs, labels, onehot_labels = data
        with torch.no_grad():
            embs, masks = get_esm_emb(esm_model, alphabet, esm_layer, seqs, device)
            outputs = model(embs.to(device=device), masks.to(device=device))  # (bs, max_labels, num_labels + 1)
            logits = outputs['pred_logits']  # (bs, max_labels, num_labels + 1)
            probas = logits.softmax(-1)[:,:,:-1]  # (bs, max_labels, num_labels)
            # set zero label prob to 0
            probas[:,:,zero_label] = 0
            preds = torch.argmax(logits, dim=2)  # (bs, max_labels)
            
            keep = probas.max(-1).values > infer_threshold
            preds[keep == False] = num_labels

        batch_size = embs.size(0)
        for i in range(batch_size):
            pred_idxs = preds[i].cpu().numpy()
            pred_probas = probas[i].cpu().numpy()
            
            pred_ecs_with_probs = [(label_to_ec[idx], pred_probas[j, idx]) for j, idx in enumerate(pred_idxs) if idx != num_labels]

            if len(pred_ecs_with_probs) == 0:
                max_prob, max_prob_label_idx = torch.max(probas[i], dim=1)
                pred_ecs_with_probs = [(label_to_ec[max_prob_label_idx[max_prob.argmax()].item()], max_prob.max().item())]
            
            pred_ecs_with_probs.sort(key=lambda x: x[1], reverse=True)
            
            pred_ecs = []
            seen = set()
            for ec, _ in pred_ecs_with_probs:
                if ec not in seen:
                    seen.add(ec)
                    pred_ecs.append(ec)

            pred_ec = pred_ecs[0] # top1 prediction for mono-function prediction
            true_ec = label_to_ec[labels[i].item()]
            y_true.append(true_ec)
            y_pred.append(pred_ec)

    metric_obj = MacroAvg(separator=".")
    metric_obj.y_pred = y_pred
    metric_obj.y_true = y_true
    last_f1 = metric_obj.get_all_metrics(level_max=4) # use level 4's f1 to select the best model
    return last_f1