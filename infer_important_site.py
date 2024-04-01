import argparse
import os
import random
import warnings

import numpy as np
import torch
import esm 
import requests

from datasets.protdataset import ProtSeqDETRDataset
from models import build_model
from engine import get_esm_emb
import util.misc as utils
from util.clean import get_ec_id_dict, get_id_seq_dict

warnings.filterwarnings("ignore")

def fetch_sequence(uniprot_id):
    """fetches the sequence for a given UniProt ID from the UniProt website"""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        sequence = ''.join(response.text.split('\n')[1:])
        return sequence
    else:
        print("Failed to fetch sequence for", uniprot_id)
        return None

def get_user_input():
    """get user input for UniProt ID and active sites"""
    uniprot_id = input("Enter UniProt ID ('exit' to quit): ")
    if uniprot_id.lower() == 'exit':
        return 'exit', None, None
    sequence = fetch_sequence(uniprot_id)
    if not sequence:
        return 'continue', None, None
    active_sites_input = input("Enter active sites separated by commas (e.g., 32,47,140): ")
    active_sites = list(map(int, active_sites_input.split(',')))
    return uniprot_id, sequence, active_sites

def save_results_to_file(uniprot_id, results):
    """save results to a file in the 'uniprot_results' directory in the current working directory"""
    results_dir = "./uniprot_results"
    os.makedirs(results_dir, exist_ok=True)  # create the directory if it doesn't exist
    file_path = os.path.join(results_dir, f"{uniprot_id}.txt")
    with open(file_path, 'w') as f:
        f.write(results)
    print(f"Results saved for {uniprot_id} in {file_path}")

def process_and_save_results(uniprot_id, label_to_ec, active_sites, enc_attn_weights, dec_attn_weights, logits):
    """process the model outputs and save the results to a file"""
    # calculate the average encoder attention weights and detect active sites
    avg_enc_attn_weights = np.mean(enc_attn_weights, axis=0).mean(axis=0)
    avg_enc_attn_weights = unit_length_norm2_normalize(avg_enc_attn_weights)
    enc_top_n_idx = np.argsort(avg_enc_attn_weights)[-args.top_n:][::-1] + 1
    encoder_detected_sites = list(set(active_sites) & set(enc_top_n_idx))
    # format the encoder results as a string
    results_str = "Encoder Top N attention sites:\n" + ', '.join(map(str, enc_top_n_idx)) + '\n'
    results_str += "Encoder detected active sites: " + ', '.join(map(str, encoder_detected_sites)) + '\n'
    # decoder predictions and attention analysis
    results_str += "\nDecoder Predictions and Attention Analysis:\n"
    for query_idx in range(logits.shape[0]):  
        pred_class = np.argmax(logits[query_idx, :-1])  # the last class is the no-enzyme class
        pred_ec = label_to_ec.get(pred_class, "Unknown EC Number")  
        query_attn_weights = np.mean(dec_attn_weights[:, query_idx, :], axis=0)  
        query_attn_weights = unit_length_norm2_normalize(query_attn_weights)  
        dec_top_n_idx = np.argsort(query_attn_weights)[-args.top_n:][::-1] + 1 

        detected_active_sites = list(set(active_sites) & set(dec_top_n_idx))

        results_str += f"\nQuery {query_idx + 1}:\n"
        results_str += f"Predicted EC: {pred_ec}\n"
        results_str += f"Decoder Top N attention sites: {', '.join(map(str, dec_top_n_idx))}\n"
        results_str += f"Detected active sites: {', '.join(map(str, detected_active_sites)) if detected_active_sites else 'None'}\n"

    save_results_to_file(uniprot_id, results_str)

def unit_length_norm2_normalize(vec_score):
    """
    Normalize a vector to unit length norm 2.
    """
    vec_score = np.array(vec_score, dtype=np.float64)
    norm_vec_score = np.sqrt(np.sum(np.power(vec_score, 2)))
    norm_vec = np.divide(
        vec_score,
        norm_vec_score,
        out=np.zeros_like(vec_score),
        where=norm_vec_score != 0,
    )
    return norm_vec

def remove_hook(hooks):
    for hook in hooks:
        hook.remove()

def get_args_parser():
    parser = argparse.ArgumentParser('infer important sites', add_help=False)
    parser.add_argument('--model_path', type=str, default='./saved_models/ProtDETR_split100.pt')
    # * Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=3, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--eos_coef', default=0.0, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--train_data', default='split100', type=str)
    parser.add_argument('--esm_layer', default=32, type=int)
    # infer
    parser.add_argument('--top_n', default=10, type=int)
    return parser

def main(args):
    # training args
    args = get_args_parser().parse_args()
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_data_pth = f"./data/multi_func/{args.train_data}.csv"
    id_ec_train, ec_id_train = get_ec_id_dict(train_data_pth)
    id_seq_train = get_id_seq_dict(train_data_pth)
    
    train_dataset = ProtSeqDETRDataset(id_ec_train, ec_id_train, id_seq_train, max_labes=args.num_queries, esm_layer=args.esm_layer)
    ec_to_label = train_dataset.ec_to_label
    label_to_ec = train_dataset.label_to_ec
    num_labels = len(ec_to_label)
    args.num_classes = num_labels

    esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm_model.eval()
    esm_model.to(device)

    model, criterion = build_model(args, train_dataset.ec_weight)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    while True:
        uniprot_id, sequence, active_sites = get_user_input()
        if uniprot_id == 'exit':
            break
        elif uniprot_id == 'continue':
            continue
        
        if sequence:
            with torch.no_grad():
                enc_attn_weights_list = [[] for _ in range(args.enc_layers)]
                dec_attn_weights_list = [[] for _ in range(args.dec_layers)]
                hooks = []
                for i in range(args.enc_layers):
                    hook = model.transformer.encoder.layers[i].self_attn.register_forward_hook(
                        lambda self, input, output, idx=i: enc_attn_weights_list[idx].append(output[1].squeeze(0))
                    )
                    hooks.append(hook)

                for i in range(args.dec_layers):
                    hook = model.transformer.decoder.layers[i].multihead_attn.register_forward_hook(
                        lambda self, input, output, idx=i: dec_attn_weights_list[idx].append(output[1].squeeze(0))
                    )
                    hooks.append(hook)

                sequence = sequence[:1022] # only take the first 1022 amino acids
                embs, masks = get_esm_emb(esm_model, alphabet, args.esm_layer, [sequence], device)
                masks = masks.to(device)
                outputs = model(embs, masks)
                logits = outputs["pred_logits"].cpu().numpy() # (1, query, num_classes)
                logits = logits.squeeze(0) # -> (query, num_classes) 
                enc_attn_weights = torch.stack([torch.cat(layer_weights, dim=0) for layer_weights in enc_attn_weights_list], dim=0).cpu().numpy() # (layers, len, len)
                dec_attn_weights = torch.stack([torch.cat(layer_weights, dim=0) for layer_weights in dec_attn_weights_list], dim=0).cpu().numpy() # (layers, num_query, len)

                process_and_save_results(uniprot_id, label_to_ec, active_sites, enc_attn_weights, dec_attn_weights, logits)

                remove_hook(hooks)

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('infer important sites', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
