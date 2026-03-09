import argparse
import csv
import random
import warnings

import numpy as np
import torch
import esm

import util.misc as utils
from util.clean import get_ec_id_dict, get_id_seq_dict
from datasets.protdataset import ProtSeqDETRDataset
from engine import decode_multi_predictions, get_esm_emb
from models import build_model

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('predict multi-func', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
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
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--train_data', default='split100', type=str)
    parser.add_argument('--train_data_path', default=None, type=str)
    parser.add_argument('--esm_layer', default=32, type=int)
    parser.add_argument('--infer_threshold', default=0.99, type=float)

    parser.add_argument('--input_path', required=True, type=str)
    parser.add_argument('--output_path', default='./predictions.tsv', type=str)
    return parser


def build_record(entry, sequence):
    sequence = sequence.strip().upper()
    return entry.strip(), sequence[:1022], len(sequence), int(len(sequence) > 1022)


def load_fasta(input_path):
    records = []
    entry = None
    sequence = []
    with open(input_path, 'r') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line[0] == '>':
                if entry is not None:
                    records.append(build_record(entry, ''.join(sequence)))
                entry = line[1:]
                sequence = []
            else:
                sequence.append(line)
    if entry is not None:
        records.append(build_record(entry, ''.join(sequence)))
    return records


def load_table(input_path):
    with open(input_path, 'r') as handle:
        delimiter = '\t' if '\t' in handle.readline() else ','
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [build_record(row['Entry'], row['Sequence']) for row in reader]


def load_input_data(input_path):
    if input_path.endswith(('.fa', '.faa', '.fasta')):
        return load_fasta(input_path)
    return load_table(input_path)


def save_predictions(output_path, records, predictions):
    delimiter = '\t' if output_path.endswith('.tsv') else ','
    with open(output_path, 'w', newline='') as handle:
        writer = csv.writer(handle, delimiter=delimiter)
        writer.writerow([
            'Entry',
            'Input_Length',
            'Used_Length',
            'Truncated',
            'Top1_EC',
            'Top1_Probability',
            'Predicted_ECs',
            'Predicted_Probabilities',
        ])
        for record, pred_pairs in zip(records, predictions):
            entry, _, input_length, truncated = record
            top1_ec, top1_prob = pred_pairs[0]
            writer.writerow([
                entry,
                input_length,
                min(input_length, 1022),
                truncated,
                top1_ec,
                f'{top1_prob:.6f}',
                ';'.join(ec for ec, _ in pred_pairs),
                ';'.join(f'{prob:.6f}' for _, prob in pred_pairs),
            ])


def main(args):
    args = get_args_parser().parse_args()
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_data = load_input_data(args.input_path)

    train_data_pth = args.train_data_path or f"./data/multi_func/{args.train_data}.csv"
    id_ec_train, ec_id_train = get_ec_id_dict(train_data_pth)
    id_seq_train = get_id_seq_dict(train_data_pth)

    train_dataset = ProtSeqDETRDataset(id_ec_train, ec_id_train, id_seq_train, max_labes=args.num_queries, esm_layer=args.esm_layer)
    ec_to_label = train_dataset.ec_to_label
    label_to_ec = train_dataset.label_to_ec
    args.num_classes = len(ec_to_label)

    esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm_model.eval()
    esm_model.to(device)

    model, criterion = build_model(args, train_dataset.ec_weight)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    predictions = []
    for i in range(0, len(input_data), args.batch_size):
        batch_data = input_data[i:i + args.batch_size]
        batch_sequences = [x[1] for x in batch_data]
        with torch.no_grad():
            embs, masks = get_esm_emb(esm_model, alphabet, args.esm_layer, batch_sequences, device)
            outputs = model(embs.to(device=device), masks.to(device=device))
            _, batch_predictions = decode_multi_predictions(outputs['pred_logits'], label_to_ec, args.infer_threshold)
        predictions.extend(batch_predictions)

    save_predictions(args.output_path, input_data, predictions)
    print(f"Saved predictions to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ProtDETR predict multi-func', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
