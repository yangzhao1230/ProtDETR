import argparse
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
import esm 

from datasets.protdataset import ProtSeqDETRDataset
from engine import evaluate_mono_enzyme_level, evaluate_mono_zero_level
from models import build_model
import util.misc as utils
from util.clean import get_ec_id_dict, get_id_seq_dict

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('eval mono-func', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model_path', type=str, default='./saved_models/ProtDETR_ECPred40.pt')
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
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--eos_coef', default=0.0, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--esm_layer', default=32, type=int)
    return parser


def main(args):
    # eval_args
    args = get_args_parser().parse_args()
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_data_pth = f"./data/mono_func/ECPred40_train.csv"
    id_ec_train, ec_id_train = get_ec_id_dict(train_data_pth)
    id_seq_train = get_id_seq_dict(train_data_pth)
    
    train_dataset = ProtSeqDETRDataset(id_ec_train, ec_id_train, id_seq_train, max_labes=args.num_queries, esm_layer=args.esm_layer)
    ec_to_label = train_dataset.ec_to_label
    label_to_ec = train_dataset.label_to_ec
    num_labels = len(ec_to_label)
    args.num_classes = num_labels

    # we comment out the validation set by default, since it takes too long to evaluate on the validation set
    # valid_data_path = f"./data/mono_func/ECPred40_valid.csv"
    # id_ec_valid, ec_id_valid = get_ec_id_dict(valid_data_path)
    # id_seq_valid = get_id_seq_dict(valid_data_path)
    # valid_dataset = ProtSeqDETRDataset(id_ec_valid, ec_id_valid, id_seq_valid, max_labes=args.num_queries, esm_layer=args.esm_layer, ec_to_label=ec_to_label, label_to_ec=label_to_ec)

    # # when we evaluate on the level 1,2,3,4, we need to remove the zero label, i.e., the non-enzyme. this remove is consistent with EnzBert
    # valid_data_path_remove_zero = './data/mono_func/ECPred40RemoveZero_valid.csv'
    # id_ec_valid_remove_zero, ec_id_valid_remove_zero = get_ec_id_dict(valid_data_path_remove_zero)
    # id_seq_valid_remove_zero = get_id_seq_dict(valid_data_path_remove_zero)
    # valid_dataset_remove_zero = ProtSeqDETRDataset(id_ec_valid_remove_zero, ec_id_valid_remove_zero, id_seq_valid_remove_zero, max_labes=args.num_queries, esm_layer=args.esm_layer, ec_to_label=ec_to_label, label_to_ec=label_to_ec)

    test_data_path = f"./data/mono_func/ECPred40_test.csv" # enzyme or not enzyme
    id_ec_test, ec_id_test = get_ec_id_dict(test_data_path)
    id_seq_test = get_id_seq_dict(test_data_path)
    test_dataset = ProtSeqDETRDataset(id_ec_test, ec_id_test, id_seq_test, max_labes=args.num_queries, esm_layer=args.esm_layer, ec_to_label=ec_to_label, label_to_ec=label_to_ec)

    test_data_remove_zero_path = './data/mono_func/ECPred40RemoveZero_test.csv' # enzyme only
    id_ec_test_remove_zero, ec_id_test_remove_zero = get_ec_id_dict(test_data_remove_zero_path)
    id_seq_test_remove_zero = get_id_seq_dict(test_data_remove_zero_path)
    test_dataset_remove_zero = ProtSeqDETRDataset(id_ec_test_remove_zero, ec_id_test_remove_zero, id_seq_test_remove_zero, max_labes=args.num_queries, esm_layer=args.esm_layer, ec_to_label=ec_to_label, label_to_ec=label_to_ec)

    esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        
    esm_model.eval()
    esm_model.to(device)

    model, criterion = build_model(args, train_dataset.ec_weight)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn, num_workers=args.num_workers)
    # valid_loader_remove_zero = DataLoader(valid_dataset_remove_zero, batch_size=args.batch_size, shuffle=False, collate_fn=valid_dataset_remove_zero.collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=args.num_workers)
    test_loader_remove_zero = DataLoader(test_dataset_remove_zero, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset_remove_zero.collate_fn, num_workers=args.num_workers)

    # evaluate_mono_zero_level(args.model_path, esm_model, alphabet, args.esm_layer, model, "ECPred40_valid_zero", valid_loader, ec_to_label, label_to_ec, args.num_queries, device)
    # evaluate_mono_enzyme_level(args.model_path, esm_model, alphabet, args.esm_layer, model, "ECPred40_valid_enzyme", valid_loader_remove_zero, ec_to_label, label_to_ec, args.num_queries, device)

    evaluate_mono_zero_level(args.model_path, esm_model, alphabet, args.esm_layer, model, "ECPred40_test_zero", test_loader, ec_to_label, label_to_ec, args.num_queries, device)
    evaluate_mono_enzyme_level(args.model_path, esm_model, alphabet, args.esm_layer, model, "ECPred40_test_enzyme", test_loader_remove_zero, ec_to_label, label_to_ec, args.num_queries, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ProtDETR eval mono', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
