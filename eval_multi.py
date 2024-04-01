import argparse
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
import esm 

import util.misc as utils
from util.clean import get_ec_id_dict, get_id_seq_dict
from datasets.protdataset import ProtSeqDETRDataset
from engine import evaluate_multi
from models import build_model

warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('eval multi-func', add_help=False)
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
    parser.add_argument('--esm_layer', default=32, type=int)
    # infer
    parser.add_argument('--infer_threshold', default=0.99, type=float)
    return parser


def main(args):
    # training args
    args = get_args_parser().parse_args()
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

    test_data_list = ["new", "price"]
    test_dataset_list = []
    for test_data in test_data_list:
        test_data_pth = f"./data/multi_func/{test_data}.csv"
        id_ec_test, ec_id_test = get_ec_id_dict(test_data_pth)
        id_seq_test = get_id_seq_dict(test_data_pth)
        test_dataset = ProtSeqDETRDataset(id_ec_test, ec_id_test, id_seq_test, max_labes=args.num_queries, esm_layer=args.esm_layer, ec_to_label=ec_to_label, label_to_ec=label_to_ec)
        test_dataset_list.append(test_dataset)

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

    test_loader_list = []
    for test_dataset in test_dataset_list:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=args.num_workers)
        test_loader_list.append(test_loader)

    for test_data, test_loader in zip(test_data_list, test_loader_list):
        evaluate_multi(args.model_path, esm_model, alphabet, args.esm_layer, model, test_data, test_loader, ec_to_label, label_to_ec, args.num_queries, device, args.infer_threshold)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ProtDETR eval multi-func', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
