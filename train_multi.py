import argparse
import os
import random
import time
import warnings

import esm
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from datasets.protdataset import ProtSeqDETRDataset
from engine import evaluate_multi, train_one_epoch
from models import build_model
import util.misc as utils
from util.clean import get_ec_id_dict, get_id_seq_dict

warnings.filterwarnings("ignore")

def get_dist_args():
  """
    Get distributed arguments from environment variables for multi-node multi-gpu training.
  """
  envvars = [
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "NODE_RANK",
    "NODE_COUNT",
    "HOSTNAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_SOCKET_IFNAME",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "AZ_BATCHAI_MPI_MASTER_NODE",
  ]
  args = dict(gpus_per_node=torch.cuda.device_count())
  missing = []
  for var in envvars:
    if var in os.environ:
      args[var] = os.environ.get(var)
      try:
        args[var] = int(args[var])
      except ValueError:
        pass
    else:
      missing.append(var)

  print(f"II Args: {args}")
  if missing:
    print(f"II Environment variables not set: {', '.join(missing)}.")
    
  return args

def get_args_parser():
    parser = argparse.ArgumentParser('ProtDETR', add_help=False)
    # training args
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--model_name', type=str, default='ProtDETR_split100')
    # Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # Transformer
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
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses (loss at each layer)")
    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float, help="Class coefficient in the matching cost")
    # weight of cross entropy loss for no-enzyme class
    parser.add_argument('--eos_coef', default=0.0, type=float, help="Relative classification weight of the no-enzyme class")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--train_data', default='split100', type=str)
    parser.add_argument('--esm_layer', default=32, type=int) # we follow ECRECER to set the layer to 32
    return parser

def main(args):
    # training args
    args = get_args_parser().parse_args()
    print(args)
    # ddp args
    dist_args = get_dist_args()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.distributed = True
    else:
        args.distributed = False
    if args.distributed:
        master_uri = "tcp://%s:%s" % (dist_args.get("MASTER_ADDR"), dist_args.get("MASTER_PORT"))
        os.environ["NCCL_DEBUG"] = "WARN"
        node_rank = dist_args.get("NODE_RANK")
        gpus_per_node = torch.cuda.device_count()
        world_size = dist_args.get("WORLD_SIZE")
        gpu_rank = dist_args.get("LOCAL_RANK")
        node_rank = 0 if node_rank is None else node_rank
        global_rank = node_rank * gpus_per_node + gpu_rank
        dist.init_process_group(
            backend="nccl", init_method=master_uri, world_size=world_size, rank=global_rank
        )
        torch.cuda.set_device(gpu_rank)
        device = torch.device("cuda", gpu_rank)
    else:
        device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_data_pth = f"./data/multi_func/{args.train_data}.csv" # e.g. split100
    id_ec_train, ec_id_train = get_ec_id_dict(train_data_pth)
    id_seq_train = get_id_seq_dict(train_data_pth)
    
    train_dataset = ProtSeqDETRDataset(id_ec_train, ec_id_train, id_seq_train, max_labes=args.num_queries, esm_layer=args.esm_layer)
    ec_to_label = train_dataset.ec_to_label
    label_to_ec = train_dataset.label_to_ec
    num_labels = len(ec_to_label)
    args.num_classes = num_labels

    # download esm model only once  
    if args.distributed:
        if utils.get_rank() == 0:
            esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
            torch.distributed.barrier()
        else:
            torch.distributed.barrier()
            esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    else:
        esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        
    esm_model.eval()
    esm_model.to(device)

    model, criterion = build_model(args, train_dataset.ec_weight)
    model.to(device)
    criterion.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_rank], output_device=gpu_rank,)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler_train, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers)

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_loss, esm_time = train_one_epoch(esm_model, alphabet, args.esm_layer,
            model, criterion, train_loader, optimizer, device, epoch,
            args.clip_max_norm)

        lr_scheduler.step()
        train_end_time = time.time()
        if utils.get_rank() == 0:
            print(f'Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Time: {(train_end_time - epoch_start_time):.2f}s | ESM Time: {esm_time:.2f}s')

    if utils.get_rank() == 0:
        if dist.is_initialized():
            torch.save(model.module.state_dict(), f"./saved_models/{args.model_name}.pt")
        else:
            torch.save(model.state_dict(), f"./saved_models/{args.model_name}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ProtDETR multi-func training script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
