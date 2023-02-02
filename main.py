import torch
import argparse
from pathlib import Path

from models.gnn_object_detection import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('GNN OBJECT DETECT', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    return parser

def main(args):
    device = torch.device(args.device)
    
    model, criterion = build_model(args)
    model.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'GNN', parents=[get_args_parser()])
    args = parser.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
