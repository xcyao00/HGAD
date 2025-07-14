import argparse
import warnings
import os
import csv
import json
from datetime import datetime

import torch
import train
from utils import init_seeds, setting_lr_parameters
warnings.filterwarnings('ignore')


def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config_json(config_path, args):
    """Save configuration to JSON file"""
    config = vars(args).copy()
    # Remove non-serializable items
    if 'device' in config:
        config['device'] = str(config['device'])
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser('HGAD: Hierarchical Gaussian Mixture Normalizing Flow Modeling for Unified Anomaly Detection')
    
    # Config file option
    parser.add_argument('--config', default=None, type=str,
                        help='Path to config file (JSON format)')
    
    # Model parameters
    parser.add_argument('--backbone_arch', default='tf_efficientnet_b6', type=str, 
                        help='feature extractor: (default: efficientnet_b6)')
    parser.add_argument('--flow_arch', default='conditional_flow_model', type=str, 
                        help='normalizing flow model (default: cnflow)')
    parser.add_argument('--feature_levels', default=3, type=int, 
                        help='nudmber of feature layers (default: 3)')
    parser.add_argument('--coupling_layers', default=12, type=int, 
                        help='number of coupling layers used in normalizing flow (default: 8)')
    parser.add_argument('--clamp_alpha', default=1.9, type=float, 
                        help='clamp alpha hyperparameter in normalizing flow (default: 1.9)')
    parser.add_argument('--pos_embed_dim', default=256, type=int,
                        help='dimension of positional enconding (default: 128)')
    parser.add_argument('--lambda1', default=1.0, type=float, 
                        help='hyperparameter lambad_1 in the loss (default: 1.0)')
    parser.add_argument('--lambda2', default=100.0, type=float, 
                        help='hyperparameter lambad_2 in the loss (default: 100.0)')
    parser.add_argument('--label_smoothing', default=0.02, type=float, 
                        help='smoothing the class labels (default: 0.02)')
    parser.add_argument('--n_intra_centers', default=10, type=int, 
                        help='number of intra-class centers (default: 10)')
    
    # Data configures
    parser.add_argument('--dataset', default='mvtec', type=str,
                        choices=['mvtec', 'btad', 'mvtec3d', 'visa', 'union'])
    parser.add_argument('--img_size', default=1024, type=int, 
                        help='image size (default: 1024)')
    parser.add_argument('--msk_size', default=256, type=int, 
                        help='mask size (default: 256)')
    parser.add_argument('--batch_size', default=8, type=int, 
                        help='train batch size (default: 32)')
    
    # training configures
    parser.add_argument('--lr', type=float, default=2e-4, 
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--lr_decay_epochs', nargs='+', default=[50, 75, 90],
                        help='learning rate decay epochs (default: [50, 75, 90])')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, 
                        help='learning rate decay rate (default: 0.1)')
    parser.add_argument('--lr_warm', type=bool, default=True, 
                        help='learning rate warm up (default: True)')
    parser.add_argument('--lr_warm_epochs', type=int, default=2, 
                        help='learning rate warm up epochs (default: 2)')
    parser.add_argument('--lr_cosine', type=bool, default=True, 
                        help='cosine learning rate schedular (default: True)')
    parser.add_argument('--temp', type=float, default=0.5, 
                        help='temp of cosine learning rate scheduler (default: 0.5)')                    
    parser.add_argument('--meta_epochs', type=int, default=25, 
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub_epochs', type=int, default=8, 
                        help='number of sub epochs to train (default: 8)')
    
    # misc
    parser.add_argument("--gpu", default='0', type=str, 
                        help='GPU device number')
    parser.add_argument("--seed", default=0, type=int, 
                        help='Random seed')
    parser.add_argument('--print_freq', default=200, type=int, 
                        help='frequency to print information')
    parser.add_argument('--output_dir', default='./outputs', type=str, 
                        help='directory to save model weights')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        config_data = load_config(args.config)
        
        # Update args with config file values (command line args take precedence)
        for key, value in config_data.items():
            if key not in ['config']:  # Don't override config path
                if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)
    
    return args
    
    
if __name__ == '__main__':
    args = parse_args()
    init_seeds(args.seed)
    setting_lr_parameters(args)
    
    args.device = torch.device("cuda:" + args.gpu)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.dataset}_seed{args.seed}_gpu{args.gpu}_{timestamp}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories for logs and results
    args.log_dir = os.path.join(exp_dir, "logs")
    args.result_dir = os.path.join(exp_dir, "results")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Save experiment configuration (both text and JSON formats)
    config_path = os.path.join(exp_dir, "config.txt")
    config_json_path = os.path.join(exp_dir, "config.json")
    
    with open(config_path, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    save_config_json(config_json_path, args)
    
    # set dataset path
    if args.dataset == 'mvtec':
        args.data_path = '/Volume/VAD/Data/MVTecAD'
    elif args.dataset == 'btad':
        args.data_path = '/data/data1/yxc/datasets/btad'
    elif args.dataset == 'mvtec3d':
        args.data_path = '/data/data1/yxc/datasets/mvtec_3d_anomaly_detection'
    elif args.dataset == 'visa':
        args.data_path = '/data/data1/yxc/datasets/visa'
    elif args.dataset == 'union':
        args.data_path = ['/data/data1/yxc/datasets/mvtec_anomaly_detection',
                          '/data/data1/yxc/datasets/btad',
                          '/data/data1/yxc/datasets/mvtec_3d_anomaly_detection',
                          '/data/data1/yxc/datasets/visa']
    else:
        raise ValueError('Unrecognized or unsupported dataset!')
    
    train.train(args)
