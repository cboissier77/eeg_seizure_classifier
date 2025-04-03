import torch
import argparse
import yaml
from training.train import train_model
from training.eval import evaluate_model
from models.lstm_gat import EEGGraphModel
from training.losses import BinaryFocalLoss
from data.dataset import EEGDatasetWrapper
from torch.utils.data import DataLoader
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lstm_gat.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Set seeds (optional)
    torch.manual_seed(cfg.get('seed', 42))

    # Load datasets
    dataset_wrapper = EEGDatasetWrapper(cfg['data']['data_dir'])
    train_dataset, val_dataset = dataset_wrapper.train_test_split(train_size=cfg['data']['train_split'], train_subsample=cfg['data']['train_subsample'])

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'])

    # Initialize model
    model = EEGGraphModel(
        input_dim=cfg['model']['input_dim'],
        lstm_hidden_dim=cfg['model']['lstm_hidden_dim'],
        gat_hidden_dim=cfg['model']['gat_hidden_dim'],
        output_dim=cfg['model']['output_dim'],
        gat_heads=cfg['model']['gat_heads']
    )

    # Loss and optimizer
    criterion = BinaryFocalLoss(alpha=cfg['loss']['alpha'], gamma=cfg['loss']['gamma'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])

    if args.mode == 'train':
        train_model(model, train_loader, val_loader, criterion, optimizer, cfg)
    else:
        # Load the best model for evaluation
        model.load_state_dict(torch.load(cfg['training']['best_model_path']))
        evaluate_model(model, val_loader, cfg)

if __name__ == '__main__':
    main()
