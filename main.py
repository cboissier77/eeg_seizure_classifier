import torch
import argparse
import yaml
import optuna
from training.train import train_model
from models.lstm_gat import EEGGraphModel
from training.losses import BinaryFocalLoss
from data.dataset import EEGDatasetWrapper
from torch.utils.data import DataLoader
import os
import copy
import json

def load_best_val_score(path):
    try:
        with open(path, 'r') as f:
            return float(json.load(f)['best_val_score'])
    except (FileNotFoundError, ValueError):
        return float('-inf')

def save_best_val_score(path, score):
    with open(path, 'w') as f:
        json.dump({'best_val_score': score}, f)

BEST_SCORE_PATH = "checkpoints/optuna/best_score.json"
best_val_score = load_best_val_score(BEST_SCORE_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lstm_gat.yaml')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def objective(trial, cfg, train_dataset, val_dataset):
    global best_val_score

    lstm_hidden_dim = trial.suggest_int('lstm_hidden_dim', 32, 256)
    lstm_layers = trial.suggest_int('lstm_layers', 1, 4)
    gat_hidden_dim = trial.suggest_int('gat_hidden_dim', 16, 128)
    gat_heads = trial.suggest_int('gat_heads', 1, 8)
    epochs = trial.suggest_int('epochs', 3, 50)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float('alpha', 0.6, 0.9)
    gamma = trial.suggest_float('gamma', 0.5, 5.0)
    fully_connected = trial.suggest_categorical('fully_connected', [True, False])

    # Update config
    model_cfg = copy.deepcopy(cfg['model'])
    training_cfg = copy.deepcopy(cfg['training'])
    loss_cfg = copy.deepcopy(cfg['loss'])

    model_cfg['lstm_hidden_dim'] = lstm_hidden_dim
    model_cfg['gat_hidden_dim'] = gat_hidden_dim
    model_cfg['gat_heads'] = gat_heads
    model_cfg['lstm_layers'] = lstm_layers
    model_cfg['fully_connected'] = fully_connected
    training_cfg['lr'] = lr
    cfg['training']['epochs'] = epochs
    loss_cfg['alpha'] = alpha
    loss_cfg['gamma'] = gamma

    train_loader = DataLoader(train_dataset, batch_size=training_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg['batch_size'])

    # Model, loss, optimizer
    model = EEGGraphModel(
        input_dim=model_cfg['input_dim'],
        lstm_hidden_dim=lstm_hidden_dim,
        gat_hidden_dim=gat_hidden_dim,
        output_dim=model_cfg['output_dim'],
        gat_heads=gat_heads,
        lstm_layers=lstm_layers,
        fully_connected=fully_connected

    ).to(device)

    criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_score = train_model(model, train_loader, val_loader, criterion, optimizer, cfg, device=device)

    # Save the best model
    if val_score > best_val_score:
        best_val_score = val_score
        os.makedirs(cfg['training']['best_model_path'], exist_ok=True)
        best_model_path = os.path.join(cfg['training']['best_model_path'], 'best_model.pt')
        torch.save(model.cpu().state_dict(), best_model_path)
        save_best_val_score(BEST_SCORE_PATH, best_val_score)
        print(f"✅ Saved new best model with val_score: {val_score:.4f}")
        model.to(device)

    return val_score

def main():
    args = parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.get('seed', 42))

    dataset_wrapper = EEGDatasetWrapper(cfg['data']['data_dir'])
    train_dataset, val_dataset = dataset_wrapper.train_test_split(
        train_size=cfg['data']['train_split'],
        train_subsample=cfg['data']['train_subsample']
    )
    os.makedirs("checkpoints/optuna", exist_ok=True)  # make sure the directory exists

    storage_path = "sqlite:///checkpoints/optuna/eeg_study.db"

    study = optuna.create_study(
        study_name="eeg_lstm_gat_optimization",
        direction='maximize',
        storage=storage_path,
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, cfg, train_dataset, val_dataset), n_trials=15,timeout=11.5 * 3600)

    print("Best trial:")
    print(study.best_trial)

    # Save best config
    best_params = study.best_trial.params
    best_model_cfg = copy.deepcopy(cfg['model'])
    best_training_cfg = copy.deepcopy(cfg['training'])
    best_loss_cfg = copy.deepcopy(cfg['loss'])

    best_model_cfg['lstm_hidden_dim'] = best_params['lstm_hidden_dim']
    best_model_cfg['gat_hidden_dim'] = best_params['gat_hidden_dim']
    best_model_cfg['gat_heads'] = best_params['gat_heads']
    best_training_cfg['lr'] = best_params['lr']
    best_training_cfg['epochs'] = best_params['epochs']
    best_loss_cfg['alpha'] = best_params['alpha']
    best_loss_cfg['gamma'] = best_params['gamma']
    best_model_cfg['lstm_layers'] = best_params['lstm_layers']
    best_model_cfg['fully_connected'] = best_params['fully_connected']
    best_training_cfg['batch_size'] = cfg['training']['batch_size']

    best_config = {
        'model': best_model_cfg,
        'training': best_training_cfg,
        'loss': best_loss_cfg
    }

    os.makedirs('configs', exist_ok=True)
    with open('configs/best_config.yaml', 'w') as f:
        yaml.dump(best_config, f)


if __name__ == '__main__':
    main()
