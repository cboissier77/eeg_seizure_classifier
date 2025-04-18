import torch
import optuna
from torch.utils.data import DataLoader
import os
from utils.utils import (
    load_best_val_score,
    save_best_val_score,
    parse_args,
    load_config,
    save_config,
)
from training.train import train_model
from models.lstm_gat import EEGGraphModel
from training.losses import BinaryFocalLoss
from data.dataset import EEGDatasetWrapper
import random


def objective(trial, cfg, dataset_wrapper, selected_indices):
    """Objective function for Optuna hyperparameter optimization.
    Args:
        trial (optuna.Trial): Optuna trial object.
        cfg (dict): Configuration dictionary.
        dataset_wrapper (EEGDatasetWrapper): EEG dataset wrapper object.
    Returns:
        float: Validation score.
    """
    global best_val_score

    ############################################################################################################
    # Define hyperparameters to optimize

    lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 32, 256)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
    gat_hidden_dim = trial.suggest_int("gat_hidden_dim", 16, 128)
    gat_heads = trial.suggest_int("gat_heads", 1, 8)
    #epochs = trial.suggest_int("epochs", 3, 50)
    epochs = cfg["training"]["epochs"]  # Fixed epochs for LOOCV
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float("alpha", 0.6, 0.9)
    gamma = trial.suggest_float("gamma", 0.5, 5.0)
    fully_connected = trial.suggest_categorical("fully_connected", [True, False])

    ############################################################################################################
    # Training with leave-one-out cross-validation

    val_scores = []

    for i in selected_indices:
        train_dataset, val_dataset = dataset_wrapper.leave_one_out_split(i)

        train_loader = DataLoader(
            train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"])

        model = EEGGraphModel(
            input_dim=cfg["model"]["input_dim"],
            lstm_hidden_dim=lstm_hidden_dim,
            gat_hidden_dim=gat_hidden_dim,
            output_dim=cfg["model"]["output_dim"],
            gat_heads=gat_heads,
            lstm_layers=lstm_layers,
            fully_connected=fully_connected,
        ).to(device)

        criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        val_score = train_model(
            model, train_loader, val_loader, criterion, optimizer, epochs, device=device
        )
        val_scores.append(val_score)

    ############################################################################################################
    # Calculate average validation score

    avg_val_score = sum(val_scores) / len(val_scores)

    if avg_val_score > best_val_score:
        best_val_score = avg_val_score
        save_best_val_score(BEST_SCORE_PATH, best_val_score)
        print(f"✅ Saved new best LOOCV avg val_score: {avg_val_score:.4f}")

    return avg_val_score


BEST_SCORE_PATH = "checkpoints/optuna/best_score.json"  # Path to save the best score
best_val_score = load_best_val_score(BEST_SCORE_PATH)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Set device to GPU if available, else CPU


def main():
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 42))
    random.seed(cfg.get("seed", 42))
    dataset_wrapper = EEGDatasetWrapper(cfg["data"]["data_dir"])
    # Randomly choose 4 subject indices
    all_indices = list(range(dataset_wrapper.num_subjects()))
    selected_indices = random.sample(
        all_indices, 4
    )  # Used for LOOCV (subsampled LOOCV)

    os.makedirs("checkpoints/optuna", exist_ok=True)  # make sure the directory exists
    storage_path = "sqlite:///checkpoints/optuna/eeg_study.db"
    study = optuna.create_study(
        study_name="eeg_lstm_gat_optimization",
        direction="maximize",
        storage=storage_path,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, cfg, dataset_wrapper, selected_indices),
        n_trials=15,
        timeout=11.5 * 3600,
    )  # Timeout set to 11.5 hours because of Cluster limit
    print("Best trial:")
    print(study.best_trial)
    save_config(cfg, "configs/best_config.yaml", study)


if __name__ == "__main__":
    main()
