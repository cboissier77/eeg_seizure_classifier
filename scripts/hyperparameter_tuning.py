import torch
import optuna
from torch.utils.data import DataLoader
import os
from utils.utils import (
    load_best_val_score,
    save_best_val_score,
    save_config,
)
from training.train import train_model
from models import EEG_LSTM_Model, EEG_LSTM_GAT_Model
from training.losses import BinaryFocalLoss


def objective(
    trial, cfg, dataset_wrapper, selected_ids, best_val_score, device, best_score_path
):
    """Objective function for Optuna hyperparameter optimization.
    Args:
        trial (optuna.Trial): Optuna trial object.
        cfg (dict): Configuration dictionary.
        dataset_wrapper (EEGDatasetWrapper): EEG dataset wrapper object.
        selected_ids (list): List of subject IDs for leave-one-out cross-validation.
        best_val_score (float): Best validation score so far.
        device (torch.device): Device to use for training (CPU or GPU).
    Returns:
        float: Validation score.
    Raises:
        ValueError: If the model name is not supported.
    """
    ############################################################################################################
    # Define hyperparameters to optimize

    model_name = cfg["model"]["name"]
    if model_name == "lstm_gat":

        lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 32, 256)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
        gat_hidden_dim = trial.suggest_int("gat_hidden_dim", 16, 128)
        gat_heads = trial.suggest_int("gat_heads", 1, 8)
        epochs = cfg["training"]["epochs"]  # Fixed epochs for LOOCV
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        alpha = trial.suggest_float("alpha", 0.6, 0.9)
        gamma = trial.suggest_float("gamma", 0.5, 5.0)
        fully_connected = trial.suggest_categorical("fully_connected", [True, False])

    elif model_name == "lstm":
        # lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 32, 256)
        lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 32, 100)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
        epochs = cfg["training"]["epochs"]
        alpha = cfg["loss"]["alpha"]
        gamma = cfg["loss"]["gamma"]
        lr = cfg["training"]["lr"]
    
    elif model_name == "lstm_freeze_gat":
        lstm_hidden_dim = cfg["model"]["lstm_hidden_dim"]
        lstm_layers = cfg["model"]["lstm_layers"]
        epochs = cfg["training"]["epochs"]
        lr = cfg["training"]["lr"]
        alpha = cfg["loss"]["alpha"]
        gamma = cfg["loss"]["gamma"]
        gat_hidden_dim = trial.suggest_int("gat_hidden_dim", 16, 32)
        gat_heads = trial.suggest_int("gat_heads", 1, 8)
        fully_connected = trial.suggest_categorical("fully_connected", [True, False])

    else:
        raise ValueError(f"Model {model_name} not supported for hyperparameter tuning.")
    ############################################################################################################
    # Training with leave-one-out cross-validation

    val_scores = []

    for i in selected_ids:
        print(f"Training with subject {i} as validation set")
        # Split the dataset into training and validation sets
        subjects_ids_train = [x for x in selected_ids if x != i]
        train_dataset, val_dataset = dataset_wrapper.leave_one_out_split(
            i, subjects_ids_train
        )

        train_loader = DataLoader(
            train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"])
        if model_name == "lstm_gat":
            model = EEG_LSTM_GAT_Model(
                input_dim=cfg["model"]["input_dim"],
                lstm_hidden_dim=lstm_hidden_dim,
                gat_hidden_dim=gat_hidden_dim,
                output_dim=cfg["model"]["output_dim"],
                gat_heads=gat_heads,
                lstm_layers=lstm_layers,
                fully_connected=fully_connected,
            ).to(device)
        elif model_name == "lstm":
            model = EEG_LSTM_Model(
                input_dim=cfg["model"]["input_dim"],
                lstm_hidden_dim=lstm_hidden_dim,
                output_dim=cfg["model"]["output_dim"],
                lstm_layers=lstm_layers,
            ).to(device)
        elif model_name == "lstm_freeze_gat":
            model = EEG_LSTM_GAT_Model(
                input_dim=cfg["model"]["input_dim"],
                lstm_hidden_dim=lstm_hidden_dim,
                gat_hidden_dim=gat_hidden_dim,
                output_dim=cfg["model"]["output_dim"],
                gat_heads=gat_heads,
                lstm_layers=lstm_layers,
                fully_connected=fully_connected,
            ).to(device)
            model.load_and_freeze_lstm(cfg["model"]["lstm_pth_path"])

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
        save_best_val_score(best_score_path, best_val_score)
        print(f"✅ Saved new best LOOCV avg val_score: {avg_val_score:.4f}")

    return avg_val_score


def hyperparameter_tuning(cfg, dataset_wrapper):
    """Hyperparameter tuning using Optuna.
    Args:
        cfg (dict): Configuration dictionary.
        dataset_wrapper (EEGDatasetWrapper): EEG dataset wrapper object.
    """

    best_score_path = (
        "checkpoints/optuna/best_score.json"  # Path to save the best score
    )
    best_val_score = load_best_val_score(best_score_path)  # Load the best score so far
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Set device to GPU if available, else CPU

    # choose ids for leave-one-out cross-validation
    selected_ids = cfg["data"][
        "hypertuning_subjects_ids"
    ]  # based on data exploration performed prior to this
    print(f"Selected subjects for LOOCV: {selected_ids}")

    os.makedirs("checkpoints/optuna", exist_ok=True)  # make sure the directory exists
    storage_path = "sqlite:///checkpoints/optuna/eeg_study.db"
    study = optuna.create_study(
        study_name="eeg_classifier_optimization",
        direction="maximize",
        storage=storage_path,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            cfg,
            dataset_wrapper,
            selected_ids,
            best_val_score,
            device,
            best_score_path,
        ),
        n_trials=cfg["training"]["n_trials"],
        timeout=11 * 3600,
    )  # Timeout set to 11.5 hours because of Cluster limit
    print("Best trial:")
    print(study.best_trial)
    save_config(cfg, "configs/best_hyperparam_config.yaml", study)
