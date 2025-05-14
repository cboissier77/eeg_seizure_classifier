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
from models import Hyper_GAT_Model
from training.losses import BinaryFocalLoss
from data.dataset import EEGGraphFeatureDataset


def objective(trial, cfg, data_wrapper, best_val_score, device, best_score_path):
    """Objective function for Optuna hyperparameter tuning.
    Args:
        trial (optuna.Trial): Optuna trial object.
        cfg (dict): Configuration dictionary.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        best_val_score (float): Best validation score so far.
        device (torch.device): Device to use for training.
        best_score_path (str): Path to save the best score.
    Returns:
        float: Validation score for the current trial.
    """
    ############################################################################################################
    # Define hyperparameters to optimize

    model_name = cfg["model"]["name"]

    if model_name == "hyper_gat":

        lr = cfg["training"]["lr"]
        gat_hidden_dim = trial.suggest_int("gat_hidden_dim", 16, 128)
        output_dim = cfg["model"]["output_dim"]
        gat_heads = trial.suggest_int("gat_heads", 1, 8)
        gat_layers = trial.suggest_int("gat_layers", 1, 4)
        epochs = cfg["training"]["epochs"]  # Fixed epochs for LOOCV
        lr = cfg["training"]["lr"]
        alpha = trial.suggest_float("alpha", 0.6, 0.9)
        gamma = trial.suggest_float("gamma", 0.5, 5.0)
        windows_size = trial.suggest_categorical(
            "windows_size", [250, 500, 1000, 1500, 2000, 3000]
        )

    else:
        raise ValueError(f"Model {model_name} not supported for hyperparameter tuning.")
    ############################################################################################################
    selected_ids = data_wrapper.get_subject_ids()
    val_id = cfg["data"]["validation_id_epoch_tuning"]
    selected_ids.remove(val_id)  # Remove the current validation ID
    train_dataset, val_dataset = data_wrapper.leave_one_out_split(val_id, selected_ids)
    # Create datasets
    train_dataset = EEGGraphFeatureDataset(train_dataset, window_size=windows_size)
    val_dataset = EEGGraphFeatureDataset(val_dataset, window_size=windows_size)
    input_dim = train_dataset[0][0].shape[1]  # Get input dimension from the first sample

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    # Create model
    model = Hyper_GAT_Model(
        input_dim=input_dim,
        gat_hidden_dim=gat_hidden_dim,
        output_dim=output_dim,
        gat_heads=gat_heads,
        gat_layers=gat_layers,
    ).to(device)

    criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_score = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs, device=device
    )
    ############################################################################################################
    # Calculate average validation score

    if val_score > best_val_score:
        best_val_score = val_score
        save_best_val_score(best_score_path, best_val_score)
        print(f"✅ Saved new best val_score: {val_score:.4f}")
        # save model checkpoint
        torch.save(
            model.state_dict(),
            f"checkpoints/optuna/best_model_{val_id}.pth",
        )
        # print param values
        print(f"gat_hidden_dim: {gat_hidden_dim}")
        print(f"gat_heads: {gat_heads}")
        print(f"gat_layers: {gat_layers}")
        print(f"alpha: {alpha}")
        print(f"gamma: {gamma}")
        print(f"windows_size: {windows_size}")
        print(f"val_score: {val_score:.4f}")

    return val_score


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
