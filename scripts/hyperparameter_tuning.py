import torch
import torch.nn as nn
from tqdm import tqdm
import optuna
import numpy as np
from torch.utils.data import DataLoader
from training.eval import evaluate_model
import os
from utils.utils import (
    load_best_val_score,
    save_best_val_score,
    save_config,
)
from models import Hyper_GAT_Model, EEG_LSTM_GAT_Model
from data.dataset import EEGGraphFeatureDataset

global best_val_score


def balance_dataset(train_dataset):
    """Balance the dataset by downsampling the majority class."""
    labels = train_dataset.get_label_array()
    labels = np.array(labels)
    train_data_label_1 = []
    train_data_label_0 = []
    for i, (train_data, label) in enumerate(train_dataset):
        if label == 1:
            train_data_label_1.append((train_data, label))
        else:
            train_data_label_0.append((train_data, label))
    train_data_label_1 = np.array(train_data_label_1, dtype=object)
    train_data_label_0 = np.array(train_data_label_0, dtype=object)
    print(f"Number of samples with label 1: {len(train_data_label_1)}")
    print(f"Number of samples with label 0: {len(train_data_label_0)}")
    num_samples = min(len(train_data_label_1), len(train_data_label_0))
    train_data_label_0 = train_data_label_0[
        np.random.choice(len(train_data_label_0), num_samples, replace=False)
    ]
    train_data_label_1 = train_data_label_1[
        np.random.choice(len(train_data_label_1), num_samples, replace=False)
    ]

    print(f"Number of samples with label 1 after resampling: {len(train_data_label_1)}")
    print(f"Number of samples with label 0 after resampling: {len(train_data_label_0)}")
    train_dataset = np.concatenate((train_data_label_0, train_data_label_1), axis=0)
    np.random.shuffle(train_dataset)
    return train_dataset


def objective(trial, cfg, data_wrapper, device, best_score_path):
    global best_val_score
    model_name = cfg["model"]["name"]
    if model_name != "hyper_gat" and model_name != "lstm_gat":
        raise ValueError(f"Model {model_name} not supported for hyperparameter tuning.")

    # Suggest hyperparameters
    lr = cfg["training"]["lr"]
    if model_name == "hyper_gat":
        gat_hidden_dim = trial.suggest_int("gat_hidden_dim", 16, 128)
        gat_heads = trial.suggest_int("gat_heads", 1, 8)
        gat_layers = trial.suggest_int("gat_layers", 1, 4)
        windows_size = trial.suggest_categorical(
            "windows_size", [250, 500, 1000, 1500, 2000, 3000]
        )
    elif model_name == "lstm_gat":
        lstm_hidden_dim = trial.suggest_int("lstm_hidden_dim", 16, 128)
        gat_hidden_dim = trial.suggest_int("gat_hidden_dim", 16, 128)
        gat_heads= trial.suggest_int("gat_heads", 1, 8)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
        fully_connected=True
    epochs = cfg["training"]["epochs"]

    # Split dataset
    selected_ids = data_wrapper.get_subject_ids()
    val_id = cfg["data"]["validation_id"]
    selected_ids.remove(val_id)
    train_dataset, val_dataset = data_wrapper.leave_one_out_split(val_id, selected_ids)

    train_dataset = balance_dataset(train_dataset)

    if model_name == "hyper_gat":
        train_dataset = EEGGraphFeatureDataset(train_dataset, window_size=windows_size)
        val_dataset = EEGGraphFeatureDataset(val_dataset, window_size=windows_size)
    elif model_name == "lstm_gat":
        train_dataset = EEGGraphFeatureDataset(
            train_dataset, window_size=0, already_preprocessed=True
        )
        val_dataset = EEGGraphFeatureDataset(
            val_dataset, window_size=0, already_preprocessed=True
        )


    input_dim = train_dataset[0][0].shape[1]

    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False
    )

    # Model, optimizer, scheduler
    if model_name == "hyper_gat":
        model = Hyper_GAT_Model(
            input_dim, gat_hidden_dim, cfg["model"]["output_dim"], gat_heads, gat_layers
        ).to(device)
    elif model_name == "lstm_gat":
        model = EEG_LSTM_GAT_Model(
            1, # input_dim is 1 for LSTM input
            lstm_hidden_dim,
            gat_hidden_dim,
            cfg["model"]["output_dim"],
            gat_heads,
            lstm_layers,
            fully_connected=fully_connected,
        ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5, verbose=True
    )

    # Training
    best_value_optuna_session = 0
    print("🧠 Starting training...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"Epoch {epoch+1} - Training Loss: {running_loss / len(train_loader):.4f}"
        )

        # Validation and scheduler update
        model.eval()
        val_macro_f1 = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1} - Validation Macro F1: {val_macro_f1:.4f}")
        scheduler.step(val_macro_f1)

        # Save best model
        if val_macro_f1 > best_val_score:
            best_val_score = val_macro_f1
            save_best_val_score(best_score_path, best_val_score)
            torch.save(
                model.state_dict(), f"checkpoints/optuna/best_model_{best_val_score}.pth"
            )
            print(f"✅ Saved new best val_score: {val_macro_f1:.4f}")
            print(
                f"gat_hidden_dim: {gat_hidden_dim}, gat_heads: {gat_heads}, gat_layers: {gat_layers}, windows_size: {windows_size}"
            )
            # save config
            if model_name == "hyper_gat":
                to_save = [
                   "gat_hidden_dim",
                   "gat_heads",
                    "gat_layers",
                    "windows_size",
                ]
            elif model_name == "lstm_gat":
                to_save = [
                    "lstm_hidden_dim",
                    "gat_hidden_dim",
                    "gat_heads",
                    "lstm_layers",
                ]
            path = f"checkpoints/optuna/best_model_config_{best_val_score}.txt"
            # dump in csv
            with open(path, "w") as f:
                for key in to_save:
                    f.write(f"{key}: {locals()[key]}\n")

        if best_value_optuna_session < val_macro_f1:
            best_value_optuna_session = val_macro_f1

    return best_value_optuna_session


def hyperparameter_tuning(cfg, dataset_wrapper):
    """Hyperparameter tuning using Optuna.
    Args:
        cfg (dict): Configuration dictionary.
        dataset_wrapper (EEGDatasetWrapper): EEG dataset wrapper object.
    """
    global best_val_score   
    best_score_path = (
        "checkpoints/optuna/best_score.json"  # Path to save the best score
    )
    best_val_score = load_best_val_score(best_score_path)  # Load the best score so far
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Set device to GPU if available, else CPU

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
            device,
            best_score_path,
        ),
        n_trials=cfg["training"]["n_trials"],
        timeout=11 * 3600,
    )  # Timeout set to 11.5 hours because of Cluster limit
    print("Best trial:")
    print(study.best_trial)
    save_config(cfg, "configs/best_hyperparam_config.yaml", study)
