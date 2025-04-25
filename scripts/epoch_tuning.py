import torch
from torch.utils.data import DataLoader
from training.train import train_model
from models import EEG_LSTM_Model, EEG_LSTM_GAT_Model
from training.losses import BinaryFocalLoss
import yaml


def epoch_tuning(cfg, dataset_wrapper):
    """
    Perform epoch tuning for the model.
    Args:
        cfg (dict): Configuration dictionary containing model and training parameters.
        dataset_wrapper (EEGDatasetWrapper): Wrapper for the EEG dataset.
    Returns:
        float: Best validation score after training.
        list: List of validation scores for each epoch.
    Raises:
        ValueError: If the model name is not supported for epoch tuning.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_ids = dataset_wrapper.get_subject_ids()
    validation_id = cfg["data"][
        "validation_id_epoch_tuning"
    ]  # ID of the subject to be used for validation
    selected_ids.remove(validation_id)
    train_dataset, val_dataset = dataset_wrapper.leave_one_out_split(
        validation_id, selected_ids
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"])

    max_epochs = cfg["training"]["max_epochs"]
    print(max_epochs)
    best_val_score = 0.0
    best_epoch = 0
    epochs_val_scores = []

    if cfg["model"]["name"] == "lstm_gat":
        model = EEG_LSTM_GAT_Model(
            input_dim=cfg["model"]["input_dim"],
            lstm_hidden_dim=cfg["model"]["lstm_hidden_dim"],
            gat_hidden_dim=cfg["model"]["gat_hidden_dim"],
            output_dim=cfg["model"]["output_dim"],
            gat_heads=cfg["model"]["gat_heads"],
            lstm_layers=cfg["model"]["lstm_layers"],
            fully_connected=cfg["model"]["fully_connected"],
        ).to(device)
    elif cfg["model"]["name"] == "lstm":
        model = EEG_LSTM_Model(
            input_dim=cfg["model"]["input_dim"],
            lstm_hidden_dim=cfg["model"]["lstm_hidden_dim"],
            output_dim=cfg["model"]["output_dim"],
            lstm_layers=cfg["model"]["lstm_layers"],
        ).to(device)
    else:
        raise ValueError(
            f"Model {cfg['model']['model_name']} not supported for epoch tuning."
        )

    criterion = BinaryFocalLoss(alpha=cfg["loss"]["alpha"], gamma=cfg["loss"]["gamma"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    # Train 5 epochs at a time before evaluating
    for epoch in range(0, max_epochs, 5):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        val_score = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            epochs=5,  # Train for one epoch
            device=device,
        )
        epochs_val_scores.append(val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch + 5

        print(f"Validation score: {val_score:.4f}")
        print(f"Best validation score: {best_val_score:.4f} at epoch {best_epoch}")

    print(f"Best validation score: {best_val_score:.4f} at epoch {best_epoch}")
    cfg["training"]["best_epoch"] = best_epoch
    cfg["training"]["best_val_score"] = best_val_score
    path = "configs/epoch_tuning_config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    print(f"Updated config saved to {path}")

    print(f"Best validation score: {best_val_score:.4f} at epoch {best_epoch}")
    print(f"epochs eval scores: {epochs_val_scores}")

    return best_val_score, epochs_val_scores
