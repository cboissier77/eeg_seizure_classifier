import torch
from torch.utils.data import DataLoader
from training.train import train_model
from models.lstm_gat import EEG_LSTM_GAT_Model, EEG_LSTM_Model
from training.losses import BinaryFocalLoss


def final_training(cfg, dataset_wrapper):
    """
    Perform final training of the model using the best hyperparameters.
    Args:
        cfg (dict): Configuration dictionary containing model and training parameters.
        dataset_wrapper (EEGDatasetWrapper): Wrapper for the EEG dataset.
    Returns:
        None
    Raises:
        ValueError: If the model name is not supported for final training.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = dataset_wrapper.all_training_split()
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    epochs = cfg["training"]["best_epoch"]
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
            fully_connected=cfg["model"]["fully_connected"],
        ).to(device)
    else:
        raise ValueError(
            f"Model {cfg['model']['model_name']} not supported for epoch tuning."
        )

    criterion = BinaryFocalLoss(
        alpha=cfg["loss"]["alpha"], gamma=cfg["loss"]["gamma"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    train_model(
        model, train_loader, None, criterion, optimizer, epochs=epochs, device=device
    )

    print(
        f"Final training completed. Predicted validation score: {cfg['training']['best_val_score']}"
    )
    # Save the model
    torch.save(model.state_dict(), cfg["training"]["best_model_path"])
    print(f"Model saved to {cfg['training']['model_save_path']}")
    return
