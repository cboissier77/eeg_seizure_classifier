import torch
from torch.utils.data import DataLoader
from models import Hyper_GAT_Model, EEG_LSTM_GAT_Model
from data.dataset import EEGGraphFeatureDataset
import pandas as pd


def testing(cfg, dataset_wrapper):
    """
    Test the model on the test dataset and generate a Kaggle submission file.
    Args:
        cfg (dict): Configuration dictionary containing model and dataset parameters.
        dataset_wrapper (EEGDatasetWrapper): Wrapper for the EEG dataset.
    """
    model_name = cfg["model"]["name"]
    device = torch.device("cpu")
    test_dataset = dataset_wrapper.test_dataset()
    if model_name != "hyper_gat":
        test_dataset = EEGGraphFeatureDataset(
            test_dataset, window_size=cfg["model"]["windows_size"]
        )
    else:
        test_dataset = EEGGraphFeatureDataset(
            test_dataset, window_size=0, already_preprocessed=True
        )

    input_dim = test_dataset[0][0].shape[1]

    # Create DataLoaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    if model_name == "hyper_gat":
        model = Hyper_GAT_Model(
            input_dim=input_dim,
            gat_hidden_dim=cfg["model"]["gat_hidden_dim"],
            output_dim=cfg["model"]["output_dim"],
            gat_heads=cfg["model"]["gat_heads"],
            gat_layers=cfg["model"]["gat_layers"],
        ).to(device)
    elif model_name == "lstm_gat":       
        model = EEG_LSTM_GAT_Model(
            input_dim=1,  # input_dim is 1 for LSTM input
            lstm_hidden_dim=cfg["model"]["lstm_hidden_dim"],
            gat_hidden_dim=cfg["model"]["gat_hidden_dim"],
            output_dim=cfg["model"]["output_dim"],
            gat_heads=cfg["model"]["gat_heads"],
            lstm_layers=cfg["model"]["lstm_layers"],
            fully_connected=cfg["model"].get("fully_connected", True),
        ).to(device)

    # Load the model weights
    model_path = cfg["testing"]["best_model_path"]
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    all_predictions = []
    all_ids = []
    with torch.no_grad():
        for batch in test_loader:
            x_batch, x_ids = batch
            x_batch = x_batch.float().to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            predictions = (probs >= 0.5).astype(int)
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    # remove underscore that are not followed by an underscore
    all_ids = [str(i).replace("__", "$$") for i in all_ids]
    all_ids = [i.replace("_", "") for i in all_ids]
    all_ids = [i.replace("$$", "_") for i in all_ids]
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    date = pd.to_datetime("now").strftime("%Y-%m-%d")
    if model_name == "hyper_gat":
        name = f"submission_hyper_gat_{date}.csv"
    elif model_name == "lstm_gat":
        name = f"submission_lstm_gat_{date}.csv"

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv(name, index=False)
    print("Kaggle submission file generated")
