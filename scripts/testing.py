# evaluate the model
from sklearn.metrics import roc_auc_score, f1_score
import torch
from torch.utils.data import DataLoader
from models.lstm_gat import EEG_LSTM_GAT_Model
import pandas as pd


def testing(cfg, dataset_wrapper):
    """
    Test the model on the test dataset and generate a Kaggle submission file.
    Args:
        cfg (dict): Configuration dictionary containing model and dataset parameters.
        dataset_wrapper (EEGDatasetWrapper): Wrapper for the EEG dataset.
    """
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["model"]["name"] == "lstm_gat":
        # Load the model
        model = EEG_LSTM_GAT_Model(
            input_dim=cfg["model"]["input_dim"],
            lstm_hidden_dim=cfg["model"]["lstm_hidden_dim"],
            gat_hidden_dim=cfg["model"]["gat_hidden_dim"],
            output_dim=cfg["model"]["output_dim"],
            gat_heads=cfg["model"]["gat_heads"],
            lstm_layers=cfg["model"]["lstm_layers"],
            fully_connected=cfg["model"]["fully_connected"],
        ).to(device)

        # Load the model weights
        model_path = cfg["model"]["best_model_path"]
        model.load_state_dict(torch.load(model_path))
    else:
        raise ValueError(f"Model {cfg['model']['name']} not supported for testing.")

    # Create test dataset
    dataset_te = dataset_wrapper.test_dataset()
    # Create DataLoader for the test dataset
    loader_te = DataLoader(
        dataset_te, batch_size=cfg["training"]["batch_size"], shuffle=False
    )
    # Set the model to evaluation mode
    model.eval()
    # Lists to store sample IDs and predictions
    all_predictions = []
    all_ids = []
    # Disable gradient computation for inference
    with torch.no_grad():
        for batch in loader_te:
            # Assume each batch returns a tuple (x_batch, sample_id)
            # If your dataset does not provide IDs, you can generate them based on the batch index.
            x_batch, x_ids = batch
            # Move the input data to the device (GPU or CPU)
            x_batch = x_batch.float().to(device)

            # Perform the forward pass to get the model's output logits
            logits = model(x_batch)

            # Convert logits to predictions.
            # For binary classification, threshold logits at 0 (adjust this if you use softmax or multi-class).
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            predictions = (probs >= 0.5).astype(int)

            # Append predictions and corresponding IDs to the lists
            all_predictions.extend(predictions.flatten().tolist())
            all_ids.extend(list(x_ids))

    # Create a DataFrame for Kaggle submission with the required format: "id,label"
    # remove underscore that are not followed by an underscore
    all_ids = [str(i).replace("__", "$$") for i in all_ids]
    all_ids = [i.replace("_", "") for i in all_ids]
    all_ids = [i.replace("$$", "_") for i in all_ids]
    print(all_ids)
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv("submission_seed1.csv", index=False)
    print("Kaggle submission file generated: submission.csv")
