# evaluate the model
from sklearn.metrics import roc_auc_score, f1_score
import torch
from torch.utils.data import DataLoader
from models import Hyper_GAT_Model
from data.dataset import EEGGraphFeatureDataset
import pandas as pd


def testing(cfg, dataset_wrapper):
    """
    Test the model on the test dataset and generate a Kaggle submission file.
    Args:
        cfg (dict): Configuration dictionary containing model and dataset parameters.
        dataset_wrapper (EEGDatasetWrapper): Wrapper for the EEG dataset.
    """
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cpu")
        # Prepare data splits
    test_dataset = dataset_wrapper.test_dataset()
    test_dataset = EEGGraphFeatureDataset(test_dataset, window_size=cfg["model"]["windows_size"])

    input_dim = test_dataset[0][0].shape[1]

    # Create DataLoaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    # Initialize model, loss, optimizer
    model = Hyper_GAT_Model(
        input_dim=input_dim,
        gat_hidden_dim=cfg["model"]["gat_hidden_dim"],
        output_dim=cfg["model"]["output_dim"],
        gat_heads=cfg["model"]["gat_heads"],
        gat_layers=cfg["model"]["gat_layers"],
    ).to(device)

    # Load the model weights
    model_path = cfg["training"]["best_model_path"]
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()
    # Lists to store sample IDs and predictions
    all_predictions = []
    all_ids = []
    # Disable gradient computation for inference
    with torch.no_grad():
        for batch in test_loader:
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
    submission_df = pd.DataFrame({"id": all_ids, "label": all_predictions})

    date = pd.to_datetime("now").strftime("%Y-%m-%d")
    name = f"submission_hyper_gat_{date}.csv"

    # Save the DataFrame to a CSV file without an index
    submission_df.to_csv(name, index=False)
    print("Kaggle submission file generated: submission.csv")
