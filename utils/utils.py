import argparse
import json
import yaml
import os
import copy
import optuna


def load_best_val_score(path: str):
    """Load the best validation score from a JSON file. Enables resuming training when interrupted.
    Args:
        path (str): Path to the JSON file.
    Returns:
        float: The best validation score. Returns -inf if the file is not found or contains invalid data.
    """
    try:
        with open(path, "r") as f:
            return float(json.load(f)["best_val_score"])
    except (FileNotFoundError, ValueError):
        return float("-inf")


def save_best_val_score(path: str, score: float):
    """Save the best validation score to a JSON file. Save the score to resume training when interrupted.
    Args:
        path (str): Path to the JSON file.
        score (float): The best validation score.
    """
    with open(path, "w") as f:
        json.dump({"best_val_score": score}, f)


def parse_args():
    """Parse command line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lstm_gat.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "hyperparameter_tuning",
            "testing",
        ],
    )
    return parser.parse_args()


def load_config(path: str):
    """Load configuration from a YAML file.
    Args:
        path (str): Path to the YAML file.
    Returns:
        dict: Configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str, study: optuna.Study):
    """Save the best configuration to a YAML file.
    Args:
        cfg (dict): Configuration dictionary.
        path (str): Path to the YAML file.
        study (optuna.Study): Optuna study object.
    """
    # Save best config
    best_params = study.best_trial.params
    best_model_cfg = copy.deepcopy(cfg["model"])
    best_training_cfg = copy.deepcopy(cfg["training"])
    best_testing_cfg = copy.deepcopy(cfg["testing"])
    best_seed_cfg = copy.deepcopy(cfg["seed"])
    best_data_cfg = copy.deepcopy(cfg["data"])

    if cfg["model"]["name"] == "hyper_gat":
        best_model_cfg["gat_hidden_dim"] = best_params["gat_hidden_dim"]
        best_model_cfg["gat_heads"] = best_params["gat_heads"]
        best_model_cfg["gat_layers"] = best_params["gat_layers"]
        best_model_cfg["windows_size"] = best_params["windows_size"]
    best_config = {
        "model": best_model_cfg,
        "training": best_training_cfg,
        "loss": best_testing_cfg,
        "seed": best_seed_cfg,
        "data": best_data_cfg,
    }

    os.makedirs("configs", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(best_config, f)
