import torch
from utils.utils import (
    parse_args,
    load_config,
)
from data.dataset import EEGDatasetWrapper
import random
from scripts import (
    hyperparameter_tuning,
    epoch_tuning,
    final_training,
    testing,
    push_to_hub,
)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 42))
    random.seed(cfg.get("seed", 42))

    dataset_wrapper = EEGDatasetWrapper(cfg["data"]["data_dir"], cfg["data"]["preprocessing"])
    if args.mode == "hyperparameter_tuning":
        print("Hyperparameter tuning...")
        hyperparameter_tuning(cfg, dataset_wrapper)
    elif args.mode == "epoch_tuning":
        print("Epoch tuning...")
        epoch_tuning(cfg, dataset_wrapper)
    elif args.mode == "final_training":
        print("Final training...")
        final_training(cfg, dataset_wrapper)
    elif args.mode == "testing":
        print("Testing...")
        testing(cfg, dataset_wrapper)
    elif args.mode == "push_to_hub":
        print("Pushing to hub...")
        push_to_hub(cfg, dataset_wrapper)
    else:
        print("Invalid mode selected. Please choose a valid mode.")
        exit(1)


if __name__ == "__main__":
    main()
