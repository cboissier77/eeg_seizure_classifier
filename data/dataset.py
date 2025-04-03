from pathlib import Path

import pandas as pd
from data.preprocess import fft_filtering
from torch.utils.data import random_split
from seiz_eeg.dataset import EEGDataset


class EEGDatasetWrapper:
    def __init__(self, data_dir: str):
        """
        Initialize the EEG dataset wrapper.
        Args:
            data_dir (str): Path to the directory containing the EEG data.
        """
        data_root = Path(data_dir)
        clips_tr = pd.read_parquet(data_root / "train/segments.parquet")
        self.dataset_tr = EEGDataset(
            clips_tr,
            signals_root=data_root / "train",
            signal_transform=fft_filtering,
            prefetch=True,
        )

    def train_test_split(self, train_size=0.8, train_subsample=1):
        """Split the dataset into training and testing sets.
        Args:
            train_size (float): Proportion of the dataset to include in the training set.
        Returns:
            Tuple[Dataset, Dataset]: The training and testing datasets.
        """
        train_size = int(len(self.dataset_tr) * train_size)
        self.train_dataset, self.test_dataset = random_split(
            self.dataset_tr, [train_size, len(self.dataset_tr) - train_size]
        )
        #subsample the training dataset
        if train_subsample < 1:
            self.train_dataset = random_split(
                self.train_dataset, [int(len(self.train_dataset) * train_subsample), len(self.train_dataset) - int(len(self.train_dataset) * train_subsample)]
            )[0]
        return self.train_dataset, self.test_dataset
