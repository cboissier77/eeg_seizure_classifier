from pathlib import Path

import pandas as pd
from data.preprocess import fft_filtering
from torch.utils.data import random_split
from seiz_eeg.dataset import EEGDataset
from sklearn.model_selection import train_test_split


class EEGDatasetWrapper:
    def __init__(self, data_dir: str):
        """
        Initialize the EEG dataset wrapper.
        Args:
            data_dir (str): Path to the directory containing the EEG data.
        """
        self.data_root = Path(data_dir)
        self.clips_tr = pd.read_parquet(self.data_root / "train/segments.parquet")

    def train_test_split(self, train_size=0.8, train_subsample=1):
        """Split the dataset into training and testing sets.
        Args:
            train_size (float): Proportion of the dataset to include in the training set.
        Returns:
            Tuple[Dataset, Dataset]: The training and testing datasets.
        """
        ids = self.clips_tr["signals_path"].astype(str).tolist()
        ids = list(map(lambda x: x[x.find("_") + 1 : x.find("_") + 5], ids))
        self.clips_tr["subject_id"] = ids
        unique_subjects = self.clips_tr["subject_id"].unique()
        print(unique_subjects)
        train_subjects, val_subjects = train_test_split(
            unique_subjects, test_size=1 - train_size, random_state=42
        )
        self.clips_val = self.clips_tr[
            self.clips_tr["subject_id"].isin(val_subjects)
        ].reset_index(drop=True)
        self.clips_tr = self.clips_tr[
            self.clips_tr["subject_id"].isin(train_subjects)
        ].reset_index(drop=True)
        if train_subsample < 1:
            self.clips_tr = self.clips_tr.sample(
                frac=train_subsample, random_state=42
            ).reset_index(drop=True)

        print(
            f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}"
        )
        print(
            f"Train samples: {len(self.clips_tr)}, Val samples: {len(self.clips_val)}"
        )
        self.train_dataset = EEGDataset(
            self.clips_tr,
            signals_root=self.data_root / "train",
            signal_transform=fft_filtering,
            prefetch=True,
        )
        self.test_dataset = EEGDataset(
            self.clips_val,
            signals_root=self.data_root / "train",
            signal_transform=fft_filtering,
            prefetch=True,
        )

        return self.train_dataset, self.test_dataset
