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
        ids = self.clips_tr["signals_path"].astype(str).tolist()
        ids = list(map(lambda x: x[x.find("_") + 1 : x.find("_") + 5], ids))
        self.clips_tr["subject_id"] = ids

    def num_subjects(self):
        """Get the number of unique subjects in the dataset.
        Returns:
            int: The number of unique subjects.
        """
        return len(self.clips_tr["subject_id"].unique())

    def get_subject_ids(self):
        """Get the unique subject IDs in the dataset.
        Returns:
            List[str]: A list of unique subject IDs.
        """
        return self.clips_tr["subject_id"].unique().tolist()

    def leave_one_out_split(self, subject_id_test: str, subjects_ids_train: list):
        """Split the dataset into training and validation sets for leave-one-out cross-validation.
        Args:
            subject_id_test (str): The subject ID to be used for validation.
            subjects_ids_train (list): The list of subject IDs to be used for training.
        Returns:
            Tuple[Dataset, Dataset]: The training and validation datasets.
        Raises:
            ValueError: If the test subject ID is in the training set.
        """
        if subject_id_test in subjects_ids_train:
            raise ValueError(
                f"Test subject {subject_id_test} is in the training set {subjects_ids_train}"
            )

        train_dataset = self.clips_tr[
            self.clips_tr["subject_id"].isin(subjects_ids_train)
        ]
        val_dataset = self.clips_tr[self.clips_tr["subject_id"] == subject_id_test]

        train_dataset = EEGDataset(
            train_dataset,
            signals_root=self.data_root / "train",
            signal_transform=fft_filtering,
            prefetch=True,
        )
        val_dataset = EEGDataset(
            val_dataset,
            signals_root=self.data_root / "train",
            signal_transform=fft_filtering,
            prefetch=True,
        )

        return train_dataset, val_dataset
