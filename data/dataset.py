from pathlib import Path

import pandas as pd
from data.preprocess import downsample, fft_filtering, time_filtering, normalize
from seiz_eeg.dataset import EEGDataset


class EEGDatasetWrapper:
    def __init__(self, data_dir: str, preprocessing:str = "downsample"):
        """
        Initialize the EEG dataset wrapper.
        Args:
            data_dir (str): Path to the directory containing the EEG data.
            preprocessing (str): Preprocessing method to apply to the data. Options are:
                - "downsample": Downsample the signal to 300 samples.
                - "fft_filtering": Apply FFT filtering to the signal.
                - "time_filtering": Apply time filtering to the signal.
                - "raw": No preprocessing.
        """
        self.data_root = Path(data_dir)
        self.clips_tr = pd.read_parquet(self.data_root / "train/segments.parquet")
        self.clips_te = pd.read_parquet(self.data_root / "test/segments.parquet")
        ids = self.clips_tr["signals_path"].astype(str).tolist()
        ids = list(map(lambda x: x[x.find("_") + 1 : x.find("_") + 5], ids))
        self.clips_tr["subject_id"] = ids
        # switch case for preprocessing
        if preprocessing == "downsample":
            self.preprocessing = downsample
        elif preprocessing == "fft_filtering":
            self.preprocessing = fft_filtering
        elif preprocessing == "time_filtering":
            self.preprocessing = time_filtering
        elif preprocessing == "raw":
            self.preprocessing = None
        elif preprocessing == "normalize":
            self.preprocessing = normalize
        else:
            raise ValueError(f"Unknown preprocessing method: {preprocessing}")

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
            signal_transform=self.preprocessing,
            prefetch=True,
        )
        val_dataset = EEGDataset(
            val_dataset,
            signals_root=self.data_root / "train",
            signal_transform=self.preprocessing,
            prefetch=True,
        )

        return train_dataset, val_dataset
    
    def all_training_split(self):
        """Split the dataset into training and validation sets for all training.
        Returns:
            Tuple[Dataset, Dataset]: The training and validation datasets.
        """
        train_dataset = self.clips_tr

        train_dataset = EEGDataset(
            train_dataset,
            signals_root=self.data_root / "train",
            signal_transform=self.preprocessing,
            prefetch=True,
        )

        return train_dataset, None
    
    def test_dataset(self):
        """Create the test dataset.
        Returns:
            Dataset: The test dataset.
        """
        test_dataset = EEGDataset(
            self.clips_te,
            signals_root=self.data_root / "test",
            signal_transform=self.preprocessing,
            prefetch=True,
            return_id=True,
        )

        return test_dataset
