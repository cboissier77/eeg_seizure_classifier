from pathlib import Path

import pandas as pd
from data.preprocess import downsample, fft_filtering, time_filtering, normalize, normalize_and_downsample
from seiz_eeg.dataset import EEGDataset
import torch
import numpy as np


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
        elif preprocessing == "normalize_and_downsample":
            self.preprocessing = normalize_and_downsample
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

class EEGGraphFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_dataset, window_size=100):
        self.eeg_dataset = eeg_dataset
        self.window_size = window_size

    def __len__(self):
        return len(self.eeg_dataset)

    def __getitem__(self, idx):
        signal, label = self.eeg_dataset[idx]  # signal shape: (seq_len, num_electrodes)
        features = self.extract_graph_features(signal)  # (num_windows, num_electrodes, fft_dim)
        return features, label

    def extract_graph_features(self, signal):
        """
        Args:
            signal: numpy array of shape (seq_len, num_electrodes)
        Returns:
            Tensor of shape (num_windows, num_freqs, num_electrodes)
        """
        seq_len, num_electrodes = signal.shape
        fs = 250  # Hz
        step = int(self.window_size)
        windows = []

        for start in range(0, seq_len - self.window_size + 1, step):
            windows.append(signal[start:start+self.window_size])

        windows = np.stack(windows)  # (num_windows, window_size, num_electrodes)
        
        # FFT along time axis
        fft_windows = np.fft.fft(windows, axis=1)
        fft_windows = np.abs(fft_windows)
        fft_windows = np.log(np.maximum(fft_windows, 1e-8))  # Log power

        # Frequency selection
        freqs = np.fft.fftfreq(self.window_size, d=1/fs)
        pos_mask = (freqs >= 0.5) & (freqs <= 30)
        fft_windows = fft_windows[:, pos_mask, :]  # (num_windows, num_freqs, num_electrodes)

        return torch.from_numpy(fft_windows).float()