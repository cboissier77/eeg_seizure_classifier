from pathlib import Path
import pandas as pd
from data.preprocess import downsample, fft_filtering, time_filtering, normalize, acf_coef
from seiz_eeg.dataset import EEGDataset
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import torch
class EEGResidualDataset(torch.utils.data.Dataset):
    def __init__(self, residuals, labels):
        self.residuals = torch.tensor(residuals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.residuals)

    def __getitem__(self, idx):
        return self.residuals[idx], self.labels[idx]



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
        elif preprocessing == "acf":
            self.preprocessing = acf_coef
        else:
            raise ValueError(f"Unknown preprocessing method: {preprocessing}")
        
    # Fit AR model
    def __fit_ar_model(self, signals_train, lags):
        """Fit individual AR models per signal and per electrode, then average coefficients."""
        n_signals, length_signal, n_electrodes = signals_train.shape

        all_coefs = []  # List for each electrode
        for electrode_idx in range(n_electrodes):
            electrode_coefs = []
            for signal_idx in range(n_signals):
                signal = signals_train[signal_idx, :, electrode_idx]  # (length_signal,)
                ar_model = AutoReg(signal, lags=lags, old_names=False, trend='c').fit()
                electrode_coefs.append(ar_model.params)  # (lags + 1,)
            electrode_coefs = np.stack(electrode_coefs, axis=0)  # (n_signals, lags + 1)
            mean_coefs = electrode_coefs.mean(axis=0)  # (lags + 1,)
            all_coefs.append(mean_coefs)
        
        all_coefs = np.stack(all_coefs, axis=0)  # (n_electrodes, lags + 1)
        return all_coefs  # Shape: (19, lags + 1)
    
    def __predict_with_ar_coefs(self, signals, mean_coefs, lags):
        """Predict signals manually using the averaged AR coefficients per electrode."""
        n_signals, length_signal, n_electrodes = signals.shape
        preds = []

        for signal_idx in range(n_signals):
            signal_preds = np.zeros((length_signal - lags, n_electrodes))
            for electrode_idx in range(n_electrodes):
                intercept = mean_coefs[electrode_idx, 0]
                ar_coefs = mean_coefs[electrode_idx, 1:]

                signal_electrode = signals[signal_idx, :, electrode_idx]  # (length_signal,)
                pred = []
                for t in range(lags, length_signal):
                    x = signal_electrode[t-lags:t][::-1]  # last lags values, reversed
                    pred_value = intercept + np.dot(ar_coefs, x)
                    pred.append(pred_value)
                signal_preds[:, electrode_idx] = pred  # Fill electrode predictions
            preds.append(signal_preds)  # Append predicted (length_signal - lag, 19)
        
        preds = np.stack(preds, axis=0)  # (n_signals, length_signal - lags, n_electrodes)
        return preds
    def __get_residuals(self, signals_train, signals_test, mean_coefs, lags):
        """Compute residuals using manually averaged AR model per electrode."""
        # Train residuals
        preds_train = self.__predict_with_ar_coefs(signals_train, mean_coefs, lags)
        residuals_train = signals_train[:, lags:, :] - preds_train  # (n_signals, length-lags, n_electrodes)

        # Test residuals
        preds_test = self.__predict_with_ar_coefs(signals_test, mean_coefs, lags)
        residuals_test = signals_test[:, lags:, :] - preds_test  # (n_signals, length-lags, n_electrodes)

        return residuals_train, residuals_test
    
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

    def leave_one_out_split(self, subject_id_test: str, subjects_ids_train: list, ar_model: bool = False, lag: int = 1):
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
        if ar_model:
            # Get label 0 signals
            signals_train_0 = []
            for i in range(len(train_dataset)):
                signal, label = train_dataset[i]
                if label == 0:
                    signals_train_0.append(signal)
            if len(signals_train_0) == 0:
                raise ValueError("No label 0 samples found for AR fitting.")

            signals_train_0 = np.stack(signals_train_0, axis=0)


            print(f"Fitting AR model with lag {lag} on label 0 signals")
            mean_coefs = self.__fit_ar_model(signals_train_0, lag)

            # Get all train and val signals
            signals_train = []
            labels_train = []
            for i in range(len(train_dataset)):
                signal, label = train_dataset[i]
                signals_train.append(signal)
                labels_train.append(label)
            signals_train = np.stack(signals_train, axis=0)

            signals_val = []
            labels_val = []
            for i in range(len(val_dataset)):
                signal, label = val_dataset[i]
                signals_val.append(signal)
                labels_val.append(label)
            signals_val = np.stack(signals_val, axis=0)

            # Compute residuals
            print(f"Computing residuals for train and val signals")
            residuals_train, residuals_val = self.__get_residuals(signals_train, signals_val, mean_coefs, lag)

            # Replace datasets with residuals
            train_dataset = EEGResidualDataset(residuals_train, labels_train)
            val_dataset = EEGResidualDataset(residuals_val, labels_val)

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
