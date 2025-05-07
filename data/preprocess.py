from scipy import signal
import numpy as np
from statsmodels.tsa.stattools import acf


bp_filter = signal.butter(4, (0.5, 30), btype="bandpass", output="sos", fs=250)


def time_filtering(x: np.ndarray) -> np.ndarray:
    """Filter signal in the time domain"""
    return signal.sosfiltfilt(bp_filter, x, axis=0).copy()


def fft_filtering(x: np.ndarray) -> np.ndarray:
    """Compute FFT and only keep"""
    x = np.abs(np.fft.fft(x, axis=0))
    x = np.log(np.where(x > 1e-8, x, 1e-8))

    win_len = x.shape[0]
    # Only frequencies b/w 0.5 and 30Hz
    return x[int(0.5 * win_len // 250) : 30 * win_len // 250]

def downsample(x: np.ndarray) -> np.ndarray:
    """Downsample the signal to 300 samples"""
    downsampled_signal = signal.resample(x, 300, axis=0)
    
    return downsampled_signal

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize min-max"""
    range_val = np.max(x) - np.min(x)
    if range_val == 0:
        x = np.zeros_like(x)  # Handle constant arrays
    else:
        x = (x - np.min(x)) / range_val
    return x

def acf_coef(x: np.ndarray, n_lags: int = 100) -> np.ndarray:
    """Compute the ACF coefficients for each EEG electrode signal"""
    x = downsample(x)  # shape (time, 19)
    acf_values = np.zeros((n_lags + 1, x.shape[1]))
    
    for i in range(x.shape[1]):
        xi = x[:, i]
        if np.std(xi) == 0:
            # Constant signal → ACF = [1, 0, 0, ..., 0]
            acf_values[:, i] = np.concatenate([[1], np.zeros(n_lags)])
        else:
            acf_values[:, i] = acf(xi, nlags=n_lags, fft=True)
    
    return acf_values
