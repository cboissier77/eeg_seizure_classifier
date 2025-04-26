from scipy import signal
import numpy as np


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
