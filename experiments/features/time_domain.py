"""Time-domain feature extraction.

Features: RMS, mean, std, kurtosis, skewness, crest factor, peak-to-peak.
Used in Stage A (Alarm Normalization) of the CL-LLMOps framework.
"""

import numpy as np
from scipy.stats import kurtosis, skew


def compute_time_features(signal):
    """Extract time-domain features from a 1D signal.

    Args:
        signal: 1D numpy array of shape [window_size].

    Returns:
        dict of feature_name -> float.
    """
    features = {}
    features["mean"] = float(np.mean(signal))
    features["std"] = float(np.std(signal))
    features["rms"] = float(np.sqrt(np.mean(signal ** 2)))
    features["peak"] = float(np.max(np.abs(signal)))
    features["peak_to_peak"] = float(np.max(signal) - np.min(signal))
    features["kurtosis"] = float(kurtosis(signal, fisher=True))
    features["skewness"] = float(skew(signal))

    rms = features["rms"]
    features["crest_factor"] = features["peak"] / rms if rms > 1e-10 else 0.0

    # Shape factor, impulse factor
    mean_abs = float(np.mean(np.abs(signal)))
    features["shape_factor"] = rms / mean_abs if mean_abs > 1e-10 else 0.0
    features["impulse_factor"] = features["peak"] / mean_abs if mean_abs > 1e-10 else 0.0

    return features


def compute_rolling_stats(signal, window_size=32):
    """Compute rolling mean and std for trend detection.

    Args:
        signal: 1D numpy array.
        window_size: rolling window length.

    Returns:
        rolling_mean, rolling_std as 1D arrays.
    """
    n = len(signal)
    if n < window_size:
        return np.array([np.mean(signal)]), np.array([np.std(signal)])

    # Use cumsum for efficient computation
    cumsum = np.cumsum(np.insert(signal, 0, 0))
    rolling_mean = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    cumsum_sq = np.cumsum(np.insert(signal ** 2, 0, 0))
    rolling_var = (cumsum_sq[window_size:] - cumsum_sq[:-window_size]) / window_size - rolling_mean ** 2
    rolling_std = np.sqrt(np.maximum(rolling_var, 0))

    return rolling_mean, rolling_std
