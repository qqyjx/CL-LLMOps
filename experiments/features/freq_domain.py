"""Frequency-domain feature extraction.

Features: FFT peaks, spectral centroid, spectral bandwidth, THD,
          envelope spectrum (for bearing defect frequencies).
Used in Stage A (Alarm Normalization) and V1 (Evidence Retrieval).
"""

import numpy as np
from scipy.signal import hilbert


def compute_fft_features(signal, fs, n_peaks=5):
    """Extract FFT-based features from a 1D signal.

    Args:
        signal: 1D numpy array.
        fs: sampling frequency (Hz).
        n_peaks: number of dominant peaks to extract.

    Returns:
        dict of feature_name -> float or list.
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    features = {}

    # Top-N peaks
    peak_indices = np.argsort(fft_mag[1:])[-n_peaks:][::-1] + 1  # skip DC
    features["fft_peak_freqs"] = freqs[peak_indices].tolist()
    features["fft_peak_mags"] = fft_mag[peak_indices].tolist()

    # Spectral centroid
    total_energy = np.sum(fft_mag[1:])
    if total_energy > 1e-10:
        features["spectral_centroid"] = float(np.sum(freqs[1:] * fft_mag[1:]) / total_energy)
        features["spectral_bandwidth"] = float(
            np.sqrt(np.sum(((freqs[1:] - features["spectral_centroid"]) ** 2) * fft_mag[1:]) / total_energy)
        )
    else:
        features["spectral_centroid"] = 0.0
        features["spectral_bandwidth"] = 0.0

    return features


def compute_thd(signal, fs, fundamental_freq=50.0, n_harmonics=10):
    """Compute Total Harmonic Distortion.

    Args:
        signal: 1D numpy array.
        fs: sampling frequency.
        fundamental_freq: fundamental frequency (50 Hz for power systems).
        n_harmonics: number of harmonics to include.

    Returns:
        THD as a float (ratio, not percentage).
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    freq_resolution = fs / n

    def get_amplitude(target_freq):
        idx = int(round(target_freq / freq_resolution))
        if 0 <= idx < len(fft_mag):
            # Use peak within +/- 2 bins for tolerance
            lo = max(0, idx - 2)
            hi = min(len(fft_mag), idx + 3)
            return float(np.max(fft_mag[lo:hi]))
        return 0.0

    fundamental_amp = get_amplitude(fundamental_freq)
    if fundamental_amp < 1e-10:
        return 0.0

    harmonic_sum_sq = sum(
        get_amplitude(fundamental_freq * h) ** 2
        for h in range(2, n_harmonics + 1)
    )
    return float(np.sqrt(harmonic_sum_sq) / fundamental_amp)


def compute_envelope_spectrum(signal, fs):
    """Compute envelope spectrum via Hilbert transform.

    Used for extracting bearing defect frequencies (BPFI, BPFO, BSF, FTF).

    Args:
        signal: 1D numpy array (vibration signal).
        fs: sampling frequency.

    Returns:
        envelope_freqs, envelope_magnitudes as 1D arrays.
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    envelope -= np.mean(envelope)  # remove DC

    n = len(envelope)
    fft_vals = np.fft.rfft(envelope)
    fft_mag = np.abs(fft_vals) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    return freqs, fft_mag


def check_defect_frequency(envelope_freqs, envelope_mag, target_freq, tolerance_hz=5.0):
    """Check if a characteristic defect frequency is present in the envelope spectrum.

    Args:
        envelope_freqs: frequency array from envelope spectrum.
        envelope_mag: magnitude array from envelope spectrum.
        target_freq: expected defect frequency (Hz).
        tolerance_hz: frequency tolerance for peak detection.

    Returns:
        (is_present: bool, peak_amplitude: float, peak_freq: float)
    """
    mask = np.abs(envelope_freqs - target_freq) <= tolerance_hz
    if not mask.any():
        return False, 0.0, 0.0

    local_mag = envelope_mag[mask]
    local_freq = envelope_freqs[mask]
    peak_idx = np.argmax(local_mag)

    # Check if peak is significant (above 2x median of full spectrum)
    threshold = 2.0 * np.median(envelope_mag[envelope_mag > 0])
    is_present = local_mag[peak_idx] > threshold

    return bool(is_present), float(local_mag[peak_idx]), float(local_freq[peak_idx])
