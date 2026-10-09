"""Unified feature extraction pipeline.

Combines time-domain and frequency-domain features into a single feature vector
for each dataset. Corresponds to Stage A (Alarm Normalization) in the paper.
"""

import numpy as np
from .time_domain import compute_time_features, compute_rolling_stats
from .freq_domain import compute_fft_features, compute_thd, compute_envelope_spectrum, check_defect_frequency

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASETS
from data.cwru_dataset import compute_defect_freqs


class FeatureExtractor:
    """Extract structured features from raw signals for each dataset.

    This class generates the evidence pack P_t described in Eq.(1) of the paper.
    """

    def __init__(self, dataset_name):
        assert dataset_name in DATASETS, f"Unknown dataset: {dataset_name}"
        self.dataset_name = dataset_name
        self.cfg = DATASETS[dataset_name]

    def extract(self, signal):
        """Extract features from a single sample.

        Args:
            signal: numpy array of shape [channels, window_size].

        Returns:
            feature_dict: dict with all extracted features.
            feature_vector: 1D numpy array for ML models.
        """
        if self.dataset_name == "pq":
            return self._extract_pq(signal)
        elif self.dataset_name == "cwru":
            return self._extract_cwru(signal)
        elif self.dataset_name == "care":
            return self._extract_care(signal)

    def extract_batch(self, signals):
        """Extract features for a batch of samples.

        Args:
            signals: numpy array of shape [N, channels, window_size].

        Returns:
            feature_vectors: numpy array of shape [N, n_features].
        """
        vectors = []
        for signal in signals:
            _, vec = self.extract(signal)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    def _extract_pq(self, signal):
        """PQ features: time stats + FFT + THD per channel."""
        fs = self.cfg["sample_rate_hz"]
        all_features = {}
        vector_parts = []

        for ch_idx in range(signal.shape[0]):
            ch = signal[ch_idx]
            prefix = f"ch{ch_idx}"

            # Time-domain
            time_feat = compute_time_features(ch)
            for k, v in time_feat.items():
                all_features[f"{prefix}_{k}"] = v
            vector_parts.extend([time_feat["rms"], time_feat["std"], time_feat["kurtosis"],
                                 time_feat["crest_factor"], time_feat["skewness"]])

            # Frequency-domain
            fft_feat = compute_fft_features(ch, fs, n_peaks=3)
            all_features[f"{prefix}_fft"] = fft_feat
            vector_parts.extend(fft_feat["fft_peak_mags"])
            vector_parts.append(fft_feat["spectral_centroid"])

            # THD (power quality specific)
            thd = compute_thd(ch, fs, fundamental_freq=50.0)
            all_features[f"{prefix}_thd"] = thd
            vector_parts.append(thd)

        return all_features, np.array(vector_parts, dtype=np.float32)

    def _extract_cwru(self, signal):
        """CWRU features: time stats + envelope spectrum + defect frequencies."""
        fs = self.cfg["sample_rate_hz"]
        defect_freqs = compute_defect_freqs()
        all_features = {}
        vector_parts = []

        for ch_idx in range(signal.shape[0]):
            ch = signal[ch_idx]
            prefix = f"ch{ch_idx}"

            # Time-domain
            time_feat = compute_time_features(ch)
            for k, v in time_feat.items():
                all_features[f"{prefix}_{k}"] = v
            vector_parts.extend([time_feat["mean"], time_feat["std"], time_feat["rms"],
                                 time_feat["kurtosis"], time_feat["crest_factor"]])

            # Envelope spectrum + defect frequencies
            env_freqs, env_mag = compute_envelope_spectrum(ch, fs)
            for df_name, df_val in defect_freqs.items():
                present, amp, freq = check_defect_frequency(env_freqs, env_mag, df_val)
                all_features[f"{prefix}_{df_name}_present"] = present
                all_features[f"{prefix}_{df_name}_amplitude"] = amp
                vector_parts.extend([float(present), amp])

        return all_features, np.array(vector_parts, dtype=np.float32)

    def _extract_care(self, signal):
        """CARE features: rolling statistics over SCADA tags."""
        all_features = {}
        vector_parts = []

        for tag_idx in range(signal.shape[0]):
            tag = signal[tag_idx]

            # Rolling statistics
            mean_val = float(np.mean(tag))
            std_val = float(np.std(tag))
            range_val = float(np.max(tag) - np.min(tag))

            # Trend slope (linear regression)
            x = np.arange(len(tag))
            if std_val > 1e-10:
                slope = float(np.polyfit(x, tag, 1)[0])
            else:
                slope = 0.0

            all_features[f"tag{tag_idx}_mean"] = mean_val
            all_features[f"tag{tag_idx}_std"] = std_val
            all_features[f"tag{tag_idx}_range"] = range_val
            all_features[f"tag{tag_idx}_slope"] = slope

            vector_parts.extend([mean_val, std_val, range_val, slope])

        return all_features, np.array(vector_parts, dtype=np.float32)
