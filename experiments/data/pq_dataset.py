"""PQ Classification Dataset loader (Kaggle).

16-class power quality disturbance classification.
Source: https://www.kaggle.com/datasets/... (to be configured)
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASETS, SPLIT_RATIOS, RANDOM_SEED, DATA_DIR, PROCESSED_DIR


PQ_LABELS = [
    "normal", "sag", "swell", "interruption",
    "harmonic", "flicker", "transient", "notch",
    "sag_harmonic", "swell_harmonic", "sag_transient",
    "swell_transient", "harmonic_flicker", "sag_flicker",
    "interruption_harmonic", "complex_disturbance",
]


class PQDataset(Dataset):
    """Power Quality Classification dataset.

    Loads preprocessed .npz files or raw CSV from Kaggle.
    Each sample: (waveform_segment, label) where waveform is [channels, window_size].
    """

    def __init__(self, split="train", transform=None):
        self.split = split
        self.transform = transform
        self.cfg = DATASETS["pq"]
        self._load_data()

    def _load_data(self):
        cache_path = PROCESSED_DIR / f"pq_{self.split}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            self.signals = data["signals"]
            self.labels = data["labels"]
            return

        raw_path = DATA_DIR / "pq_classification"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"PQ raw data not found at {raw_path}. "
                "Download the Kaggle PQ Classification dataset to this path "
                "before running experiments."
            )
        self._load_from_raw(raw_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, signals=self.signals, labels=self.labels)

    def _load_from_raw(self, raw_path):
        """Load from downloaded Kaggle CSV files."""
        import pandas as pd

        # Expect: raw_path/signals.csv and raw_path/labels.csv
        # or raw_path/*.csv with columns for signal data
        csv_files = sorted(raw_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found under {raw_path}."
            )

        df = pd.read_csv(csv_files[0])
        signals_all = df.iloc[:, :-1].values.astype(np.float32)
        labels_all = df.iloc[:, -1].values.astype(np.int64)

        # Reshape to [N, channels, window_size]
        n_samples = len(labels_all)
        channels = self.cfg["channels"]
        window = self.cfg["window_size"]
        if signals_all.shape[1] == channels * window:
            signals_all = signals_all.reshape(n_samples, channels, window)
        else:
            signals_all = signals_all[:, :channels * window].reshape(n_samples, channels, window)

        self._split_and_assign(signals_all, labels_all)

    def _split_and_assign(self, signals, labels):
        """Stratified train/val/test split."""
        train_ratio = SPLIT_RATIOS["train"]
        val_ratio = SPLIT_RATIOS["val"]
        test_ratio = SPLIT_RATIOS["test"]

        X_train, X_rest, y_train, y_rest = train_test_split(
            signals, labels, test_size=1 - train_ratio,
            random_state=RANDOM_SEED, stratify=labels
        )
        relative_val = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_rest, y_rest, test_size=1 - relative_val,
            random_state=RANDOM_SEED, stratify=y_rest
        )

        splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
        self.signals, self.labels = splits[self.split]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        if self.transform:
            signal = self.transform(signal)
        return signal, label
