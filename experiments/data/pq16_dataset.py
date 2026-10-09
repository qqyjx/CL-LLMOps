"""PQ16 (IEEE 1159-2019) dataset loader.

Loads pre-generated splits from ``processed/pq16_{split}.npz``. The
generator lives in ``data/generate_pq16.py`` and is intentionally kept
out of the public CL-LLMOps release.

Channels: 6 (V_abc + I_abc), window: 640 samples at 3.2 kHz, 16 classes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR


PQ16_LABELS = [
    "normal", "sag", "swell", "interruption",
    "harmonic", "flicker", "transient", "notch",
    "sag_harmonic", "swell_harmonic", "sag_transient",
    "swell_transient", "harmonic_flicker", "sag_flicker",
    "interruption_harmonic", "complex_disturbance",
]


class PQ16Dataset(Dataset):
    """16-class IEEE 1159 PQ dataset loaded from pre-generated npz splits."""

    N_CLASSES = 16
    CHANNELS = 6
    WINDOW_SIZE = 640
    SAMPLE_RATE_HZ = 3200

    def __init__(self, split: str = "train", transform=None):
        assert split in {"train", "val", "test"}, f"Unknown split: {split}"
        self.split = split
        self.transform = transform
        self._load()

    def _load(self):
        path = PROCESSED_DIR / f"pq16_{self.split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"PQ16 split not found at {path}. "
                "Generate it first:\n"
                "    python experiments/data/generate_pq16.py --n_per_class 400"
            )
        data = np.load(path)
        self.signals = data["signals"].astype(np.float32)
        self.labels = data["labels"].astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        if self.transform:
            signal = self.transform(signal)
        return signal, label
