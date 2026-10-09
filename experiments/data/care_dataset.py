"""CARE Wind Turbine SCADA Dataset loader.

Binary anomaly detection (normal vs pre-fault) from wind turbine SCADA data.
Source: CARE to Compare (Gueck & Roelofs, MDPI Data 2024, 9(12):138),
Zenodo record 15846963 (v6). Raw staging: one CSV per turbine(-dataset)
with a row-level `label` column marking pre-fault periods.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASETS, SPLIT_RATIOS, RANDOM_SEED, DATA_DIR, PROCESSED_DIR


CARE_LABELS = ["normal", "pre_fault"]

# Representative SCADA tags used in the paper
SCADA_TAGS = [
    "active_power", "reactive_power", "rotor_speed", "generator_speed",
    "generator_temp_1", "generator_temp_2", "bearing_temp_a", "bearing_temp_b",
    "gearbox_oil_temp", "gearbox_bearing_temp", "nacelle_temp", "ambient_temp",
    "pitch_angle_1", "pitch_angle_2", "pitch_angle_3",
    "nacelle_wind_speed", "yaw_angle", "grid_voltage", "grid_frequency",
    "hydraulic_pressure", "tower_accel_x", "tower_accel_y",
    # ... remaining tags follow the full CARE schema (60 total)
]


class CAREDataset(Dataset):
    """CARE Wind Turbine SCADA anomaly detection dataset.

    Each sample: (scada_window, label) where scada_window is [n_tags, window_steps].
    window_steps = 144 (24 hours at 10-min intervals).
    """

    def __init__(self, split="train", transform=None):
        self.split = split
        self.transform = transform
        self.cfg = DATASETS["care"]
        self._load_data()

    def _load_data(self):
        cache_path = PROCESSED_DIR / f"care_{self.split}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            self.signals = data["signals"]
            self.labels = data["labels"]
            return

        raw_path = DATA_DIR / "care_wind"
        if not (raw_path.exists() and any(raw_path.iterdir())):
            raise FileNotFoundError(
                f"CARE raw data not found at {raw_path}. "
                "Download the CARE Wind SCADA dataset to this path "
                "before running experiments."
            )
        self._load_from_raw(raw_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, signals=self.signals, labels=self.labels)

    def _load_from_raw(self, raw_path):
        """Load CSVs and split at the FILE (dataset) level.

        Each of the 95 CARE CSV files (one anomaly-event or normal-behaviour
        dataset) lands in exactly one partition (70/15/15). Files are
        stratified by whether they contain any pre-fault rows, then windowed
        independently into their assigned partition. A turbine can
        contribute several files, so the same turbine can appear in more
        than one partition (6 of the 10 test turbines also occur in
        training); see the paper, Section 5.1.
        """
        import pandas as pd

        csv_files = sorted(raw_path.glob("*.csv"))
        if not csv_files:
            raise RuntimeError(
                f"No valid CARE segments extracted from {raw_path}."
            )

        frames = []
        file_has_fault = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Expect columns: timestamp, tag1, tag2, ..., label (or separate label file)
            feature_cols = [c for c in df.columns if c not in ("timestamp", "label", "anomaly")]

            values = df[feature_cols].values.astype(np.float32)
            # Determine label: check for 'label' or 'anomaly' column
            if "label" in df.columns:
                label_series = df["label"].values
            elif "anomaly" in df.columns:
                label_series = df["anomaly"].values
            else:
                # Assume normal if no label column
                label_series = np.zeros(len(df), dtype=int)
            frames.append((values, label_series))
            file_has_fault.append(int(np.any(label_series)))

        indices = np.arange(len(frames))
        flags = np.asarray(file_has_fault)

        def _stratify_or_none(subset_flags):
            # Stratification needs >= 2 members of each class; tiny file
            # collections (e.g. smoke fixtures) fall back to unstratified.
            counts = np.bincount(subset_flags, minlength=2)
            return subset_flags if counts.min() >= 2 else None

        train_ratio = SPLIT_RATIOS["train"]
        val_ratio = SPLIT_RATIOS["val"]
        test_ratio = SPLIT_RATIOS["test"]
        train_idx, rest_idx = train_test_split(
            indices, test_size=1 - train_ratio,
            random_state=RANDOM_SEED, stratify=_stratify_or_none(flags))
        relative_val = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            rest_idx, test_size=1 - relative_val,
            random_state=RANDOM_SEED,
            stratify=_stratify_or_none(flags[rest_idx]))
        split_files = {"train": train_idx, "val": val_idx,
                       "test": test_idx}

        window = self.cfg["window_steps"]
        signals_all = []
        labels_all = []
        for i in split_files[self.split]:
            values, label_series = frames[i]
            # Sliding window segmentation
            for start in range(0, len(values) - window):
                seg = values[start:start + window].T  # [n_tags, window]
                lbl = 1 if label_series[start:start + window].any() else 0
                signals_all.append(seg)
                labels_all.append(lbl)

        if not signals_all:
            raise RuntimeError(
                f"No valid CARE segments extracted from {raw_path} "
                f"for split {self.split}."
            )

        self.signals = np.array(signals_all)
        self.labels = np.array(labels_all, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        if self.transform:
            signal = self.transform(signal)
        return signal, label
