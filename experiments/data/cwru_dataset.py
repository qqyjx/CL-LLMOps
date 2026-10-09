"""CWRU Bearing Fault Dataset loader.

10-class bearing fault classification from Case Western Reserve University.
Source: https://engineering.case.edu/bearingdatacenter
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATASETS, SPLIT_RATIOS, RANDOM_SEED, DATA_DIR, PROCESSED_DIR


CWRU_LABELS = [
    "normal",
    "IR_007", "IR_014", "IR_021",
    "OR_007", "OR_014", "OR_021",
    "RE_007", "RE_014", "RE_021",
]

# Characteristic defect frequencies (normalized by shaft speed)
# For 6205-2RS bearing at ~1797 RPM
BEARING_PARAMS = {
    "n_balls": 9,
    "ball_dia_mm": 7.94,
    "pitch_dia_mm": 39.04,
    "contact_angle_deg": 0,
}


def compute_defect_freqs(shaft_rpm=1797):
    """Compute BPFI, BPFO, BSF, FTF from bearing geometry."""
    n = BEARING_PARAMS["n_balls"]
    d = BEARING_PARAMS["ball_dia_mm"]
    D = BEARING_PARAMS["pitch_dia_mm"]
    theta = np.radians(BEARING_PARAMS["contact_angle_deg"])
    fr = shaft_rpm / 60.0

    ftf = fr / 2 * (1 - d / D * np.cos(theta))
    bpfo = n / 2 * fr * (1 - d / D * np.cos(theta))
    bpfi = n / 2 * fr * (1 + d / D * np.cos(theta))
    bsf = D / (2 * d) * fr * (1 - (d / D * np.cos(theta)) ** 2)

    return {"bpfi": bpfi, "bpfo": bpfo, "bsf": bsf, "ftf": ftf}


class CWRUDataset(Dataset):
    """CWRU Bearing Fault dataset.

    Each sample: (vibration_segment, label)
    vibration_segment: [2, window_size] (DE + FE accelerometers)
    """

    def __init__(self, split="train", load_hp=None, transform=None):
        self.split = split
        self.load_hp = load_hp  # None=all loads, or specific HP
        self.transform = transform
        self.cfg = DATASETS["cwru"]
        self._load_data()

    def _load_data(self):
        suffix = f"_hp{self.load_hp}" if self.load_hp is not None else ""
        cache_path = PROCESSED_DIR / f"cwru_{self.split}{suffix}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            self.signals = data["signals"]
            self.labels = data["labels"]
            return

        raw_path = DATA_DIR / "cwru"
        if not (raw_path.exists() and any(raw_path.iterdir())):
            raise FileNotFoundError(
                f"CWRU raw data not found at {raw_path}. "
                "Download CWRU Bearing Fault .mat files to this path "
                "before running experiments."
            )
        self._load_from_raw(raw_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, signals=self.signals, labels=self.labels)

    def _load_from_raw(self, raw_path):
        """Load from downloaded .mat files."""
        from scipy.io import loadmat

        signals_all = []
        labels_all = []
        window = self.cfg["window_size"]
        overlap = self.cfg["overlap"]
        step = int(window * (1 - overlap))

        mat_files = sorted(raw_path.glob("*.mat"))
        for mat_file in mat_files:
            data = loadmat(str(mat_file))
            label = self._filename_to_label(mat_file.stem)
            if label is None:
                continue

            # Extract DE and FE accelerometer data
            de_key = [k for k in data.keys() if "DE_time" in k]
            fe_key = [k for k in data.keys() if "FE_time" in k]

            if de_key:
                de_signal = data[de_key[0]].flatten()
            else:
                continue

            fe_signal = data[fe_key[0]].flatten() if fe_key else np.zeros_like(de_signal)

            # Segment with overlap
            for start in range(0, len(de_signal) - window, step):
                seg_de = de_signal[start:start + window]
                seg_fe = fe_signal[start:start + window] if len(fe_signal) > start + window else np.zeros(window)
                segment = np.stack([seg_de, seg_fe]).astype(np.float32)
                signals_all.append(segment)
                labels_all.append(label)

        if not signals_all:
            raise RuntimeError(
                f"No valid CWRU segments extracted from {raw_path}."
            )

        signals_all = np.array(signals_all)
        labels_all = np.array(labels_all, dtype=np.int64)
        self._split_and_assign(signals_all, labels_all)

    def _filename_to_label(self, filename):
        """Map CWRU .mat filename to class index."""
        fname = filename.lower()
        if "normal" in fname or "97" in fname or "98" in fname or "99" in fname or "100" in fname:
            return 0
        fault_map = {
            ("ir", "007"): 1, ("ir", "014"): 2, ("ir", "021"): 3,
            ("or", "007"): 4, ("or", "014"): 5, ("or", "021"): 6,
            ("b", "007"): 7, ("b", "014"): 8, ("b", "021"): 9,
        }
        for (fault_type, size), label in fault_map.items():
            if fault_type in fname and size in fname:
                return label
        return None

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
