"""Dataset loaders for CL-LLMOps experiments."""

from .pq_dataset import PQDataset
from .pq16_dataset import PQ16Dataset
from .cwru_dataset import CWRUDataset
from .care_dataset import CAREDataset

__all__ = ["PQDataset", "PQ16Dataset", "CWRUDataset", "CAREDataset", "load_splits"]


def load_splits(dataset_name):
    """Load train/val/test splits as numpy arrays.

    Returns:
        dict with keys X_train, y_train, X_val, y_val, X_test, y_test
    """
    cls_map = {
        "pq": PQDataset,
        "pq16": PQ16Dataset,
        "cwru": CWRUDataset,
        "care": CAREDataset,
    }
    cls = cls_map[dataset_name]

    train_ds = cls(split="train")
    val_ds = cls(split="val")
    test_ds = cls(split="test")

    return {
        "X_train": train_ds.signals, "y_train": train_ds.labels,
        "X_val": val_ds.signals, "y_val": val_ds.labels,
        "X_test": test_ds.signals, "y_test": test_ds.labels,
    }
