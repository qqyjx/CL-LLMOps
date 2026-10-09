"""Domain knowledge base for evidence retrieval (paper Section 5).

Builds the FAISS-indexed KB used by Stage V1 (evidence retrieval):
  channel 1 -- curated fault-signature descriptions per class,
    following IEEE 1159 (PQ), bearing defect characteristics (CWRU),
    and SCADA alarm interpretation (CARE);
  channel 2 -- historical case reports summarized from the training
    split (per-class signal statistics of verified episodes);
  channel 3 -- maintenance procedure templates mapping fault types to
    interventions with risk levels (ISO 10816 / IEC 61400 informed).

The KnowledgeBase object exposes the faiss-index subset used by
CLLLMOpsPipeline (`search`, `reconstruct`) plus document access
(`get_doc`) so the evaluation-time judge can inspect the retrieved
evidence TEXT, not only its embedding. Falls back to a pure-numpy
inner-product index when faiss is unavailable.
"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import KB_CASES_PER_CLASS
from evaluation.action_rules import (
    get_class_names, severity_of, action_level_for,
)


# ── Curated signature descriptions (channel 1) ──

_PQ16_SIGNATURES = {
    "normal": "steady RMS voltage near 1.0 pu, THD below 5 percent, "
              "no transient events",
    "sag": "RMS voltage drop to 0.1-0.9 pu lasting 0.5 cycle to 1 min, "
           "typically from faults or motor starting",
    "swell": "RMS voltage rise to 1.1-1.8 pu, often from load "
             "switching or single-phase faults",
    "interruption": "supply voltage below 0.1 pu, loss of service",
    "harmonic": "periodic waveform distortion with elevated THD and "
                "integer-multiple spectral components",
    "flicker": "systematic low-frequency envelope modulation of "
               "voltage amplitude, 0.1-7 percent",
    "transient": "sub-cycle impulsive or oscillatory deviation with "
                 "fast rise time and high peak magnitude",
    "notch": "repetitive sub-cycle voltage notching from power "
             "electronic commutation",
    "complex_disturbance": "multiple simultaneous disturbance modes "
                           "with mixed spectral and RMS anomalies",
}

_PQ_SIGNATURES = {
    "Normal": _PQ16_SIGNATURES["normal"],
    "Sag": _PQ16_SIGNATURES["sag"],
    "Swell": _PQ16_SIGNATURES["swell"],
    "Interruption": _PQ16_SIGNATURES["interruption"],
    "Harmonics": _PQ16_SIGNATURES["harmonic"],
    "Flicker": _PQ16_SIGNATURES["flicker"],
    "Notch": _PQ16_SIGNATURES["notch"],
    "Spike": "isolated unipolar sub-cycle voltage spike with high "
             "crest factor",
}

_CWRU_SIGNATURES = {
    "Normal": "smooth vibration spectrum, no defect-frequency "
              "harmonics, low kurtosis",
    "IR": "envelope-spectrum energy at ball-pass frequency inner race "
          "(BPFI) with shaft-speed sidebands",
    "OR": "envelope-spectrum energy at ball-pass frequency outer race "
          "(BPFO), stationary amplitude",
    "Ball": "envelope-spectrum energy at ball spin frequency (BSF) "
            "with cage-frequency modulation",
}

_CARE_SIGNATURES = {
    "Normal": "SCADA tags within operating envelope, stable "
              "temperature and power-curve residuals",
    "Pre-fault": "drifting SCADA channels: rising component "
                 "temperature residuals and abnormal power-curve "
                 "deviation preceding a logged fault",
}

_SIZE_NOTE = {"007": "0.007 inch incipient defect",
              "014": "0.014 inch developed defect",
              "021": "0.021 inch severe defect"}


def _signature_text(dataset_name, class_name):
    """Compose a curated signature description for one class."""
    if dataset_name == "pq16":
        if class_name in _PQ16_SIGNATURES:
            return _PQ16_SIGNATURES[class_name]
        parts = [p for p in class_name.split("_")
                 if p in _PQ16_SIGNATURES]
        return "compound disturbance: " + "; combined with ".join(
            _PQ16_SIGNATURES[p] for p in parts)
    if dataset_name == "pq":
        if class_name in _PQ_SIGNATURES:
            return _PQ_SIGNATURES[class_name]
        parts = [p for p in class_name.split("+")
                 if p in _PQ_SIGNATURES]
        return "compound disturbance: " + "; combined with ".join(
            _PQ_SIGNATURES[p] for p in parts)
    if dataset_name == "cwru":
        if class_name == "Normal":
            return _CWRU_SIGNATURES["Normal"]
        fault = class_name[:-3] if class_name[-3:].isdigit() \
            else class_name
        size = class_name[len(fault):]
        base = _CWRU_SIGNATURES.get(fault, "bearing defect signature")
        return f"{base}; {_SIZE_NOTE.get(size, size)}"
    return _CARE_SIGNATURES.get(class_name, class_name)


class KnowledgeBase:
    """Embedding index over structured KB documents.

    Exposes the faiss API subset used by the pipeline (`search`,
    `reconstruct`) plus text access for the evaluation judge.
    Embeddings are L2-normalized so inner product = cosine similarity.
    """

    def __init__(self, embeddings, docs):
        embs = np.asarray(embeddings, dtype=np.float32)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        self.embeddings = embs
        self.docs = docs  # list of {"text", "tag", "source"}
        try:
            import faiss
            self._index = faiss.IndexFlatIP(embs.shape[1])
            self._index.add(embs)
        except ImportError:
            self._index = None

    def __len__(self):
        return len(self.docs)

    def search(self, query, k):
        """Top-k inner-product search; returns (scores, indices)."""
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self._index is not None:
            return self._index.search(query, k)
        sims = self.embeddings @ query[0]
        idx = np.argsort(sims)[::-1][:k]
        return sims[idx].reshape(1, -1), idx.reshape(1, -1)

    def reconstruct(self, i):
        """Stored (normalized) embedding at index i."""
        return self.embeddings[int(i)]

    def get_doc(self, i):
        """Structured document at index i (text/tag/source)."""
        return self.docs[int(i)]


def build_knowledge_base(dataset_name, encoder, splits=None,
                         cases_per_class=KB_CASES_PER_CLASS, seed=42):
    """Construct the per-dataset KB (three channels, see module doc).

    Args:
        dataset_name: "pq16" | "pq" | "cwru" | "care"
        encoder: sentence encoder with .encode(list[str]) -> (N, d)
        splits: dict from data.load_splits (channel 2 case reports are
            drawn from its train partition); None skips channel 2
        cases_per_class: historical case reports per class
        seed: RandomState seed for case sampling
    Returns:
        KnowledgeBase
    """
    class_names = get_class_names(dataset_name)
    docs = []

    # Channel 1: curated signatures
    for name in class_names:
        docs.append({
            "text": f"Fault signature [{name}]: "
                    f"{_signature_text(dataset_name, name)}",
            "tag": name,
            "source": "standard/textbook signature",
        })

    # Channel 2: historical case reports from the train split
    if splits is not None:
        rng = np.random.RandomState(seed)
        X, y = splits["X_train"], splits["y_train"]
        for cls_id, name in enumerate(class_names):
            idx = np.where(y == cls_id)[0]
            if len(idx) == 0:
                continue
            chosen = rng.choice(idx, size=min(cases_per_class, len(idx)),
                                replace=False)
            for j in chosen:
                sig = np.asarray(X[j], dtype=np.float64)
                docs.append({
                    "text": f"Case report [{name}]: verified episode "
                            f"with signal rms={np.sqrt((sig**2).mean()):.4f}, "
                            f"std={sig.std():.4f}, "
                            f"peak={np.abs(sig).max():.4f}; diagnosis "
                            f"confirmed by maintenance outcome",
                    "tag": name,
                    "source": "historical case (train split)",
                })

    # Channel 3: maintenance procedure templates
    for name in class_names:
        sev = severity_of(dataset_name, name)
        docs.append({
            "text": f"Maintenance procedure [{name}]: severity "
                    f"category {sev}; recommended action level "
                    f"{action_level_for(dataset_name, name)}; follow "
                    f"lockout-tagout and standard remediation steps",
            "tag": name,
            "source": "procedure template",
        })

    embeddings = encoder.encode([d["text"] for d in docs])
    return KnowledgeBase(embeddings, docs)
