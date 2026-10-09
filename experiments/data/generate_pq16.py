"""IEEE 1159-2019 compliant 16-class PQ disturbance generator.

Generates synthetic three-phase power-quality waveforms for the paper's
main experiments. Held back from the public CL-LLMOps release (runbook-
level artefact); kept in the paper workspace so results can be reproduced.

Signal model (paper Section 5.1, "PQ data provenance"):

  * Three-phase 50 Hz sinusoid, 6 channels (V_abc + I_abc) with a 30 deg
    inductive phase lag on the current.
  * Sampling: 3.2 kHz, 640-sample window (200 ms, 10 fundamental cycles).
  * Disturbance parameters per class drawn from IEEE 1159 Tables 4.2-4.4:
        sag        : 0.1-0.9 pu, 1-9 cycles (simplified for 10-cycle window)
        swell      : 1.1-1.8 pu, 1-9 cycles
        interrupt  : <=0.1 pu,  1-9 cycles
        harmonic   : 3rd/5th/7th orders, 2-20% each
        flicker    : 8-25 Hz envelope, 2-10% modulation depth
        transient  : 400-2000 Hz, exponential damping tau in [2, 10] ms
        notch      : 2 or 6 events/cycle, width 0.3-1.5 ms, depth 20-60%
  * Additive Gaussian measurement noise sigma = 0.005 pu.
  * 16 classes: 1 normal + 7 primitives + 7 pair composites + 1 "complex"
    (three or more primitives simultaneously).

Usage:
    python experiments/data/generate_pq16.py --n_per_class 400

At n_per_class=400, produces 6400 total samples (4480 train + 960 val +
960 test at 70/15/15), which covers the demo slice of 4000 train + 1000
test plus validation margin. The A800 runbook calls the same generator
with n_per_class=15000 for the paper's 240K-sample table.

Outputs (relative to the experiments/ directory):
    data/processed/pq16_train.npz
    data/processed/pq16_val.npz
    data/processed/pq16_test.npz

Each npz contains:
    signals : float32, shape [N, 6, 640]
    labels  : int64,   shape [N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED_DIR, RANDOM_SEED, SPLIT_RATIOS


# ── IEEE 1159 signal-model constants ──
FS_HZ = 3200                       # sampling rate
WINDOW = 640                       # samples per window (200 ms, 10 cycles)
F0 = 50.0                          # fundamental frequency
V_NOMINAL = 1.0                    # per-unit voltage amplitude
PHASE_LAG_I = np.deg2rad(30.0)     # inductive lag on current
NOISE_SIGMA = 0.005                # additive Gaussian noise, pu

PQ_LABELS = [
    "normal", "sag", "swell", "interruption",
    "harmonic", "flicker", "transient", "notch",
    "sag_harmonic", "swell_harmonic", "sag_transient",
    "swell_transient", "harmonic_flicker", "sag_flicker",
    "interruption_harmonic", "complex_disturbance",
]

PRIMITIVE_IDS = {
    "sag": 1, "swell": 2, "interruption": 3, "harmonic": 4,
    "flicker": 5, "transient": 6, "notch": 7,
}


# ── Helpers ──
def _t_axis():
    return np.arange(WINDOW) / FS_HZ


def _base_three_phase(rng):
    """Clean three-phase V_abc + I_abc waveform (6 channels)."""
    t = _t_axis()
    omega = 2 * np.pi * F0 * t
    v_a = V_NOMINAL * np.sin(omega)
    v_b = V_NOMINAL * np.sin(omega - 2 * np.pi / 3)
    v_c = V_NOMINAL * np.sin(omega + 2 * np.pi / 3)
    i_a = V_NOMINAL * np.sin(omega - PHASE_LAG_I)
    i_b = V_NOMINAL * np.sin(omega - 2 * np.pi / 3 - PHASE_LAG_I)
    i_c = V_NOMINAL * np.sin(omega + 2 * np.pi / 3 - PHASE_LAG_I)
    sig = np.stack([v_a, v_b, v_c, i_a, i_b, i_c], axis=0)
    return sig.astype(np.float32)


def _event_window(rng, min_cycles=1, max_cycles=9):
    """Return (start, end) sample indices for a disturbance event."""
    cycles = rng.uniform(min_cycles, max_cycles)
    dur = int(round(cycles * FS_HZ / F0))
    dur = min(dur, WINDOW - 32)
    start = rng.integers(0, WINDOW - dur)
    return start, start + dur


# ── Primitive disturbance injectors (mutate `signal` in place) ──
def _apply_sag(sig, rng):
    depth = rng.uniform(0.1, 0.9)           # residual amplitude, pu
    start, end = _event_window(rng)
    scale = np.ones(WINDOW, dtype=np.float32)
    scale[start:end] = depth
    sig[:3] *= scale
    sig[3:] *= scale


def _apply_swell(sig, rng):
    amp = rng.uniform(1.1, 1.8)
    start, end = _event_window(rng)
    scale = np.ones(WINDOW, dtype=np.float32)
    scale[start:end] = amp
    sig[:3] *= scale


def _apply_interruption(sig, rng):
    depth = rng.uniform(0.0, 0.1)
    start, end = _event_window(rng)
    scale = np.ones(WINDOW, dtype=np.float32)
    scale[start:end] = depth
    sig[:3] *= scale
    sig[3:] *= scale


def _apply_harmonic(sig, rng):
    t = _t_axis()
    orders = [3, 5, 7]
    rng.shuffle(orders)
    use = orders[: rng.integers(1, 4)]
    for order in use:
        mag = rng.uniform(0.02, 0.20)
        phi_v = rng.uniform(0, 2 * np.pi)
        for ch in range(3):
            phase_shift = -2 * np.pi / 3 * ch if ch > 0 else 0.0
            sig[ch] += mag * np.sin(2 * np.pi * order * F0 * t + phi_v + phase_shift)
            sig[ch + 3] += mag * np.sin(
                2 * np.pi * order * F0 * t + phi_v + phase_shift - PHASE_LAG_I
            )


def _apply_flicker(sig, rng):
    t = _t_axis()
    f_flicker = rng.uniform(8.0, 25.0)
    depth = rng.uniform(0.02, 0.10)
    envelope = 1.0 + depth * np.sin(2 * np.pi * f_flicker * t)
    sig[:3] *= envelope.astype(np.float32)
    sig[3:] *= envelope.astype(np.float32)


def _apply_transient(sig, rng):
    t = _t_axis()
    f_osc = rng.uniform(400.0, 2000.0)
    tau = rng.uniform(0.002, 0.010)
    amp = rng.uniform(0.3, 1.5)
    start = rng.integers(WINDOW // 8, WINDOW * 3 // 4)
    t_rel = np.zeros(WINDOW, dtype=np.float32)
    t_rel[start:] = (np.arange(WINDOW - start) / FS_HZ)
    mask = (np.arange(WINDOW) >= start).astype(np.float32)
    burst = (
        mask * amp * np.exp(-t_rel / tau) *
        np.sin(2 * np.pi * f_osc * t_rel)
    )
    sig[:3] += burst[None, :]


def _apply_notch(sig, rng):
    events_per_cycle = rng.choice([2, 6])
    width_s = rng.uniform(0.0003, 0.0015)
    depth = rng.uniform(0.2, 0.6)
    width_samp = max(1, int(round(width_s * FS_HZ)))
    period_samp = int(round(FS_HZ / F0 / events_per_cycle))
    for start in range(4, WINDOW - width_samp, period_samp):
        jitter = rng.integers(-2, 3)
        s = max(0, start + jitter)
        e = min(WINDOW, s + width_samp)
        sig[:3, s:e] *= (1.0 - depth)


PRIMITIVE_APPLY = {
    "sag": _apply_sag,
    "swell": _apply_swell,
    "interruption": _apply_interruption,
    "harmonic": _apply_harmonic,
    "flicker": _apply_flicker,
    "transient": _apply_transient,
    "notch": _apply_notch,
}


# ── Class dispatcher ──
COMPOSITE_MAP = {
    "sag_harmonic":           ("sag", "harmonic"),
    "swell_harmonic":         ("swell", "harmonic"),
    "sag_transient":          ("sag", "transient"),
    "swell_transient":        ("swell", "transient"),
    "harmonic_flicker":       ("harmonic", "flicker"),
    "sag_flicker":            ("sag", "flicker"),
    "interruption_harmonic":  ("interruption", "harmonic"),
}


def _generate_one(class_name, rng):
    sig = _base_three_phase(rng)
    if class_name == "normal":
        pass
    elif class_name in PRIMITIVE_APPLY:
        PRIMITIVE_APPLY[class_name](sig, rng)
    elif class_name in COMPOSITE_MAP:
        for prim in COMPOSITE_MAP[class_name]:
            PRIMITIVE_APPLY[prim](sig, rng)
    elif class_name == "complex_disturbance":
        prims = list(PRIMITIVE_APPLY.keys())
        rng.shuffle(prims)
        k = rng.integers(3, 5)
        for prim in prims[:k]:
            PRIMITIVE_APPLY[prim](sig, rng)
    else:
        raise ValueError(f"Unknown class: {class_name}")

    sig += rng.normal(0, NOISE_SIGMA, size=sig.shape).astype(np.float32)
    return sig


def generate_dataset(n_per_class: int, seed: int = RANDOM_SEED):
    """Return (signals [N,6,640] float32, labels [N] int64)."""
    rng = np.random.default_rng(seed)
    signals = np.empty((n_per_class * len(PQ_LABELS), 6, WINDOW), dtype=np.float32)
    labels = np.empty(n_per_class * len(PQ_LABELS), dtype=np.int64)
    for cls_id, cls_name in enumerate(PQ_LABELS):
        for k in range(n_per_class):
            idx = cls_id * n_per_class + k
            signals[idx] = _generate_one(cls_name, rng)
            labels[idx] = cls_id
    perm = rng.permutation(len(labels))
    return signals[perm], labels[perm]


def main():
    parser = argparse.ArgumentParser(description="Generate PQ16 IEEE 1159 dataset.")
    parser.add_argument("--n_per_class", type=int, default=400,
                        help="samples per class (default 400; paper runbook uses 15000).")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out_dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    print(f"Generating PQ16: {args.n_per_class} samples/class x 16 classes "
          f"= {args.n_per_class * 16} total, seed={args.seed}")
    X, y = generate_dataset(args.n_per_class, seed=args.seed)
    print(f"  signals: {X.shape} {X.dtype}   labels: {y.shape} {y.dtype}")

    train_ratio = SPLIT_RATIOS["train"]
    val_ratio = SPLIT_RATIOS["val"]
    test_ratio = SPLIT_RATIOS["test"]

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=1 - train_ratio, random_state=args.seed, stratify=y
    )
    rel_val = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=1 - rel_val, random_state=args.seed, stratify=y_rest
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, (Xs, ys) in [("train", (X_train, y_train)),
                           ("val",   (X_val,   y_val)),
                           ("test",  (X_test,  y_test))]:
        path = args.out_dir / f"pq16_{name}.npz"
        np.savez_compressed(path, signals=Xs, labels=ys)
        print(f"  wrote {path}  N={len(ys)}")

    print("PQ16 generation done.")


if __name__ == "__main__":
    main()
