"""Reimplementations of recent LLM/multimodal fault-diagnosis baselines.

* FD-LLM (arXiv:2412.01218): the runner uses this module's 15-feature
  serialization with a rank-4 LoRA instruction-tuning loop.
* T2MFDF (IEEE TIM 2025, DOI 10.1109/TIM.2025.3583374): a time-series
  encoder is fused with BERT-family fault-description embeddings through a
  shared class scorer.  Results are explicitly reported as reimplementations
  on this repository's splits, not copied from either source paper.
"""

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.action_rules import get_class_names
from framework.knowledge_base import _signature_text


STAT_NAMES = (
    "mean", "rms", "std", "crest_factor", "skewness", "shape_factor",
    "kurtosis", "peak_to_peak", "energy_factor", "impulse_factor",
    "peak_frequency", "spectral_peak_to_peak", "spectral_kurtosis",
    "spectral_bandwidth", "spectral_skewness",
)


def statistical_features(signal):
    """FD-LLM's ten time + five frequency statistics, channel averaged."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    eps = 1e-8
    mean = x.mean(axis=1)
    centered = x - mean[:, None]
    std = x.std(axis=1) + eps
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    peak = np.max(np.abs(x), axis=1)
    abs_mean = np.mean(np.abs(x), axis=1) + eps
    skew = np.mean((centered / std[:, None]) ** 3, axis=1)
    kurt = np.mean((centered / std[:, None]) ** 4, axis=1)

    spectrum = np.abs(np.fft.rfft(x, axis=1)) / max(x.shape[1], 1)
    freq = np.linspace(0.0, 1.0, spectrum.shape[1])
    weight = spectrum / (spectrum.sum(axis=1, keepdims=True) + eps)
    centroid = (weight * freq[None, :]).sum(axis=1)
    bandwidth = np.sqrt((weight * (freq[None, :]
                                    - centroid[:, None]) ** 2).sum(axis=1))
    fstd = np.sqrt((weight * (freq[None, :]
                              - centroid[:, None]) ** 2).sum(axis=1)) + eps
    fskew = (weight * ((freq[None, :] - centroid[:, None])
                       / fstd[:, None]) ** 3).sum(axis=1)
    fkurt = (weight * ((freq[None, :] - centroid[:, None])
                       / fstd[:, None]) ** 4).sum(axis=1)
    values = np.stack([
        mean, rms, std, peak / (rms + eps), skew,
        rms / abs_mean, kurt, np.ptp(x, axis=1),
        np.mean(x ** 2, axis=1) / (abs_mean ** 2), peak / abs_mean,
        freq[np.argmax(spectrum, axis=1)], np.ptp(spectrum, axis=1),
        fkurt, bandwidth, fskew,
    ], axis=1)
    return np.nan_to_num(values.mean(axis=0), nan=0.0,
                         posinf=0.0, neginf=0.0).astype(np.float32)


def fdllm_prompt(signal, dataset_name):
    features = statistical_features(signal)
    serialized = "; ".join(
        f"{name}={value:.6g}" for name, value in zip(STAT_NAMES, features))
    classes = get_class_names(dataset_name)
    return (
        f"Instruction: Given equipment domain {dataset_name} and the "
        "following time-domain and frequency-domain statistical features, "
        "predict the operating fault class. Return only JSON as "
        '{"class_id": <integer>}.\n'
        f"Valid class IDs: 0..{len(classes)-1}. "
        f"Class map: {json.dumps(dict(enumerate(classes)))}\n"
        f"Input: {serialized}\nOutput:"
    )


def stratified_cap(y, per_class, seed):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    selected = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        selected.extend(rng.choice(
            idx, size=min(per_class, len(idx)), replace=False).tolist())
    return np.asarray(rng.permutation(selected), dtype=int)


class T2MFDFModel(nn.Module):
    """Time/text multimodal fusion with a shared class-conditional scorer."""

    def __init__(self, in_channels, text_embeddings, hidden=128):
        super().__init__()
        self.time_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, 15, stride=4, padding=7),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, hidden, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        text_embeddings = torch.as_tensor(text_embeddings,
                                          dtype=torch.float32)
        self.register_buffer("text_embeddings", text_embeddings)
        self.text_projector = nn.Sequential(
            nn.Linear(text_embeddings.shape[1], hidden), nn.LayerNorm(hidden))
        self.scorer = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        time = self.time_encoder(x).squeeze(-1)
        text = self.text_projector(self.text_embeddings)
        t = time[:, None, :].expand(-1, len(text), -1)
        d = text[None, :, :].expand(len(time), -1, -1)
        fused = torch.cat([t, d, t * d, torch.abs(t - d)], dim=-1)
        return self.scorer(fused).squeeze(-1)


def train_t2mfdf(dataset_name, splits, encoder, device="cuda", seed=42,
                 train_per_class=500, val_per_class=100,
                 test_per_class=200, epochs=50, batch_size=64,
                 checkpoint=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    class_names = get_class_names(dataset_name)
    descriptions = [
        f"Fault class {name}: {_signature_text(dataset_name, name)}"
        for name in class_names
    ]
    text_embeddings = encoder.encode(descriptions)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    tr = stratified_cap(y_train, train_per_class, seed)
    va = stratified_cap(y_val, val_per_class, seed + 1)
    te = stratified_cap(y_test, test_per_class, seed + 2)

    model = T2MFDFModel(X_train.shape[1], text_embeddings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train[tr], dtype=torch.float32),
        torch.tensor(y_train[tr], dtype=torch.long)),
        batch_size=batch_size, shuffle=True)
    best_f1, stale, best_state = -1.0, 0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Paper-style multimodal data augmentation on the signal branch.
            xb = xb * (1.0 + 0.03 * torch.randn(
                len(xb), 1, 1, device=device))
            xb = xb + 0.01 * xb.std(dim=-1, keepdim=True) * \
                torch.randn_like(xb)
            loss = nn.functional.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if (epoch + 1) % 2 == 0:
            val = _torch_metrics(model, X_val[va], y_val[va], device,
                                 batch_size)
            if val["f1"] > best_f1 + 1e-5:
                best_f1, stale = val["f1"], 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 2
            if stale >= 10:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
    metrics = _torch_metrics(model, X_test[te], y_test[te], device,
                             batch_size)
    metrics.update({"n_train": int(len(tr)), "n_val": int(len(va)),
                    "n_test": int(len(te)), "seed": seed})
    return metrics


def _torch_metrics(model, X, y, device, batch_size):
    model.eval()
    outputs = []
    loader = DataLoader(torch.tensor(X, dtype=torch.float32),
                        batch_size=batch_size)
    with torch.no_grad():
        for xb in loader:
            outputs.append(model(xb.to(device)).cpu().numpy())
    logits = np.concatenate(outputs)
    pred = logits.argmax(axis=1)
    k = min(3, logits.shape[1])
    top = np.argsort(logits, axis=1)[:, -k:]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, average="macro")),
        "top3_accuracy": float(np.mean([
            int(label) in row for label, row in zip(y, top)
        ])),
    }


class FDLLMTokenDataset(Dataset):
    def __init__(self, X, y, dataset_name, tokenizer, max_length=512):
        self.rows = []
        for signal, label in zip(X, y):
            prompt = fdllm_prompt(signal, dataset_name)
            answer = f' {{"class_id": {int(label)}}}' + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=True)[
                "input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)[
                "input_ids"]
            input_ids = (prompt_ids + answer_ids)[:max_length]
            n_prompt = min(len(prompt_ids), len(input_ids))
            labels = [-100] * n_prompt + input_ids[n_prompt:]
            self.rows.append((input_ids, labels))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def causal_collate(batch, pad_token_id):
    width = max(len(row[0]) for row in batch)
    input_ids, labels, masks = [], [], []
    for ids, target in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(target + [-100] * pad)
        masks.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def parse_class_id(text, n_classes):
    match = re.search(r'"?class_id"?\s*:\s*(\d+)', text)
    if not match:
        match = re.search(r"\b(\d+)\b", text)
    if not match:
        return 0
    return int(np.clip(int(match.group(1)), 0, n_classes - 1))
