"""Deep learning baselines: B3 (1D-CNN / WDCNN) and B4 (Transformer).

These baselines operate on raw time-series signals (no hand-crafted features).
They produce only a classification label (no explanations or actions).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, RANDOM_SEED, DEVICE, BASELINE_EPOCHS, BASELINE_LR


# ── B3: 1D-CNN (WDCNN-style) ──


class CNN1D(nn.Module):
    """Wide-kernel first-layer 1D-CNN for vibration/PQ signals."""

    def __init__(self, in_channels, n_classes, seq_len=2048):
        super().__init__()
        # Wide first conv captures impulse-like features
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)


# ── B4: Transformer Encoder ──


class PatchEmbedding(nn.Module):
    """Split signal into patches and project to d_model."""

    def __init__(self, in_channels, d_model, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size, d_model)

    def forward(self, x):
        # x: (B, C, L) -> (B, N, d_model) where N = L // patch_size
        B, C, L = x.shape
        n_patches = L // self.patch_size
        x = x[:, :, :n_patches * self.patch_size]
        x = x.reshape(B, C, n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, n_patches, C * self.patch_size)
        return self.proj(x)


class TransformerClassifier(nn.Module):
    """Transformer encoder for time-series classification."""

    def __init__(self, in_channels, n_classes, seq_len=2048,
                 d_model=128, n_heads=4, n_layers=4, patch_size=32):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, d_model, patch_size)
        n_patches = seq_len // patch_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.encoder(x)
        return self.head(x[:, 0])


# ── Unified Trainer ──


class DeepLearningTrainer:
    """Training and evaluation wrapper for DL baselines."""

    def __init__(self, model, model_name="cnn", device=None,
                 epochs=None, patience=15):
        self.device = device or DEVICE
        self.model = model.to(self.device)
        self.model_name = model_name
        base_name = "transformer" if model_name.startswith("transformer") \
            else "cnn"
        self.epochs = epochs or BASELINE_EPOCHS.get(base_name, 100)
        self.lr = BASELINE_LR.get(base_name, 1e-3)
        self.patience = patience

    def _make_loader(self, X, y, batch_size=64, shuffle=True):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t),
                          batch_size=batch_size, shuffle=shuffle)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        train_loader = self._make_loader(X_train, y_train)
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()
        best_val_f1 = 0.0
        stale = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            if X_val is not None and (epoch + 1) % 5 == 0:
                metrics = self._evaluate_internal(X_val, y_val)
                if metrics["f1"] > best_val_f1:
                    best_val_f1 = metrics["f1"]
                    self.save()
                    stale = 0
                else:
                    stale += 5
                print(f"Epoch {epoch+1}/{self.epochs}  "
                      f"loss={total_loss/len(train_loader):.4f}  "
                      f"val_f1={metrics['f1']:.4f}")
                if stale >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        best_path = MODEL_DIR / f"{self.model_name}_best.pt"
        if best_path.exists():
            self.load(best_path)

    def predict(self, X):
        self.model.train(False)
        loader = self._make_loader(X, np.zeros(len(X)), shuffle=False)
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                out = self.model(X_batch.to(self.device))
                preds.append(out.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    def predict_proba(self, X):
        self.model.train(False)
        loader = self._make_loader(X, np.zeros(len(X)), shuffle=False)
        all_proba = []
        with torch.no_grad():
            for X_batch, _ in loader:
                out = self.model(X_batch.to(self.device))
                proba = torch.softmax(out, dim=1).cpu().numpy()
                all_proba.append(proba)
        return np.concatenate(all_proba)

    def _evaluate_internal(self, X, y):
        preds = self.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "f1": f1_score(y, preds, average="macro"),
        }

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        proba = self.predict_proba(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        k = min(3, proba.shape[1])
        topk_preds = np.argsort(proba, axis=1)[:, -k:]
        top3 = sum(1 for i, y in enumerate(y_test)
                   if y in topk_preds[i]) / len(y_test)
        return {"accuracy": acc, "f1": f1, "top3_accuracy": top3}

    def save(self, path=None):
        path = path or MODEL_DIR / f"{self.model_name}_best.pt"
        torch.save(self.model.state_dict(), path)

    def load(self, path=None):
        path = path or MODEL_DIR / f"{self.model_name}_best.pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device))
