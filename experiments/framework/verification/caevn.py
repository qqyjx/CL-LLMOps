"""CAEVN: Cross-Attention Evidence Verification Network.

Architecture (from paper Section 4.3):
  - Hypothesis embedding h (from LLM output, 384-dim via sentence-transformer)
  - Evidence embeddings E = [e_1, ..., e_K] (from knowledge base retrieval)
  - Multi-head cross-attention: Q=h, K=V=E (4 heads, d_k=96)
  - Feed-forward fusion -> scalar verification score v in [0, 1]
  - Trained with triplet-style loss: (hypothesis, supporting_evidence, contradicting_evidence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    CAEVN_HEADS, CAEVN_DK, CAEVN_HIDDEN, CAEVN_EPOCHS,
    CAEVN_LR, CAEVN_WEIGHT_DECAY, CAEVN_BATCH_SIZE,
    CAEVN_DROPOUT, CAEVN_ENTROPY_BETA, EMBEDDING_DIM,
    MODEL_DIR, DEVICE,
)


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention: hypothesis attends to evidence."""

    def __init__(self, d_model, n_heads, d_k, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, n_heads * d_k)
        self.W_k = nn.Linear(d_model, n_heads * d_k)
        self.W_v = nn.Linear(d_model, n_heads * d_k)
        self.W_o = nn.Linear(n_heads * d_k, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: (B, 1, d_model) - hypothesis embedding
            key_value: (B, K, d_model) - evidence embeddings
            mask: (B, K) - True for valid evidence positions
        Returns:
            output: (B, 1, d_model) - attended representation
            attn_weights: (B, n_heads, 1, K) - attention weights
        """
        B = query.shape[0]
        Q = self.W_q(query).view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key_value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(key_value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).reshape(B, 1, self.n_heads * self.d_k)
        output = self.W_o(context)
        output = self.norm(output + query)
        return output, attn_weights


class CAEVN(nn.Module):
    """Cross-Attention Evidence Verification Network.

    Input: hypothesis embedding (384-dim) + K evidence embeddings (384-dim each)
    Output: verification score in [0, 1]
    """

    def __init__(self, d_model=EMBEDDING_DIM, n_heads=CAEVN_HEADS,
                 d_k=CAEVN_DK, hidden=CAEVN_HIDDEN, dropout=CAEVN_DROPOUT):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(d_model, n_heads, d_k, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Linear(hidden // 2, 1)

    def forward(self, hypothesis, evidence, evidence_mask=None):
        """
        Args:
            hypothesis: (B, d_model) - single hypothesis embedding
            evidence: (B, K, d_model) - K retrieved evidence embeddings
            evidence_mask: (B, K) - True for valid evidence
        Returns:
            score: (B,) - verification score in [0, 1]
            attn_weights: (B, n_heads, 1, K)
        """
        h = hypothesis.unsqueeze(1)
        attended, attn_weights = self.cross_attn(h, evidence, evidence_mask)
        attended = attended.squeeze(1)
        features = self.ffn(attended)
        score = torch.sigmoid(self.score_head(features)).squeeze(-1)
        return score, attn_weights


class MLPVerifier(nn.Module):
    """Ablation verifier (mlp_verifier variant): no cross-attention.

    Concatenates the hypothesis embedding with mean-pooled evidence and
    scores via the same 2-layer MLP head as CAEVN. Trained on the same
    triples by train_caevn.py (checkpoint caevn_{ds}_mlp.pt).
    """

    def __init__(self, d_model=EMBEDDING_DIM, hidden=CAEVN_HIDDEN,
                 dropout=CAEVN_DROPOUT):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(2 * d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Linear(hidden // 2, 1)

    def forward(self, hypothesis, evidence, evidence_mask=None):
        """Same signature as CAEVN.forward; attn weights are None."""
        pooled = evidence.mean(dim=1)
        features = self.ffn(torch.cat([hypothesis, pooled], dim=-1))
        score = torch.sigmoid(self.score_head(features)).squeeze(-1)
        return score, None


class CAEVNTrainer:
    """Training loop for CAEVN with triplet-style verification loss."""

    def __init__(self, model=None, device=None, epochs=CAEVN_EPOCHS,
                 lr=CAEVN_LR, entropy_beta=CAEVN_ENTROPY_BETA):
        self.device = device or DEVICE
        self.model = (model or CAEVN()).to(self.device)
        self.epochs = epochs
        self.entropy_beta = entropy_beta
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr,
            weight_decay=CAEVN_WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs)

    def _verification_loss(self, score_pos, score_neg, attn_pos):
        """Combined loss: margin ranking + attention entropy regularization."""
        margin_loss = F.margin_ranking_loss(
            score_pos, score_neg,
            target=torch.ones_like(score_pos),
            margin=0.3,
        )
        if attn_pos is None:  # MLPVerifier has no attention weights
            return margin_loss
        attn_flat = attn_pos.mean(dim=1).squeeze(1)
        entropy = -(attn_flat * (attn_flat + 1e-8).log()).sum(dim=-1).mean()
        return margin_loss + self.entropy_beta * entropy

    def train_model(self, train_triples, val_triples=None):
        """Train on (hypothesis, positive_evidence, negative_evidence) triples.

        Each triple: (h: [d], e_pos: [K, d], e_neg: [K, d])
        Packed as numpy arrays.
        """
        h_train, ep_train, en_train = train_triples
        dataset = TensorDataset(
            torch.tensor(h_train, dtype=torch.float32),
            torch.tensor(ep_train, dtype=torch.float32),
            torch.tensor(en_train, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=CAEVN_BATCH_SIZE, shuffle=True)
        best_val_loss = float("inf")

        for epoch in range(self.epochs):
            self.model.train(True)
            total_loss = 0
            for h, e_pos, e_neg in loader:
                h, e_pos, e_neg = (h.to(self.device), e_pos.to(self.device),
                                   e_neg.to(self.device))
                score_pos, attn_pos = self.model(h, e_pos)
                score_neg, _ = self.model(h, e_neg)
                loss = self._verification_loss(score_pos, score_neg, attn_pos)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()
            avg_loss = total_loss / len(loader)

            if val_triples is not None and (epoch + 1) % 5 == 0:
                val_loss = self._validate(val_triples)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save()
                print(f"Epoch {epoch+1}/{self.epochs}  "
                      f"train_loss={avg_loss:.4f}  val_loss={val_loss:.4f}")

    def _validate(self, val_triples):
        self.model.train(False)
        h, e_pos, e_neg = val_triples
        with torch.no_grad():
            h_t = torch.tensor(h, dtype=torch.float32).to(self.device)
            ep_t = torch.tensor(e_pos, dtype=torch.float32).to(self.device)
            en_t = torch.tensor(e_neg, dtype=torch.float32).to(self.device)
            s_pos, a_pos = self.model(h_t, ep_t)
            s_neg, _ = self.model(h_t, en_t)
            loss = self._verification_loss(s_pos, s_neg, a_pos)
        return loss.item()

    def verify(self, hypothesis_emb, evidence_embs):
        """Run inference: return verification score and attention weights.

        Args:
            hypothesis_emb: (d_model,) numpy array
            evidence_embs: (K, d_model) numpy array
        Returns:
            score: float in [0, 1]
            attn: (n_heads, K) numpy array
        """
        self.model.train(False)
        with torch.no_grad():
            h = torch.tensor(hypothesis_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
            e = torch.tensor(evidence_embs, dtype=torch.float32).unsqueeze(0).to(self.device)
            score, attn = self.model(h, e)
        if attn is None:  # MLPVerifier
            return score.item(), None
        return score.item(), attn.squeeze(0).squeeze(-2).cpu().numpy()

    def verify_each(self, hypothesis_emb, evidence_embs):
        """Score every evidence item independently in one batched pass."""
        self.model.train(False)
        evidence_embs = np.asarray(evidence_embs, dtype=np.float32)
        if len(evidence_embs) == 0:
            return np.empty(0, dtype=float)
        with torch.no_grad():
            h = torch.tensor(
                np.repeat(np.asarray(hypothesis_emb, dtype=np.float32)[None],
                          len(evidence_embs), axis=0),
                dtype=torch.float32, device=self.device)
            e = torch.tensor(evidence_embs[:, None, :],
                             dtype=torch.float32, device=self.device)
            scores, _ = self.model(h, e)
        return scores.detach().cpu().numpy()

    def save(self, path=None):
        path = path or MODEL_DIR / "caevn_best.pt"
        torch.save(self.model.state_dict(), path)

    def load(self, path=None):
        path = path or MODEL_DIR / "caevn_best.pt"
        self.model.load_state_dict(
            torch.load(path, map_location=self.device))
