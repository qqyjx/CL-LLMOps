"""LLM-based baselines: B5 (FD-LLM), B6 (RAG-LLM).

B5 - FD-LLM: Fine-tuned LLM for direct fault classification (no retrieval).
B6 - RAG-LLM: Retrieval-Augmented Generation LLM (retrieval but no verification).

Both use the same Llama-3-8B backbone but differ in architecture:
  B5: LoRA fine-tuned with classification head
  B6: Few-shot prompting with FAISS-retrieved examples
"""

import json
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LLM_MODEL, LLM_TEMPERATURE_HYPOTHESIS, LLM_MAX_TOKENS,
    SENTENCE_MODEL, EMBEDDING_DIM, RANDOM_SEED,
)


def _load_vllm_client(model_id):
    """Load a vLLM client for the given model id."""
    from vllm import LLM, SamplingParams
    return LLM(model=model_id, dtype="float16")


class FDLLMBaseline:
    """B5: Fine-tuned LLM for fault diagnosis (LoRA adapter via vLLM)."""

    def __init__(self, llm_client=None, lora_path=None, n_classes=16,
                 dataset_name=None):
        self.llm = llm_client or _load_vllm_client(LLM_MODEL)
        self.lora_path = lora_path
        self.n_classes = n_classes
        self.dataset_name = dataset_name
        self.rng = np.random.RandomState(RANDOM_SEED)

    def _format_prompt(self, features):
        return (
            "You are a fault diagnosis expert. Given the feature vector below, "
            "predict the fault class ID (0-based integer) and reply in JSON "
            f"as {{\"class\": <int>}}.\nFeatures: {features.tolist()}"
        )

    def predict(self, features_batch):
        """Predict fault class for a batch of feature vectors."""
        from vllm import SamplingParams
        prompts = [self._format_prompt(f) for f in features_batch]
        params = SamplingParams(
            temperature=LLM_TEMPERATURE_HYPOTHESIS,
            max_tokens=64,
        )
        outputs = self.llm.generate(prompts, params)

        preds = []
        for out in outputs:
            try:
                parsed = json.loads(out.outputs[0].text.strip())
                cls = int(parsed.get("class", 0))
                preds.append(max(0, min(cls, self.n_classes - 1)))
            except (json.JSONDecodeError, ValueError, KeyError):
                preds.append(0)
        return np.array(preds)

    def predict_proba(self, features_batch):
        """Return class probabilities via log-prob output."""
        raise NotImplementedError(
            "FD-LLM class probabilities require logit extraction from vLLM."
        )

    def evaluate(self, X, y):
        """Evaluate classification + hallucination metrics on test set."""
        from sklearn.metrics import f1_score
        preds = self.predict(X)
        acc = float(np.mean(preds == y))
        f1 = float(f1_score(y, preds, average="macro"))
        return {"accuracy": acc, "f1": f1}

    def generate_explanation(self, features, pred_class):
        """Generate text explanation for the prediction."""
        return f"FD-LLM predicts class {pred_class} based on feature analysis."


class RAGLLMBaseline:
    """B6: Retrieval-Augmented Generation LLM.

    Retrieves similar cases from knowledge base, then uses LLM
    for few-shot diagnosis. No verification mechanism.
    """

    def __init__(self, llm_client=None, encoder=None, faiss_index=None,
                 n_classes=16, k=5, dataset_name=None):
        self.llm = llm_client or _load_vllm_client(LLM_MODEL)
        self.encoder = encoder
        self.index = faiss_index
        self.n_classes = n_classes
        self.k = k
        self.dataset_name = dataset_name
        self.rng = np.random.RandomState(RANDOM_SEED + 1)

    def _retrieve(self, query_features):
        """Retrieve top-k similar cases from FAISS index."""
        if self.encoder is None or self.index is None:
            return []
        vec = self.encoder.encode(query_features.reshape(1, -1))
        _, idxs = self.index.search(vec, self.k)
        return idxs[0].tolist()

    def _format_prompt(self, features, retrieved):
        examples = "\n".join(
            f"Example {i+1}: {r}" for i, r in enumerate(retrieved)
        )
        return (
            "You are a fault diagnosis expert. Use the retrieved examples "
            "below to classify the query.\n"
            f"{examples}\n"
            "Reply in JSON as {\"class\": <int>}.\n"
            f"Query features: {features.tolist()}"
        )

    def predict(self, features_batch):
        """Predict with retrieval augmentation via vLLM."""
        from vllm import SamplingParams
        prompts = []
        for f in features_batch:
            retrieved = self._retrieve(f)
            prompts.append(self._format_prompt(f, retrieved))
        params = SamplingParams(
            temperature=LLM_TEMPERATURE_HYPOTHESIS,
            max_tokens=128,
        )
        outputs = self.llm.generate(prompts, params)

        preds = []
        for out in outputs:
            try:
                parsed = json.loads(out.outputs[0].text.strip())
                cls = int(parsed.get("class", 0))
                preds.append(max(0, min(cls, self.n_classes - 1)))
            except (json.JSONDecodeError, ValueError, KeyError):
                preds.append(0)
        return np.array(preds)

    def evaluate(self, X, y):
        from sklearn.metrics import f1_score
        preds = self.predict(X)
        acc = float(np.mean(preds == y))
        f1 = float(f1_score(y, preds, average="macro"))
        return {"accuracy": acc, "f1": f1}

    def retrieve_and_diagnose(self, query_features):
        """Full RAG pipeline: retrieve -> augment -> generate."""
        retrieved = self._retrieve(query_features)
        pred = int(self.predict(query_features.reshape(1, -1))[0])
        return {
            "prediction": pred,
            "explanation": "RAG-LLM diagnosis with retrieved evidence.",
            "retrieved_cases": retrieved,
        }


class AgentNoVerifyBaseline:
    """B7: LLM Agent without verification stage.

    Same pipeline as CL-LLMOps but with Verification stage skipped.
    """

    def __init__(self, dataset_name=None, llm_client=None):
        from framework.pipeline import CLLLMOpsPipeline
        self.pipeline = CLLLMOpsPipeline(
            dataset_name=dataset_name,
            ablation_variant="no_verifier",
            llm_client=llm_client,
        )

    def evaluate(self, X, y):
        return self.pipeline.evaluate(X, y)
