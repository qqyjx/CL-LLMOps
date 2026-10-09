import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from framework.llm_client import VLLMClient
from framework.verification.calibrated_gate import CalibratedProbabilityGate
from framework.verification.baseline_verifiers import (
    NLIEntailmentVerifier, SelfConsistencyGuardrailVerifier,
)
from framework.verification.fuzzy_gating import MamdaniFIS
from baselines.recent_sota import (
    STAT_NAMES, T2MFDFModel, parse_class_id, statistical_features,
)
from run_statistical_protocol import resample_splits


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class _FakeNLI:
    id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}

    def __call__(self, pair):
        assert "text" in pair and "text_pair" in pair
        return [[
            {"label": "contradiction", "score": 0.05},
            {"label": "neutral", "score": 0.10},
            {"label": "entailment", "score": 0.85},
        ]]


class R1ProtocolTests(unittest.TestCase):
    def test_vllm_native_n_and_seed_stream(self):
        calls = []

        urls = []

        def fake_urlopen(req, timeout):
            payload = json.loads(req.data)
            calls.append(payload)
            urls.append(req.full_url)
            return _Response({"choices": [
                {"index": i, "message": {"content": f"sample-{i}"}}
                for i in reversed(range(payload["n"]))
            ]})

        client = VLLMClient(endpoint="http://unit.test/v1", seed=7)
        with mock.patch("urllib.request.urlopen", fake_urlopen):
            self.assertEqual(
                client.generate_n("p", n=3),
                ["sample-0", "sample-1", "sample-2"])
            client.generate("p2")
            client.reseed(100)
            client.generate("p3")
        self.assertEqual([c["seed"] for c in calls], [7, 8, 100])
        self.assertEqual([c["n"] for c in calls], [3, 1, 1])
        # chat endpoint applies the model's template (2026-07-23 fix);
        # raw /completions made Llama-3.1 continue the prompt instead of
        # answering it.
        self.assertTrue(all(u.endswith("/chat/completions") for u in urls))
        self.assertEqual(calls[0]["messages"],
                         [{"role": "user", "content": "p"}])

    def test_calibrated_gate_matches_all_proceed_endpoint(self):
        gate = CalibratedProbabilityGate("platt").fit(
            [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1],
            ["proceed"] * 4)
        decisions = [gate.infer(v, 0.5, 0.5)[0]
                     for v in [0.1, 0.2, 0.8, 0.9]]
        self.assertEqual(decisions, ["proceed"] * 4)

    def test_guardrail_and_nli_scores_are_bounded(self):
        text = json.dumps({
            "diagnosis": "IR",
            "root_cause": "bearing defect",
            "severity": "high",
            "evidence_cited": ["IR signature"],
        })
        docs = [{"tag": "IR", "text": "IR bearing signature"}]
        guard, _ = SelfConsistencyGuardrailVerifier().score(
            text, docs, consistency=0.8, class_names=["Normal", "IR"])
        nli, _ = NLIEntailmentVerifier(backend=_FakeNLI()).score(text, docs)
        self.assertTrue(0.0 <= guard <= 1.0)
        self.assertAlmostEqual(nli, 0.85)

    def test_evidence_gate_distinguishes_defer_and_escalate(self):
        gate = MamdaniFIS()
        self.assertEqual(gate.infer(0.9, coverage=0.9,
                                    contradiction=0.0)[0], "proceed")
        self.assertEqual(gate.infer(0.9, coverage=0.1,
                                    contradiction=0.0)[0], "defer")
        self.assertEqual(gate.infer(0.55, coverage=0.55,
                                    contradiction=0.0)[0], "reject")
        self.assertEqual(gate.infer(0.9, coverage=0.9,
                                    contradiction=0.8)[0], "reject")

    def test_statistical_protocol_never_resamples_training_artifacts(self):
        splits = {
            "X_train": np.arange(12).reshape(6, 2),
            "y_train": np.array([0, 0, 0, 1, 1, 1]),
            "X_val": np.arange(8).reshape(4, 2) + 100,
            "y_val": np.array([0, 0, 1, 1]),
            "X_test": np.arange(16).reshape(8, 2) + 200,
            "y_test": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        }
        sampled = resample_splits(splits, 17)
        np.testing.assert_array_equal(sampled["X_train"], splits["X_train"])
        np.testing.assert_array_equal(sampled["X_val"], splits["X_val"])
        self.assertEqual(np.bincount(sampled["y_test"]).tolist(), [4, 4])
        self.assertTrue(np.all(sampled["X_test"] >= 200))

    def test_recent_sota_feature_and_fusion_shapes(self):
        rng = np.random.default_rng(3)
        signal = rng.normal(size=(2, 128)).astype(np.float32)
        features = statistical_features(signal)
        self.assertEqual(features.shape, (len(STAT_NAMES),))
        self.assertTrue(np.isfinite(features).all())
        model = T2MFDFModel(2, rng.normal(size=(4, 24)), hidden=16)
        logits = model(torch.tensor(signal[None], dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertEqual(parse_class_id('{"class_id": 3}', 4), 3)


if __name__ == "__main__":
    unittest.main()
