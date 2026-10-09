"""LLM inference clients for the CL-LLMOps pipeline.

Two interchangeable clients behind the same `generate()` interface, so
the pipeline code path is IDENTICAL in local and server runs except
for the client object:

  * VLLMClient  -- OpenAI-compatible HTTP client against the vLLM
    server started by scripts/start_vllm_server.sh (server/A800 path).
  * MockLLMClient -- deterministic, seedable local stand-in that
    parses the pipeline's actual prompts and emits plausible
    structured JSON, including a controllable fraction of wrong
    diagnoses and hallucinated evidence citations so that downstream
    metrics are non-degenerate. NO metric is computed here; the mock
    only produces outputs which the evaluation judge and metric code
    score honestly.

Select via env ASOC_LLM_MODE=vllm|mock (config.LLM_MODE) or by
passing a client instance to the pipeline / baselines.
"""

import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    LLM_MODEL, LLM_MODE, VLLM_ENDPOINT, VLLM_API_KEY, RANDOM_SEED,
)
from evaluation.action_rules import (
    get_class_names, severity_of, action_level_for, SEVERITY_TO_ACTION,
)


class BaseLLMClient:
    """Minimal client interface used by HypothesisGenerator/ActionGenerator."""

    def generate(self, prompt, temperature=0.3, max_tokens=512):
        raise NotImplementedError

    def generate_n(self, prompt, n, temperature=0.3, max_tokens=512):
        """Generate ``n`` independent completions for one prompt.

        Backends may override this with a native batched request.  The
        fallback keeps custom clients compatible with the pipeline.
        """
        return [self.generate(prompt, temperature=temperature,
                              max_tokens=max_tokens) for _ in range(n)]


class VLLMClient(BaseLLMClient):
    """OpenAI-compatible completion client for the vLLM server.

    Server-only in practice (requires a running vLLM endpoint); kept
    here so the pipeline wiring is identical to the mock path.
    """

    def __init__(self, endpoint=VLLM_ENDPOINT, model=LLM_MODEL,
                 api_key=VLLM_API_KEY, seed=None, timeout=None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.seed = seed
        # Generous default: under the campaign's worker pool a request may
        # sit in the vLLM queue behind many concurrent decoding jobs.
        self.timeout = timeout or int(
            os.environ.get("ASOC_LLM_TIMEOUT", "600"))
        self._request_index = 0

    def reseed(self, seed):
        """Reset the deterministic request stream for a decoding run."""
        self.seed = seed
        self._request_index = 0

    def generate_n(self, prompt, n, temperature=0.3, max_tokens=512):
        # Chat endpoint (2026-07-23): raw /completions on an instruct model
        # skips the chat template, and Llama-3.1 then continues the prompt
        # (echoing instruction bullets and emitting an empty "example
        # response" JSON before the real one) — which broke severity
        # parsing campaign-wide. /chat/completions applies the model's
        # template so the model answers instead of continuing.
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": int(n),
        }
        if self.seed is not None:
            # Distinct requests receive distinct deterministic streams;
            # vLLM itself draws independent samples for choices within n.
            payload["seed"] = int(self.seed) + self._request_index
        self._request_index += 1
        req = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            })
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = sorted(body["choices"], key=lambda c: c.get("index", 0))
        texts = [choice["message"]["content"] for choice in choices]
        if len(texts) != int(n):
            raise RuntimeError(
                f"vLLM returned {len(texts)} choices for requested n={n}")
        return texts

    def generate(self, prompt, temperature=0.3, max_tokens=512):
        return self.generate_n(prompt, n=1, temperature=temperature,
                               max_tokens=max_tokens)[0]


# ── Mock client ──

# Controllable output-quality knobs (fractions of episodes). These do
# NOT set any metric directly: they only shape the mock's raw text
# outputs, which the independent evaluation judge then scores.
MOCK_WRONG_DIAGNOSIS_RATE = 0.15   # diagnosis differs from alarm top-1
MOCK_HALLUCINATION_RATE = 0.12     # fabricated evidence citation
MOCK_SEVERITY_NOISE_RATE = 0.10    # severity off by one category
MOCK_NUMERIC_FABRICATION_SCALE = (2.5, 5.0)

_SEVERITY_ORDER = ["none", "low", "medium", "high", "critical"]


class MockLLMClient(BaseLLMClient):
    """Deterministic, seedable stand-in for the vLLM backend.

    Parses the actual HYPOTHESIS_PROMPT / ACTION_PROMPT templates used
    by framework.hypothesis / framework.action and returns structured
    JSON of the same shape the real backbone is prompted for. All
    randomness flows through a single RandomState so runs are exactly
    reproducible given (dataset_name, seed) and the call sequence.
    """

    def __init__(self, dataset_name="pq", seed=RANDOM_SEED,
                 wrong_rate=MOCK_WRONG_DIAGNOSIS_RATE,
                 halluc_rate=MOCK_HALLUCINATION_RATE,
                 severity_noise=MOCK_SEVERITY_NOISE_RATE):
        self.dataset_name = dataset_name
        self.class_names = get_class_names(dataset_name)
        self.wrong_rate = wrong_rate
        self.halluc_rate = halluc_rate
        self.severity_noise = severity_noise
        self.reseed(seed)

    def reseed(self, seed):
        """Reset the RNG (decoding-seed plumbing for repeated runs)."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    # -- prompt parsing helpers --

    @staticmethod
    def _parse_field(prompt, pattern, cast=str, default=None):
        m = re.search(pattern, prompt)
        return cast(m.group(1)) if m else default

    def _parse_top3(self, prompt):
        return re.findall(r"\((\d+), '([^']+)', ([0-9.]+)\)", prompt)

    # -- generation --

    def generate(self, prompt, temperature=0.3, max_tokens=512):
        if "diagnostic hypothesis" in prompt:
            return self._generate_hypothesis(prompt, temperature)
        if "maintenance action plan" in prompt:
            return self._generate_action(prompt)
        if "audit log" in prompt:
            return self._generate_explanation(prompt)
        return "{}"

    def _generate_explanation(self, prompt):
        """Deterministic canned Layer-3 explanation (audit-log text)."""
        decision = self._parse_field(
            prompt, r"Gating decision:\s*(\w+)", default="proceed")
        v_score = self._parse_field(
            prompt, r"Combined score:\s*([0-9.]+)", cast=float, default=0.0)
        return (f"The combined verification score {v_score:.2f} and the "
                f"listed evidence tags are consistent with the gating "
                f"decision '{decision}'. Mock audit explanation; no "
                f"evidence beyond the listed items was considered.")

    def _generate_hypothesis(self, prompt, temperature):
        fault_name = self._parse_field(
            prompt, r"Fault type:\s*(.+)", default=self.class_names[0]).strip()
        feat_mean = self._parse_field(
            prompt, r"mean=(-?\d+\.?\d*(?:[eE][-+]?\d+)?)",
            cast=float, default=0.0)
        top3 = self._parse_top3(prompt)

        # Diagnosis: mostly the alarm class; a temperature-scaled
        # fraction picks a competitor from the alarm top-3 (or any
        # other class), emulating decoding noise.
        p_wrong = self.wrong_rate * max(temperature / 0.3, 0.25)
        diagnosis = fault_name
        if self.rng.rand() < p_wrong:
            alt = [name for _, name, _ in top3 if name != fault_name]
            if alt and self.rng.rand() < 0.8:
                diagnosis = alt[int(self.rng.randint(len(alt)))]
            else:
                others = [c for c in self.class_names if c != fault_name]
                diagnosis = others[int(self.rng.randint(len(others)))]

        # Severity: standard-derived for the diagnosed class, with an
        # occasional off-by-one category slip.
        try:
            severity = severity_of(self.dataset_name, diagnosis)
        except KeyError:
            severity = "medium"
        if self.rng.rand() < self.severity_noise:
            idx = _SEVERITY_ORDER.index(severity)
            step = 1 if (idx == 0 or (idx < 4 and self.rng.rand() < 0.5)) \
                else -1
            severity = _SEVERITY_ORDER[idx + step]

        # Evidence citations: honest ones reference the diagnosed
        # class's signature and echo the actual feature mean; a
        # hallucinated one cites an unrelated class and fabricates the
        # numeric claim.
        cited_class, cited_mean = diagnosis, feat_mean
        if self.rng.rand() < self.halluc_rate:
            others = [c for c in self.class_names
                      if c.lower() != diagnosis.lower()]
            cited_class = others[int(self.rng.randint(len(others)))]
            lo, hi = MOCK_NUMERIC_FABRICATION_SCALE
            cited_mean = feat_mean * self.rng.uniform(lo, hi) + 1.0
        evidence_cited = [
            f"Knowledge-base signature match for {cited_class} pattern",
            f"Observed feature mean={cited_mean:.4f} consistent with "
            f"{cited_class} profile",
        ]

        return json.dumps({
            "diagnosis": diagnosis,
            "root_cause": f"Physical mechanism consistent with "
                          f"{diagnosis} under operating stress",
            "severity": severity,
            "evidence_cited": evidence_cited,
            "inspection_points": [
                f"primary {self.dataset_name} sensing chain",
                "upstream supply/drive-train interface",
            ],
            "cascading_effects": f"Unaddressed {diagnosis} may degrade "
                                 f"adjacent components",
        })

    def _generate_action(self, prompt):
        # The upstream hypothesis JSON is embedded in the action prompt
        # ("Root cause hypothesis: {...}"); recover diagnosis+severity
        # and map them through the deterministic rule table, matching
        # paper Section 4's fault-type x severity action mapping.
        diagnosis, severity = None, None
        m = re.search(r"Root cause hypothesis:\s*(\{.*\})", prompt,
                      re.DOTALL)
        if m:
            try:
                hyp = json.loads(m.group(1))
                diagnosis = hyp.get("diagnosis")
                severity = hyp.get("severity")
            except json.JSONDecodeError:
                pass
        if diagnosis is None:
            diagnosis = self._parse_field(
                prompt, r"Fault type:\s*(.+)",
                default=self.class_names[0]).strip()
        try:
            level = action_level_for(self.dataset_name, diagnosis,
                                     severity=severity)
        except (KeyError, ValueError):
            level = SEVERITY_TO_ACTION.get(severity, "A2")

        return json.dumps({
            "action_level": level,
            "immediate_actions": [f"isolate and verify {diagnosis} "
                                  f"indication"],
            "short_term_repairs": ["execute standard remediation "
                                   "procedure for the diagnosed fault"],
            "preventive_measures": ["add condition-monitoring trigger"],
            "required_parts": ["per maintenance manual"],
            "safety_precautions": ["lockout-tagout before intervention"],
        })


def get_default_client(dataset_name="pq", seed=None):
    """Build the default LLM client per config.LLM_MODE.

    ASOC_LLM_MODE=mock  -> MockLLMClient (local, deterministic)
    ASOC_LLM_MODE=vllm  -> VLLMClient against VLLM_ENDPOINT (server)
    """
    if seed is None:
        import config as config_module
        seed = config_module.RANDOM_SEED
    if LLM_MODE == "mock":
        return MockLLMClient(dataset_name=dataset_name, seed=seed)
    return VLLMClient(seed=seed)
