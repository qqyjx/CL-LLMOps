"""Global configuration for CL-LLMOps experiments."""

import os
from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data" / "raw"
PROCESSED_DIR = EXPERIMENT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"
OPERATING_POINT_JSON = RESULTS_DIR / "operating_point.json"
MODEL_DIR = EXPERIMENT_DIR / "checkpoints"

# Ensure directories exist
for d in [DATA_DIR, PROCESSED_DIR, RESULTS_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset Config ──
DATASETS = {
    "pq16": {
        "name": "PQ Classification (16-class synthetic)",
        "n_classes": 16,
        "sample_rate_hz": 3200,
        "channels": 6,
        "window_size": 640,
    },
    "pq": {
        "name": "PQ Classification",
        "n_classes": 16,
        "sample_rate_hz": 3200,
        "channels": 6,  # V_abc + I_abc
        "window_size": 640,  # ~200ms at 3.2kHz
    },
    "cwru": {
        "name": "CWRU Bearing Fault",
        "n_classes": 10,
        "sample_rate_hz": 12000,
        "channels": 2,  # DE + FE accelerometer
        "window_size": 2048,
        "overlap": 0.5,
        "loads": [0, 1, 2, 3],  # HP
        "fault_sizes": [0.007, 0.014, 0.021],  # inches
    },
    "care": {
        "name": "CARE Wind Turbine SCADA",
        "n_classes": 2,  # normal vs pre-fault
        "interval_min": 10,
        "n_tags": 60,
        "window_steps": 144,  # 24 hours
    },
}

# ── Train/Val/Test Split ──
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# ── Feature Extraction ──
TIME_FEATURES = ["mean", "std", "rms", "kurtosis", "crest_factor", "peak", "skewness"]
FREQ_FEATURES = ["fft_peaks", "spectral_centroid", "spectral_bandwidth"]
BEARING_FEATURES = ["bpfi", "bpfo", "bsf", "ftf"]  # characteristic defect freqs

# ── LLM Config ──
LLM_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
LLM_MODEL_MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_BACKBONES = {
    "llama3": LLM_MODEL,
    "mistral7b": LLM_MODEL_MISTRAL,
}
LLM_TEMPERATURE_HYPOTHESIS = 0.3
LLM_TEMPERATURE_ACTION = 0.1
LLM_MAX_TOKENS = 8192
LLM_HYPOTHESIS_MAX_TOKENS = int(
    os.environ.get("ASOC_HYPOTHESIS_MAX_TOKENS", "512"))
LLM_ACTION_MAX_TOKENS = int(
    os.environ.get("ASOC_ACTION_MAX_TOKENS", "512"))
# Layer-3 audit-log explanation (post-gating, never enters the gate)
LLM_TEMPERATURE_EXPLANATION = 0.1
LLM_EXPLANATION_MAX_TOKENS = int(
    os.environ.get("ASOC_EXPLANATION_MAX_TOKENS", "256"))
EXPLAIN_AUDIT = os.environ.get(
    "ASOC_EXPLAIN_AUDIT", "1").lower() not in {"0", "false"}
SELF_CONSISTENCY_M = 5  # number of samples

# vLLM OpenAI-compatible server (set by scripts/start_vllm_server.sh)
VLLM_ENDPOINT = os.environ.get("VLLM_ENDPOINT", "http://127.0.0.1:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")

# LLM client mode: "vllm" (server) or "mock" (local deterministic
# stand-in from framework/llm_client.py; identical code path)
LLM_MODE = os.environ.get("ASOC_LLM_MODE", "vllm")

# ── Sentence Encoder ──
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── CAEVN (Cross-Attention Evidence Verifier) ──
CAEVN_HEADS = 4
CAEVN_DK = 96  # d_k per head = EMBEDDING_DIM / CAEVN_HEADS
CAEVN_HIDDEN = 256
CAEVN_EPOCHS = 50
CAEVN_LR = 1e-3
CAEVN_WEIGHT_DECAY = 1e-4
CAEVN_BATCH_SIZE = 64
CAEVN_DROPOUT = 0.2
CAEVN_ENTROPY_BETA = 0.01
CAEVN_TRAIN_TRIPLES_PER_CLASS = 1000  # per dataset, per class

# ── Fuzzy Inference System (Mamdani) ──
FIS_PROCEED_THRESHOLD = 0.7   # alpha_p
FIS_DEFER_THRESHOLD = 0.3     # alpha_d
FIS_HIGH_RISK_MARGIN = 0.1    # delta
RULE_NEURAL_WEIGHT = 0.4      # lambda_r
FIS_EVIDENCE_COVERAGE_THRESHOLD = 0.5

# ── MOPSO ──
MOPSO_SWARM_SIZE = 50
MOPSO_ITERATIONS = 100
MOPSO_ARCHIVE_SIZE = 30
MOPSO_INERTIA = (0.9, 0.4)   # linear decay
MOPSO_C1 = 2.0
MOPSO_C2 = 2.0

# ── Regression Testing ──
REGRESSION_PASS_THRESHOLD = 0.95
REGRESSION_CYCLES = 5
INITIAL_TEST_SET_SIZE = 50

# ── Evaluation Pipeline (framework/pipeline.py + evaluation/) ──
EVIDENCE_TOP_K = 5             # J retrieved evidence items (paper Sec. 5)
KB_CASES_PER_CLASS = 3         # channel-2 historical cases per class
ALARM_TRAIN_CAP = int(os.environ.get("ASOC_ALARM_TRAIN_CAP", "4000"))
# Stratified cap on episodes per evaluate() call (LLM cost control);
# classical baselines still use the full test split.
PIPELINE_EVAL_MAX_EPISODES = int(
    os.environ.get("ASOC_EVAL_MAX_EPISODES", "2000"))
# Fixed val-subset size cached for evaluate_with_params (MOPSO loop)
MOO_EVAL_SAMPLES = int(os.environ.get("ASOC_MOO_EVAL_SAMPLES", "120"))
MOO_MAX_M = 9                  # hypothesis samples cached per episode
REGRESSION_STRIDE = 10         # run Stage 5 every k-th episode in evaluate()
REGRESSION_SUITE_SIZE = 30     # regression cases drawn from val split

# ── Baseline Training ──
# Stratified cap on SVM training rows: an RBF kernel matrix is quadratic in
# n, so the full 168K pq16 training split is infeasible. 20K keeps the fit
# tractable (6 fits with probability=True) while covering all 16 classes.
SVM_MAX_TRAIN = int(os.environ.get("ASOC_SVM_MAX_TRAIN", "20000"))
BASELINE_EPOCHS = {"cnn": 100, "transformer": 100, "fdllm_lora": 30}
BASELINE_LR = {"cnn": 1e-3, "transformer": 1e-4, "fdllm_lora": 2e-5}

# ── Hardware ──
DEVICE = os.environ.get("ASOC_DEVICE", "cuda")
GPU_IDS = [int(x) for x in os.environ.get("ASOC_GPUS", "0").split(",") if x]
