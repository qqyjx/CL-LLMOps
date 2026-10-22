# CL-LLMOps

Code and result files for the article

> Y. Xiao, D. Xu, S. Li, W. Ding, Y. Huang. **Fuzzy-gated verification of LLM agents for industrial fault diagnosis: An evolutionary-optimized closed-loop framework.** *Applied Soft Computing*, 204, 116560, 2027. https://doi.org/10.1016/j.asoc.2026.116560

CL-LLMOps routes an LLM diagnostic agent through five stages (alarm normalization, hypothesis generation, verification, action planning, regression testing). The verification layer combines rule checks, a cross-attention evidence verifier (CAEVN) and a Mamdani fuzzy gate that decides to proceed, escalate, or defer. MOPSO tunes the gate parameters.

## Contents

| Path | Contents |
|---|---|
| `experiments/framework/` | The five-stage pipeline, CAEVN, the fuzzy gate, the LLM client and prompt templates |
| `experiments/optimization/` | MOPSO and the objective functions |
| `experiments/baselines/` | SVM, XGBoost, ANFIS, 1D-CNN, Transformer, and the FD-LLM / T2MFDF reproductions |
| `experiments/data/` | Data loaders, the PQ generator, and the CWRU and CARE preparation scripts |
| `experiments/run_*.py` | One script per experiment (table below) |
| `results/` | The result files behind every table and figure of the article |
| `results/human_audit_packet.jsonl` | The 120-episode, disagreement-oversampled audit packet for human adjudication (Section 6.2) |
| `tests/` | Tests for the statistical protocol |

## Setup

Python 3.10 or later. The LLM stages need a CUDA GPU and a vLLM server; the article used one NVIDIA A800 80 GB GPU, PyTorch 2.10 and vLLM 0.19.

```bash
pip install -r experiments/requirements.txt
bash experiments/scripts/start_vllm_server.sh llama3 0      # serves Llama-3.1-8B-Instruct
export VLLM_ENDPOINT=http://127.0.0.1:8000/v1
```

`meta-llama/Llama-3.1-8B-Instruct` is a gated model on Hugging Face; request access first. `experiments/config.py` lists every setting and its environment-variable override.

## Data

The loaders read raw files from `experiments/data/raw/` and cache splits in `experiments/data/processed/`.

| Dataset | How to obtain |
|---|---|
| PQ (synthetic, 16 classes, 240,000 samples) | `python experiments/data/generate_pq16.py --n_per_class 15000 --seed 42` |
| CWRU bearing (10 classes, 40 files, 0–3 HP) | `bash experiments/data/download_cwru.sh` (checks the files against `cwru_CHECKSUMS.txt`) |
| CARE wind SCADA (95 datasets, Zenodo record 15846963, v6) | Download and extract the Zenodo archive, then `python experiments/data/prepare_care.py --care-root <extracted>/CARE_To_Compare` |

`prepare_care.py` keeps the `prediction` rows of each dataset, labels rows inside the anomaly event window as pre-fault, and keeps the 52 average channels with the highest non-zero fraction per farm. `care_conversion_manifest.json` records the selected channels and row counts of the run used in the article.

Known data issues, also stated in Section 5.1 of the article:

- CWRU: windows are cut with 50% overlap before the 70/15/15 split, so overlapping windows of one recording can fall in different partitions. The release file for the 2 HP normal baseline (`99.mat`) also contains the 1 HP recording, and the loader reads the first `DE_time` array, so the normal class contains three distinct recordings; 97 of the 883 test windows are identical to training windows.
- CARE: the split assigns whole datasets (files) to partitions; a turbine can contribute several datasets, and 6 of the 10 test turbines also contribute training datasets.

The code is released as it was run, so these issues are reproduced, not corrected.

## Reproducing the article

| Article item | Script | Result file |
|---|---|---|
| CAEVN verifiers | `experiments/train_caevn.py` | checkpoints (not released) |
| Fig. 5, deployed operating point | `run_mopso.py`, `plot_pareto_front.py` | `mopso_results.json`, `operating_point.json` |
| Tables 6–7, paired comparison and significance | `run_statistical_protocol.py` | `statistical_protocol.json` |
| Section 6.1, verifier-training resampling | `run_verifier_bootstrap.py` | `verifier_bootstrap.json` |
| Section 6.1, per-domain re-calibration | `per_domain_recalibration.py` | `per_domain_recalibration.json` |
| Table 8, backbones (Llama row from the ablation run) | `run_backbone.py`, `run_ablation.py` | `backbone_results.json`, `ablation_results.json` |
| Table 9, classical and deep baselines | `run_main.py` | `experiment_results.json` |
| Table 9, FD-LLM and T2MFDF reproductions | `run_recent_sota.py` | `recent_sota.json` |
| Section 6.2, independent judge and audit packet | `run_independent_judge.py`, `judge_robustness.py` | `independent_judge.json`, `judge_robustness.json`, `human_audit_packet.jsonl` |
| Table 10, verification-layer baselines | `run_verification_baselines.py` | `verification_baselines.json` |
| Table 11, ablation | `run_ablation.py` | `ablation_results.json` |
| Section 6.3, non-Ruspini robustness | `run_non_ruspini_robustness.py` | `non_ruspini_robustness.json` |
| Table 12, Fig. 6, MOPSO vs NSGA-II | `run_moo_comparison.py`, `plot_moo_distributions.py` | `moo_comparison.json` |
| Table 13, cross-domain regression gate | `run_regression_crossdomain.py` | `regression_crossdomain.json` |

Run each script from the repository root, for example `python experiments/run_statistical_protocol.py`; `--help` lists the options. Table numbers in some script docstrings refer to earlier drafts; the table above follows the published article.

The tables in the article are computed from the per-experiment files above at full precision, with sample standard deviations and a single half-up rounding to the printed digit. `results/results.json` is a 4-decimal summary kept for convenience. In the two result files that recorded server paths, the absolute path prefixes were removed; no other value was changed.

## License and citation

The code is released under the MIT License (`LICENSE`). Please cite the article above; `CITATION.cff` gives the reference in machine-readable form.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{xiao2027fuzzy,
  title = {Fuzzy-gated verification of LLM agents for industrial fault diagnosis: An evolutionary-optimized closed-loop framework},
  volume = {204},
  ISSN = {1568-4946},
  url = {http://dx.doi.org/10.1016/j.asoc.2026.116560},
  DOI = {10.1016/j.asoc.2026.116560},
  journal = {Applied Soft Computing},
  publisher = {Elsevier BV},
  author = {Xiao, Yifan and Xu, Dexuan and Li, Shijie and Ding, Weiping and Huang, Yu},
  year = {2027},
  month = Jan,
  pages = {116560},
}
```
