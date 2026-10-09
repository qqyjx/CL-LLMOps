"""Run recent FD-LLM and T2MFDF reimplementations for Reviewer 4.4."""

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from config import MODEL_DIR, RESULTS_DIR
from data import load_splits
from framework.pipeline import get_sentence_encoder
from baselines.recent_sota import (
    FDLLMTokenDataset, causal_collate, fdllm_prompt, parse_class_id,
    stratified_cap, train_t2mfdf,
)
from evaluation.action_rules import get_class_names


def train_fdllm(dataset_name, splits, model_path, seed,
                train_per_class, test_per_class, batch_size,
                grad_accum, epochs, max_length):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "FD-LLM requires PEFT; install experiments/requirements-r1.txt"
        ) from exc
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to("cuda")
    model.config.use_cache = False
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
        lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
        bias="none")
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    tr = stratified_cap(splits["y_train"], train_per_class, seed)
    te = stratified_cap(splits["y_test"], test_per_class, seed + 1)
    train_data = FDLLMTokenDataset(
        splits["X_train"][tr], splits["y_train"][tr], dataset_name,
        tokenizer, max_length=max_length)
    loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=partial(causal_collate,
                           pad_token_id=tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01)
    total_updates = max(1, epochs * len(loader) // grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates)
    model.train()
    optimizer.zero_grad()
    update = 0
    started = time.time()
    for epoch in range(epochs):
        running = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            loss = model(**batch).loss / grad_accum
            loss.backward()
            running += float(loss.detach()) * grad_accum
            if (step + 1) % grad_accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                update += 1
        print(f"[{dataset_name}] FD-LLM epoch={epoch+1}/{epochs} "
              f"loss={running/len(loader):.4f} updates={update}", flush=True)

    adapter_dir = MODEL_DIR / f"fdllm_{dataset_name}_seed{seed}"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    tokenizer.padding_side = "left"
    model.eval()
    prompts = [fdllm_prompt(x, dataset_name)
               for x in splits["X_test"][te]]
    preds = []
    eval_batch = max(1, min(8, batch_size * 4))
    with torch.no_grad():
        for start in range(0, len(prompts), eval_batch):
            batch_prompts = prompts[start:start + eval_batch]
            tokens = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length).to("cuda")
            generated = model.generate(
                **tokens, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
            suffix = generated[:, tokens["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(suffix, skip_special_tokens=True)
            preds.extend(parse_class_id(t, len(get_class_names(dataset_name)))
                         for t in texts)
    y_true = splits["y_test"][te]
    elapsed = time.time() - started
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, average="macro")),
        "top3_accuracy": None,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "seed": seed, "adapter": str(adapter_dir),
        "elapsed_seconds": elapsed,
        "trainable_parameters": int(sum(
            p.numel() for p in model.parameters() if p.requires_grad)),
    }
    del model
    torch.cuda.empty_cache()
    return metrics


def summarize(per_seed):
    keys = ("accuracy", "f1", "top3_accuracy")
    summary = {}
    for key in keys:
        vals = [row.get(key) for row in per_seed]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            summary[key] = {"mean": float(np.mean(vals)),
                            "std": float(np.std(vals)), "raw": vals}
        else:
            summary[key] = None
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+",
                        default=["pq16", "cwru", "care"])
    parser.add_argument("--methods", nargs="+",
                        choices=["fdllm", "t2mfdf"],
                        default=["fdllm", "t2mfdf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--model-path", type=Path,
                        default=Path(os.environ.get(
                            "ASOC_LLAMA_MODEL_PATH",
                            "meta-llama/Llama-3.1-8B-Instruct")))
    parser.add_argument("--fdllm-train-per-class", type=int, default=200)
    parser.add_argument("--fdllm-test-per-class", type=int, default=50)
    parser.add_argument("--fdllm-batch-size", type=int, default=2)
    parser.add_argument("--fdllm-grad-accum", type=int, default=8)
    parser.add_argument("--fdllm-epochs", type=int, default=3)
    parser.add_argument("--fdllm-max-length", type=int, default=512)
    parser.add_argument("--t2mfdf-train-per-class", type=int, default=500)
    parser.add_argument("--t2mfdf-val-per-class", type=int, default=100)
    parser.add_argument("--t2mfdf-test-per-class", type=int, default=200)
    parser.add_argument("--t2mfdf-epochs", type=int, default=50)
    parser.add_argument("--out", type=Path,
                        default=RESULTS_DIR / "recent_sota.json")
    args = parser.parse_args()

    encoder = get_sentence_encoder() if "t2mfdf" in args.methods else None
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reviewer_item": "R4.4",
            "reporting": "repository-split reimplementations; no numbers "
                         "copied from source papers",
            "fdllm": {
                "source": "arXiv:2412.01218",
                "protocol": "15 statistical features, rank-4 LoRA, "
                            "lr=1e-4, bf16, 3 epochs",
            },
            "t2mfdf": {
                "source": "IEEE TIM 2025",
                "doi": "10.1109/TIM.2025.3583374",
                "protocol": "time-series encoder + fault-description "
                            "BERT-family embeddings + multimodal fusion",
            },
            "seeds": args.seeds,
        },
        "datasets": {},
    }
    for dataset in args.datasets:
        splits = load_splits(dataset)
        payload["datasets"][dataset] = {}
        for method in args.methods:
            rows = []
            for seed in args.seeds:
                print(f"[{dataset}] {method} seed={seed}", flush=True)
                if method == "fdllm":
                    result = train_fdllm(
                        dataset, splits, args.model_path, seed,
                        args.fdllm_train_per_class,
                        args.fdllm_test_per_class,
                        args.fdllm_batch_size, args.fdllm_grad_accum,
                        args.fdllm_epochs, args.fdllm_max_length)
                else:
                    result = train_t2mfdf(
                        dataset, splits, encoder, seed=seed,
                        train_per_class=args.t2mfdf_train_per_class,
                        val_per_class=args.t2mfdf_val_per_class,
                        test_per_class=args.t2mfdf_test_per_class,
                        epochs=args.t2mfdf_epochs,
                        checkpoint=MODEL_DIR /
                        f"t2mfdf_{dataset}_seed{seed}.pt")
                rows.append(result)
            payload["datasets"][dataset][method] = {
                "per_seed": rows, "summary": summarize(rows)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
