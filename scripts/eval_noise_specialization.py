#!/usr/bin/env python3
"""
Evaluate noise-injected and noise-guided SFT models across MMLU domains.

Compares:
1. Baseline (no noise)
2. Noise-only (after injection, no SFT)
3. Noise + SFT (after noise-guided SFT)
4. SFT-only (standard SFT without noise)
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_noise_spec")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate noise specialization")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_paths", nargs="+", required=True,
                        help="Model paths to evaluate (label:path format)")
    parser.add_argument("--output_dir", type=str, default="./results/eval_specialization")
    parser.add_argument("--max_samples", type=int, default=100)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def eval_mmlu_subject(model, tokenizer, subject, max_samples=100):
    choices = ["A", "B", "C", "D"]
    try:
        ds = load_dataset("cais/mmlu", subject, split="test")
    except Exception:
        try:
            ds = load_dataset("cais/mmlu", subject, split="validation")
        except Exception:
            return {"accuracy": 0.0, "total": 0, "error": True}

    if max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    correct, total = 0, 0
    for ex in ds:
        prompt = f"Question: {ex['question']}\n"
        for i, opt in enumerate(ex.get("choices", [])):
            prompt += f"{choices[i]}. {opt}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=8, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_ids = outputs[:, inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip().upper()[:1]

        gold_idx = ex.get("answer", 0)
        gold = choices[gold_idx] if isinstance(gold_idx, int) and gold_idx < 4 else str(gold_idx).strip().upper()[:1]

        correct += int(answer == gold)
        total += 1

    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def evaluate_model(model, tokenizer, subjects, max_samples):
    results = {}
    accs = []
    for subject in tqdm(subjects, desc="Subjects"):
        r = eval_mmlu_subject(model, tokenizer, subject, max_samples)
        results[subject] = r
        if not r.get("error"):
            accs.append(r["accuracy"])
    results["_overall"] = {
        "mean_accuracy": float(np.mean(accs)) if accs else 0,
        "std_accuracy": float(np.std(accs)) if accs else 0,
        "num_subjects": len(accs),
    }
    return results


def compute_specialization_metrics(all_results: dict, subjects: list) -> dict:
    """Compare models and compute specialization metrics."""
    labels = list(all_results.keys())
    if len(labels) < 2:
        return {}

    baseline_label = labels[0]
    baseline = all_results[baseline_label]

    metrics = {}
    for label in labels[1:]:
        model_results = all_results[label]

        improvements = []
        degradations = []
        per_subject_delta = {}

        for subject in subjects:
            base_acc = baseline.get(subject, {}).get("accuracy", 0)
            model_acc = model_results.get(subject, {}).get("accuracy", 0)
            delta = model_acc - base_acc
            per_subject_delta[subject] = delta

            if delta > 0.01:
                improvements.append((subject, delta))
            elif delta < -0.01:
                degradations.append((subject, delta))

        deltas = list(per_subject_delta.values())

        gain_subjects = [s for s, d in per_subject_delta.items() if d > 0.02]
        loss_subjects = [s for s, d in per_subject_delta.items() if d < -0.02]

        specificity = len(gain_subjects) / max(len(subjects), 1)
        collateral = len(loss_subjects) / max(len(subjects), 1)

        mean_gain = np.mean([d for d in deltas if d > 0]) if any(d > 0 for d in deltas) else 0
        mean_loss = np.mean([abs(d) for d in deltas if d < 0]) if any(d < 0 for d in deltas) else 0
        selectivity = mean_gain / max(mean_gain + mean_loss, 1e-8)

        metrics[label] = {
            "overall_delta": model_results.get("_overall", {}).get("mean_accuracy", 0) -
                            baseline.get("_overall", {}).get("mean_accuracy", 0),
            "num_improved": len(improvements),
            "num_degraded": len(degradations),
            "num_unchanged": len(subjects) - len(improvements) - len(degradations),
            "specificity": float(specificity),
            "collateral_damage": float(collateral),
            "selectivity": float(selectivity),
            "per_subject_delta": per_subject_delta,
            "top_improvements": sorted(improvements, key=lambda x: -x[1])[:5],
            "top_degradations": sorted(degradations, key=lambda x: x[1])[:5],
        }

    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config_path)
    os.makedirs(args.output_dir, exist_ok=True)

    subjects = cfg["evaluation"]["mmlu_subjects"]

    all_results = {}

    for model_spec in args.model_paths:
        if ":" in model_spec:
            label, path = model_spec.split(":", 1)
        else:
            label = os.path.basename(model_spec)
            path = model_spec

        logger.info("Evaluating model: %s (%s)", label, path)

        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        results = evaluate_model(model, tokenizer, subjects, args.max_samples)
        all_results[label] = results

        logger.info("  %s overall: %.4f", label, results["_overall"]["mean_accuracy"])

        with open(os.path.join(args.output_dir, f"eval_{label}.json"), "w") as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()

    spec_metrics = compute_specialization_metrics(all_results, subjects)

    summary = {
        "models": {},
        "specialization": spec_metrics,
    }
    for label, results in all_results.items():
        summary["models"][label] = results.get("_overall", {})

    with open(os.path.join(args.output_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for label, results in all_results.items():
        overall = results.get("_overall", {})
        logger.info("  %s: mean_acc=%.4f (std=%.4f)",
                    label, overall.get("mean_accuracy", 0), overall.get("std_accuracy", 0))

    if spec_metrics:
        logger.info("\nSPECIALIZATION METRICS:")
        for label, m in spec_metrics.items():
            logger.info("  %s:", label)
            logger.info("    Overall delta: %+.4f", m["overall_delta"])
            logger.info("    Improved: %d, Degraded: %d", m["num_improved"], m["num_degraded"])
            logger.info("    Selectivity: %.3f", m["selectivity"])
            if m["top_improvements"]:
                logger.info("    Top improvements: %s",
                            [(s, f"+{d:.3f}") for s, d in m["top_improvements"][:3]])

    logger.info("Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
