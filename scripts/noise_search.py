#!/usr/bin/env python3
"""
Systematic noise injection search.

Iterate over layers x scales x types (Gaussian, SVD-structured, low-rank).
Evaluate on 20+ MMLU subdomains per configuration to find beneficial noise patterns.
"""

import argparse
import copy
import json
import logging
import os
import sys
from itertools import product
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.noise_injection import NoiseConfig, NoiseType, inject_noise, resolve_attn_implementation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("noise_search")


def parse_args():
    parser = argparse.ArgumentParser(description="SpecNoise Grid Search")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--noise_types", nargs="+", default=None)
    parser.add_argument("--scales", nargs="+", type=float, default=None)
    parser.add_argument("--layer_groups", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def eval_mmlu_subject(model, tokenizer, subject: str, max_samples: int = 100) -> dict:
    """Evaluate on a single MMLU subject."""
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
        question = ex["question"]
        opts = ex.get("choices", [])
        prompt = f"Question: {question}\n"
        for i, opt in enumerate(opts):
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


def evaluate_all_subjects(model, tokenizer, subjects: list,
                          max_samples: int = 100) -> dict:
    """Evaluate on all subjects and return per-subject and overall accuracy."""
    results = {}
    accs = []
    for subject in subjects:
        r = eval_mmlu_subject(model, tokenizer, subject, max_samples)
        results[subject] = r
        if not r.get("error"):
            accs.append(r["accuracy"])

    results["_overall"] = {
        "mean_accuracy": float(sum(accs) / len(accs)) if accs else 0,
        "num_subjects": len(accs),
    }
    return results


def _noise_type_extra_kwargs_variants(noise_type: str, cfg: dict) -> list:
    """Return a list of extra_kw dicts so each rank/top_k gets its own search run."""
    ns = cfg["noise_search"]
    if noise_type == "low_rank":
        return [{"low_rank_rank": r} for r in ns["low_rank"]["ranks"]]
    if noise_type == "svd_structured":
        return [{"svd_top_k": k} for k in ns["svd_structured"]["top_k_singular"]]
    return [{}]


def run_grid_search(model, tokenizer, cfg, args):
    """Run full grid search over noise configurations."""
    noise_types = args.noise_types or cfg["noise_search"]["noise_types"]
    scales = args.scales or cfg["noise_search"]["scales"]
    layer_group_names = args.layer_groups or list(cfg["noise_search"]["layer_groups"].keys())
    subjects = cfg["evaluation"]["mmlu_subjects"]
    max_samples = cfg["evaluation"]["max_samples_per_subject"]
    target_modules = cfg["noise_search"]["target_modules"]

    original_state = copy.deepcopy(model.state_dict())

    logger.info("Evaluating baseline model")
    baseline = evaluate_all_subjects(model, tokenizer, subjects, max_samples)
    baseline_acc = baseline["_overall"]["mean_accuracy"]
    logger.info("Baseline accuracy: %.4f", baseline_acc)

    all_results = {"baseline": baseline}
    best_configs = {}

    total_configs = (
        sum(len(_noise_type_extra_kwargs_variants(nt, cfg)) for nt in noise_types)
        * len(scales)
        * len(layer_group_names)
    )
    config_idx = 0

    for noise_type, scale, layer_group_name in product(noise_types, scales, layer_group_names):
        layer_indices = cfg["noise_search"]["layer_groups"].get(layer_group_name)

        for extra_kwargs in _noise_type_extra_kwargs_variants(noise_type, cfg):
            config_idx += 1
            logger.info(
                "Config %d/%d: type=%s, scale=%.4f, layers=%s, extra=%s",
                config_idx,
                total_configs,
                noise_type,
                scale,
                layer_group_name,
                repr(extra_kwargs),
            )

            noise_cfg = NoiseConfig(
                noise_type=NoiseType(noise_type),
                scale=scale,
                layer_indices=layer_indices,
                target_modules=target_modules,
                low_rank_rank=extra_kwargs.get("low_rank_rank", 4),
                svd_top_k=extra_kwargs.get("svd_top_k", 4),
            )

            suffix_parts = []
            if "low_rank_rank" in extra_kwargs:
                suffix_parts.append(f"r{extra_kwargs['low_rank_rank']}")
            if "svd_top_k" in extra_kwargs:
                suffix_parts.append(f"k{extra_kwargs['svd_top_k']}")
            variant_suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

            model.load_state_dict(original_state)

            model, inject_stats = inject_noise(model, noise_cfg, seed=args.seed)

            results = evaluate_all_subjects(model, tokenizer, subjects, max_samples)
            mean_acc = results["_overall"]["mean_accuracy"]
            delta = mean_acc - baseline_acc

            config_key = f"{noise_type}_s{scale}_{layer_group_name}{variant_suffix}"
            all_results[config_key] = {
                "noise_type": noise_type,
                "scale": scale,
                "layer_group": layer_group_name,
                "extra_kwargs": extra_kwargs,
                "inject_stats": {k: v for k, v in inject_stats.items() if k != "snr_values"},
                "mean_accuracy": mean_acc,
                "delta_from_baseline": delta,
                "per_subject": results,
            }

            improved_subjects = []
            degraded_subjects = []
            for subject in subjects:
                base_acc = baseline.get(subject, {}).get("accuracy", 0)
                new_acc = results.get(subject, {}).get("accuracy", 0)
                if new_acc > base_acc + 0.02:
                    improved_subjects.append((subject, new_acc - base_acc))
                elif new_acc < base_acc - 0.02:
                    degraded_subjects.append((subject, base_acc - new_acc))

            all_results[config_key]["improved_subjects"] = improved_subjects
            all_results[config_key]["degraded_subjects"] = degraded_subjects

            logger.info(
                "  Mean acc: %.4f (delta: %+.4f), improved: %d, degraded: %d",
                mean_acc, delta, len(improved_subjects), len(degraded_subjects),
            )

            for subject, improvement in improved_subjects:
                if subject not in best_configs or improvement > best_configs[subject]["improvement"]:
                    best_configs[subject] = {
                        "config_key": config_key,
                        "noise_type": noise_type,
                        "scale": scale,
                        "layer_group": layer_group_name,
                        "extra_kwargs": extra_kwargs,
                        "improvement": improvement,
                    }

    model.load_state_dict(original_state)

    all_results["best_per_subject"] = best_configs

    return all_results


def main():
    args = parse_args()
    cfg = load_config(args.config_path)
    output_dir = args.output_dir or cfg["output"]["base_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model: %s", cfg["model"]["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"], trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name_or_path"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=resolve_attn_implementation(cfg["model"].get("attn_implementation", "flash_attention_2")),
    )
    model.eval()

    logger.info("Starting noise grid search")
    results = run_grid_search(model, tokenizer, cfg, args)

    out_path = os.path.join(output_dir, "noise_search_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("NOISE SEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info("Baseline: %.4f", results["baseline"]["_overall"]["mean_accuracy"])

    if "best_per_subject" in results:
        logger.info("Best noise configs per subject:")
        for subject, info in sorted(results["best_per_subject"].items()):
            logger.info("  %s: %s (improvement: +%.3f)",
                        subject, info["config_key"], info["improvement"])

    logger.info("Full results saved to %s", out_path)


if __name__ == "__main__":
    main()
