#!/usr/bin/env python3
"""
Fisher information landscape analysis.

Computes diagonal Fisher information for each layer, compares before/after
noise injection, and identifies which layers are most sensitive.
Outputs: fisher_heatmap.pdf, layer_sensitivity.pdf
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
logger = logging.getLogger("fisher_analysis")


def parse_args():
    p = argparse.ArgumentParser(description="Fisher Information Analysis")
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--noise_results", type=str, default=None,
                   help="noise_search_results.json (to pick best noise config)")
    p.add_argument("--output_dir", type=str, default="./results/fisher_analysis")
    p.add_argument("--num_samples", type=int, default=200,
                   help="Number of text samples for Fisher estimation")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--noise_type", type=str, default="gaussian")
    p.add_argument("--noise_scale", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_layer_param_groups(model):
    """Group parameters by transformer layer index and module type."""
    groups = defaultdict(dict)
    for name, param in model.named_parameters():
        if not param.requires_grad or param.dim() != 2:
            continue
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is None:
            continue
        module = parts[-2] if len(parts) >= 2 else parts[-1]
        groups[layer_idx][module] = name
    return dict(groups)


def compute_diagonal_fisher(model, tokenizer, texts, max_length=256):
    """Estimate diagonal Fisher information via gradient squared averages."""
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() == 2:
            fisher[name] = torch.zeros_like(param, dtype=torch.float32)

    device = next(model.parameters()).device
    n_samples = 0
    for text in tqdm(texts, desc="Computing Fisher"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=max_length).to(device)
        if inputs["input_ids"].shape[1] < 2:
            continue
        model.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()
        n_samples += 1

        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += param.grad.data.float().pow(2)

    for name in fisher:
        fisher[name] /= max(n_samples, 1)

    return fisher


def fisher_to_layer_summary(fisher, layer_groups):
    """Reduce per-param Fisher to per-(layer, module) scalar summary."""
    summary = {}
    for layer_idx, modules in sorted(layer_groups.items()):
        summary[layer_idx] = {}
        for module_name, param_name in modules.items():
            if param_name in fisher:
                val = fisher[param_name].mean().item()
                summary[layer_idx][module_name] = val
    return summary


def plot_fisher_heatmap(summary_before, summary_after, output_path):
    """Plot Fisher information as a heatmap (layers x modules)."""
    layers = sorted(set(summary_before.keys()) | set(summary_after.keys()))
    modules = sorted(set(
        m for s in [summary_before, summary_after]
        for layer_data in s.values() for m in layer_data
    ))

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(layers) * 0.3)),
                             sharey=True)

    for ax, summary, title in [
        (axes[0], summary_before, "Before Noise"),
        (axes[1], summary_after, "After Noise"),
    ]:
        matrix = np.zeros((len(layers), len(modules)))
        for i, layer in enumerate(layers):
            for j, mod in enumerate(modules):
                matrix[i, j] = summary.get(layer, {}).get(mod, 0.0)

        log_matrix = np.log10(matrix + 1e-12)
        im = ax.imshow(log_matrix, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Module")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(len(modules)))
        ax.set_xticklabels(modules, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=8)
        fig.colorbar(im, ax=ax, label="log10(Fisher)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved heatmap: %s", output_path)


def plot_layer_sensitivity(summary_before, summary_after, output_path):
    """Plot per-layer sensitivity: how much Fisher changes after noise."""
    layers = sorted(set(summary_before.keys()) & set(summary_after.keys()))
    deltas = []
    ratios = []
    for layer in layers:
        before_mean = np.mean(list(summary_before[layer].values())) if summary_before[layer] else 0
        after_mean = np.mean(list(summary_after[layer].values())) if summary_after[layer] else 0
        deltas.append(after_mean - before_mean)
        ratios.append(after_mean / max(before_mean, 1e-12))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    colors = ["#e74c3c" if d > 0 else "#3498db" for d in deltas]
    ax1.bar(range(len(layers)), deltas, color=colors, alpha=0.8)
    ax1.set_ylabel("Fisher Delta (after - before)")
    ax1.set_title("Layer Sensitivity to Noise Injection")
    ax1.axhline(y=0, color="black", linewidth=0.5)

    ax2.bar(range(len(layers)), ratios, color="#2ecc71", alpha=0.8)
    ax2.set_ylabel("Fisher Ratio (after / before)")
    ax2.set_xlabel("Layer Index")
    ax2.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, fontsize=8, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sensitivity plot: %s", output_path)


def main():
    args = parse_args()
    cfg = load_cfg(args.config_path)
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = cfg["model"]["name_or_path"]
    logger.info("Loading model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=resolve_attn_implementation(),
    ).cuda()

    logger.info("Loading text samples for Fisher estimation")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 50][:args.num_samples]
    except Exception:
        ds = load_dataset("openai/gsm8k", "main", split="train")
        texts = [f"{ex['question']} {ex['answer']}" for ex in ds][:args.num_samples]

    layer_groups = get_layer_param_groups(model)
    logger.info("Found %d transformer layers", len(layer_groups))

    # Fisher BEFORE noise
    logger.info("Computing Fisher (before noise)...")
    fisher_before = compute_diagonal_fisher(model, tokenizer, texts, args.max_length)
    summary_before = fisher_to_layer_summary(fisher_before, layer_groups)
    del fisher_before
    torch.cuda.empty_cache()

    # Inject noise
    if args.noise_results:
        with open(args.noise_results) as f:
            nr = json.load(f)
        best_key = max(
            ((k, v) for k, v in nr.items()
             if isinstance(v, dict) and "delta_from_baseline" in v),
            key=lambda x: x[1]["delta_from_baseline"],
            default=(None, None),
        )[0]
        if best_key:
            info = nr[best_key]
            noise_cfg = NoiseConfig(
                noise_type=NoiseType(info["noise_type"]),
                scale=info["scale"],
                layer_indices=cfg["noise_search"]["layer_groups"].get(info["layer_group"]),
                target_modules=cfg["noise_search"]["target_modules"],
            )
        else:
            noise_cfg = NoiseConfig(noise_type=NoiseType(args.noise_type),
                                    scale=args.noise_scale,
                                    target_modules=cfg["noise_search"]["target_modules"])
    else:
        noise_cfg = NoiseConfig(noise_type=NoiseType(args.noise_type),
                                scale=args.noise_scale,
                                target_modules=cfg["noise_search"]["target_modules"])

    logger.info("Injecting noise: type=%s, scale=%.4f",
                noise_cfg.noise_type.value, noise_cfg.scale)
    model, inject_stats = inject_noise(model, noise_cfg, seed=args.seed)

    # Fisher AFTER noise
    logger.info("Computing Fisher (after noise)...")
    fisher_after = compute_diagonal_fisher(model, tokenizer, texts, args.max_length)
    summary_after = fisher_to_layer_summary(fisher_after, layer_groups)
    del fisher_after
    torch.cuda.empty_cache()

    # Plots
    plot_fisher_heatmap(summary_before, summary_after,
                        os.path.join(args.output_dir, "fisher_heatmap.pdf"))
    plot_layer_sensitivity(summary_before, summary_after,
                           os.path.join(args.output_dir, "layer_sensitivity.pdf"))

    # Identify most sensitive layers
    sensitivity = {}
    for layer in sorted(set(summary_before.keys()) & set(summary_after.keys())):
        b = np.mean(list(summary_before[layer].values())) if summary_before[layer] else 0
        a = np.mean(list(summary_after[layer].values())) if summary_after[layer] else 0
        sensitivity[layer] = {"before": b, "after": a,
                              "delta": a - b, "ratio": a / max(b, 1e-12)}

    top_sensitive = sorted(sensitivity.items(), key=lambda x: abs(x[1]["delta"]),
                           reverse=True)[:10]

    results = {
        "noise_config": {"type": noise_cfg.noise_type.value,
                         "scale": noise_cfg.scale},
        "inject_stats": {k: v for k, v in inject_stats.items() if k != "snr_values"},
        "num_fisher_samples": len(texts),
        "layer_sensitivity": {str(k): v for k, v in sensitivity.items()},
        "top_sensitive_layers": [(k, v) for k, v in top_sensitive],
        "summary_before": {str(k): v for k, v in summary_before.items()},
        "summary_after": {str(k): v for k, v in summary_after.items()},
    }
    with open(os.path.join(args.output_dir, "fisher_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("FISHER ANALYSIS COMPLETE")
    logger.info("Most noise-sensitive layers:")
    for layer, info in top_sensitive[:5]:
        logger.info("  Layer %d: delta=%.2e, ratio=%.2f", layer, info["delta"], info["ratio"])
    logger.info("Results: %s", args.output_dir)


if __name__ == "__main__":
    main()
