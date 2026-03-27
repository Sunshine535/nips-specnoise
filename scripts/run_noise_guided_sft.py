#!/usr/bin/env python3
"""
Noise-guided SFT with four training strategies across multiple domains.

Strategies:
  A. standard_sft        -- baseline, no noise
  B. pre_noise_sft       -- inject noise then fine-tune
  C. noise_regularized   -- NoiseRegularizer active during SFT
  D. iterative_noise     -- re-inject noise every N steps
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.noise_injection import (
    NoiseConfig, NoiseRegularizer, NoiseType, inject_noise,
    resolve_attn_implementation,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_noise_guided_sft")

DOMAIN_DATASETS = {
    "medical": {
        "path": "Open-Orca/SlimOrca",
        "keywords": ["medical", "health", "disease", "patient", "diagnosis",
                      "treatment", "symptom", "clinical", "doctor", "medicine"],
    },
    "legal": {
        "path": "Open-Orca/SlimOrca",
        "keywords": ["law", "legal", "court", "statute", "regulation",
                      "contract", "attorney", "judge", "liability", "plaintiff"],
    },
    "code": {
        "path": "Open-Orca/SlimOrca",
        "keywords": ["code", "python", "function", "program", "algorithm",
                      "debug", "class", "variable", "javascript", "software"],
    },
    "finance": {
        "path": "Open-Orca/SlimOrca",
        "keywords": ["finance", "stock", "investment", "revenue", "profit",
                      "market", "banking", "economic", "portfolio", "interest rate"],
    },
}

STRATEGIES = ["standard_sft", "pre_noise_sft", "noise_regularized", "iterative_noise"]


def parse_args():
    p = argparse.ArgumentParser(description="Noise-Guided SFT (4 strategies x N domains)")
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--noise_results", type=str, required=True,
                   help="Path to noise_search_results.json")
    p.add_argument("--output_dir", type=str, default="./results/noise_guided_sft")
    p.add_argument("--domains", nargs="+", default=None)
    p.add_argument("--strategies", nargs="+", default=None, choices=STRATEGIES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--iterative_noise_interval", type=int, default=100,
                   help="Re-inject noise every N steps (strategy D)")
    p.add_argument("--noise_reg_weight", type=float, default=0.1)
    p.add_argument("--max_train_samples", type=int, default=5000)
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def select_noise_config(noise_results, cfg):
    """Pick the best-performing noise config from grid-search results."""
    best_key, best_delta = None, -float("inf")
    for key, data in noise_results.items():
        if key.startswith("_") or key in ("baseline", "best_per_subject"):
            continue
        if isinstance(data, dict) and "delta_from_baseline" in data:
            if data["delta_from_baseline"] > best_delta:
                best_delta = data["delta_from_baseline"]
                best_key = key
    if best_key and best_key in noise_results:
        info = noise_results[best_key]
        layer_indices = cfg["noise_search"]["layer_groups"].get(info["layer_group"])
        logger.info("Selected noise config: %s (delta: %+.4f)", best_key, best_delta)
        return NoiseConfig(
            noise_type=NoiseType(info["noise_type"]),
            scale=info["scale"],
            layer_indices=layer_indices,
            target_modules=cfg["noise_search"]["target_modules"],
            low_rank_rank=info.get("extra_kwargs", {}).get("low_rank_rank", 4),
            svd_top_k=info.get("extra_kwargs", {}).get("svd_top_k", 4),
        )
    logger.warning("No beneficial noise found, defaulting to gaussian 0.01")
    return NoiseConfig(noise_type=NoiseType.GAUSSIAN, scale=0.01,
                       target_modules=cfg["noise_search"]["target_modules"])


def load_domain_dataset(domain, max_samples):
    """Load and filter dataset for a specific domain via keyword matching."""
    dcfg = DOMAIN_DATASETS[domain]
    keywords = dcfg["keywords"]
    try:
        ds = load_dataset(dcfg["path"], split="train", streaming=True)
        records = []
        for ex in ds:
            blob = json.dumps(ex.get("conversations", "")).lower()
            if any(kw in blob for kw in keywords):
                convs = ex.get("conversations", [])
                parts = []
                for turn in convs:
                    role = turn.get("from", turn.get("role", ""))
                    val = turn.get("value", turn.get("content", ""))
                    if role in ("human", "user"):
                        parts.append(f"### Instruction:\n{val}")
                    elif role in ("gpt", "assistant"):
                        parts.append(f"### Response:\n{val}")
                if parts:
                    records.append({"text": "\n\n".join(parts)})
                if len(records) >= max_samples:
                    break

        if len(records) < 100:
            logger.warning("Only %d samples for %s, padding with GSM8K",
                           len(records), domain)
            gsm = load_dataset("openai/gsm8k", "main", split="train")
            for ex in gsm:
                records.append({
                    "text": (f"### Instruction:\n{ex['question']}\n\n"
                             f"### Response:\n{ex['answer']}")
                })
                if len(records) >= max_samples:
                    break

        return Dataset.from_list(records[:max_samples])
    except Exception as e:
        logger.warning("Failed %s: %s, falling back to GSM8K", dcfg["path"], e)
        ds = load_dataset("openai/gsm8k", "main", split="train")
        recs = [{"text": (f"### Instruction:\n{x['question']}\n\n"
                          f"### Response:\n{x['answer']}")} for x in ds]
        return Dataset.from_list(recs[:max_samples])


class SNRTracker(TrainerCallback):
    """Log weight-level signal-to-noise ratio over training."""

    def __init__(self, orig_state, log_every=50):
        self.orig = orig_state
        self.log_every = log_every
        self.history = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_every != 0 or model is None:
            return
        vals = []
        for n, p in model.named_parameters():
            if n in self.orig:
                ref = self.orig[n].to(p.device)
                d = (p.data.float() - ref.float()).norm()
                w = ref.float().norm()
                if d > 1e-8:
                    vals.append((w / d).item())
        if vals:
            avg = sum(vals) / len(vals)
            self.history.append({"step": state.global_step, "mean_snr": avg})


class IterativeNoiseInjector(TrainerCallback):
    """Re-inject noise periodically (Strategy D)."""

    def __init__(self, ncfg, interval, seed):
        self.ncfg = ncfg
        self.interval = interval
        self.seed = seed
        self.count = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.interval != 0 or state.global_step == 0:
            return
        if model is None:
            return
        self.count += 1
        inject_noise(model, self.ncfg, seed=self.seed + self.count * 1000)
        logger.info("Step %d: re-injected noise (#%d)", state.global_step, self.count)


class RegularizedSFTTrainer(SFTTrainer):
    """SFTTrainer subclass that adds noise-regularization loss (Strategy C)."""

    def __init__(self, *a, noise_regularizer=None, **kw):
        super().__init__(*a, **kw)
        self.noise_regularizer = noise_regularizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if self.noise_regularizer is not None:
            loss = loss + self.noise_regularizer.compute_loss(model)
        return (loss, outputs) if return_outputs else loss


def run_strategy(strategy, model_name, noise_config, dataset, out_dir, args):
    """Execute one (strategy, domain, seed) combination."""
    logger.info("=" * 70)
    logger.info("Strategy: %s  |  Output: %s", strategy, out_dir)
    logger.info("=" * 70)
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                        padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation=resolve_attn_implementation(),
    ).cuda()

    orig_state = {n: p.data.clone().cpu()
                  for n, p in model.named_parameters()
                  if p.requires_grad and p.dim() == 2}

    inject_stats = None
    noise_reg = None
    cbs = []

    if strategy == "pre_noise_sft":
        model, inject_stats = inject_noise(model, noise_config, seed=args.seed)
    elif strategy == "noise_regularized":
        model, inject_stats = inject_noise(model, noise_config, seed=args.seed)
        noise_reg = NoiseRegularizer(model, noise_config, weight=args.noise_reg_weight)
    elif strategy == "iterative_noise":
        model, inject_stats = inject_noise(model, noise_config, seed=args.seed)
        cbs.append(IterativeNoiseInjector(noise_config,
                                          args.iterative_noise_interval, args.seed))

    snr_cb = SNRTracker(orig_state)
    cbs.append(snr_cb)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM, bias="none",
    )

    sft_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,
    )

    Cls = RegularizedSFTTrainer if noise_reg else SFTTrainer
    kw = dict(model=model, args=sft_args, train_dataset=dataset,
              processing_class=tok, peft_config=lora, callbacks=cbs)
    if noise_reg:
        kw["noise_regularizer"] = noise_reg

    trainer = Cls(**kw)
    result = trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    info = {
        "strategy": strategy,
        "noise_config": {
            "type": noise_config.noise_type.value,
            "scale": noise_config.scale,
            "layer_indices": noise_config.layer_indices,
        },
        "inject_stats": (
            {k: v for k, v in inject_stats.items() if k != "snr_values"}
            if inject_stats else None
        ),
        "training_metrics": {
            k: v for k, v in result.metrics.items()
            if isinstance(v, (int, float, str))
        },
        "snr_history": snr_cb.history,
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)

    del model, trainer
    torch.cuda.empty_cache()
    return info


def main():
    args = parse_args()
    cfg = load_cfg(args.config_path)

    with open(args.noise_results) as f:
        noise_results = json.load(f)

    noise_config = select_noise_config(noise_results, cfg)
    model_name = cfg["model"]["name_or_path"]
    domains = args.domains or list(DOMAIN_DATASETS.keys())
    strategies = args.strategies or STRATEGIES

    logger.info("Model: %s", model_name)
    logger.info("Domains: %s  Strategies: %s", domains, strategies)
    logger.info("Noise: type=%s scale=%.4f",
                noise_config.noise_type.value, noise_config.scale)

    all_results = {}
    for domain in domains:
        logger.info("\n" + "#" * 70)
        logger.info("DOMAIN: %s", domain)
        logger.info("#" * 70)

        dataset = load_domain_dataset(domain, args.max_train_samples)
        logger.info("Loaded %d samples for %s", len(dataset), domain)
        all_results[domain] = {}

        for strategy in strategies:
            od = os.path.join(args.output_dir, domain, strategy, f"seed{args.seed}")
            all_results[domain][strategy] = run_strategy(
                strategy, model_name, noise_config, dataset, od, args,
            )

    summary = os.path.join(args.output_dir, f"all_strategies_seed{args.seed}.json")
    with open(summary, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nAll training complete. Summary: %s", summary)


if __name__ == "__main__":
    main()
