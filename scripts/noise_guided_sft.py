#!/usr/bin/env python3
"""
Stage 2: Use found beneficial noise directions as initialization/regularization for SFT.

Loads noise search results, applies the best noise configuration as initialization,
then fine-tunes with TRL SFTTrainer while regularizing toward the noise-augmented state.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.noise_injection import (
    NoiseConfig,
    NoiseRegularizer,
    NoiseType,
    inject_noise,
    resolve_attn_implementation,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("noise_guided_sft")


def parse_args():
    parser = argparse.ArgumentParser(description="Noise-Guided SFT")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--noise_results", type=str, required=True,
                        help="Path to noise_search_results.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--target_subject", type=str, default=None,
                        help="Use best noise for this subject (default: overall best)")
    parser.add_argument("--noise_type", type=str, default=None)
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--noise_layer_group", type=str, default=None)
    parser.add_argument("--no_regularization", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def select_noise_config(noise_results: dict, cfg: dict, args) -> NoiseConfig:
    """Select the noise configuration to use for SFT initialization."""
    if args.noise_type and args.noise_scale and args.noise_layer_group:
        layer_indices = cfg["noise_search"]["layer_groups"].get(args.noise_layer_group)
        return NoiseConfig(
            noise_type=NoiseType(args.noise_type),
            scale=args.noise_scale,
            layer_indices=layer_indices,
            target_modules=cfg["noise_search"]["target_modules"],
        )

    best_per_subject = noise_results.get("best_per_subject", {})

    if args.target_subject and args.target_subject in best_per_subject:
        info = best_per_subject[args.target_subject]
        layer_indices = cfg["noise_search"]["layer_groups"].get(info["layer_group"])
        return NoiseConfig(
            noise_type=NoiseType(info["noise_type"]),
            scale=info["scale"],
            layer_indices=layer_indices,
            target_modules=cfg["noise_search"]["target_modules"],
        )

    best_key = None
    best_delta = -float("inf")
    for key, data in noise_results.items():
        if key.startswith("_") or key == "baseline" or key == "best_per_subject":
            continue
        if isinstance(data, dict) and "delta_from_baseline" in data:
            if data["delta_from_baseline"] > best_delta:
                best_delta = data["delta_from_baseline"]
                best_key = key

    if best_key and best_key in noise_results:
        info = noise_results[best_key]
        layer_indices = cfg["noise_search"]["layer_groups"].get(info["layer_group"])
        logger.info("Selected best overall config: %s (delta: %+.4f)", best_key, best_delta)
        return NoiseConfig(
            noise_type=NoiseType(info["noise_type"]),
            scale=info["scale"],
            layer_indices=layer_indices,
            target_modules=cfg["noise_search"]["target_modules"],
        )

    logger.warning("No beneficial noise found, using default Gaussian noise")
    return NoiseConfig(
        noise_type=NoiseType.GAUSSIAN,
        scale=0.01,
        target_modules=cfg["noise_search"]["target_modules"],
    )


def prepare_sft_dataset(cfg: dict, tokenizer):
    """Prepare SFT dataset (GSM8K formatted for instruction tuning)."""
    sft_cfg = cfg["noise_guided_sft"]
    dataset = load_dataset(sft_cfg["dataset"], "main", split="train")

    def format_for_sft(example):
        prompt = (
            "Solve the following math problem step by step.\n\n"
            f"Question: {example['question']}\n\n"
            f"Answer: {example['answer']}"
        )
        return {"text": prompt}

    dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)
    return dataset


class NoiseGuidedSFTTrainer(SFTTrainer):
    """SFT Trainer with noise regularization."""

    def __init__(self, *args, noise_regularizer: NoiseRegularizer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_regularizer = noise_regularizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        if self.noise_regularizer is not None:
            reg_loss = self.noise_regularizer.compute_loss(model)
            loss = loss + reg_loss

        if return_outputs:
            return loss, outputs
        return loss


def main():
    args = parse_args()
    cfg = load_config(args.config_path)
    sft_cfg = cfg["noise_guided_sft"]
    output_dir = args.output_dir or os.path.join(cfg["output"]["base_dir"], "noise_guided_sft")
    os.makedirs(output_dir, exist_ok=True)

    with open(args.noise_results) as f:
        noise_results = json.load(f)

    noise_config = select_noise_config(noise_results, cfg, args)
    logger.info("Noise config: type=%s, scale=%.4f", noise_config.noise_type.value, noise_config.scale)

    logger.info("Loading model: %s", cfg["model"]["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"], trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name_or_path"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        attn_implementation=resolve_attn_implementation(cfg["model"].get("attn_implementation", "flash_attention_2")),
        trust_remote_code=True,
    ).cuda()

    logger.info("Injecting noise as initialization")
    model, inject_stats = inject_noise(model, noise_config, seed=args.seed)
    logger.info("Injection stats: %d params, norm=%.4f, SNR=%.1f",
                inject_stats["num_injected"],
                inject_stats["total_noise_norm"],
                inject_stats["mean_snr"])

    noise_regularizer = None
    if not args.no_regularization:
        reg_weight = sft_cfg.get("noise_regularization_weight", 0.1)
        noise_regularizer = NoiseRegularizer(model, noise_config, weight=reg_weight)
        logger.info("Noise regularization enabled (weight=%.4f)", reg_weight)

    logger.info("Preparing SFT dataset")
    dataset = prepare_sft_dataset(cfg, tokenizer)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg["num_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=200,
        max_seq_length=sft_cfg["max_length"],
        dataset_text_field="text",
        report_to="wandb",
        run_name=f"specnoise-sft-{noise_config.noise_type.value}",
        dataloader_num_workers=4,
    )

    trainer = NoiseGuidedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        noise_regularizer=noise_regularizer,
    )

    logger.info("Starting noise-guided SFT")
    train_result = trainer.train()

    logger.info("Saving model to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    run_info = {
        "noise_config": {
            "type": noise_config.noise_type.value,
            "scale": noise_config.scale,
            "layer_indices": noise_config.layer_indices,
        },
        "inject_stats": {k: v for k, v in inject_stats.items() if k != "snr_values"},
        "regularization": not args.no_regularization,
        "training_metrics": metrics,
    }
    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    logger.info("Noise-guided SFT complete!")


if __name__ == "__main__":
    main()
