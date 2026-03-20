#!/usr/bin/env python3
"""
Domain-specific evaluation for noise-augmented models.

Evaluates on: MedQA, LegalBench, HumanEval, FinBench, and MMLU (general).
Compares noise-augmented vs baseline and reports retention ratio.
"""

import argparse
import json
import logging
import os
import re

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_domain_performance")


def parse_args():
    p = argparse.ArgumentParser(description="Domain Performance Evaluation")
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--model_paths", nargs="+", required=True,
                   help="label:path pairs, e.g. 'baseline:Qwen/Qwen3.5-9B'")
    p.add_argument("--output_dir", type=str, default="./results/domain_eval")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--domains", nargs="+",
                   default=["medical", "legal", "code", "finance", "mmlu"])
    p.add_argument("--base_model", type=str, default=None,
                   help="Base model for LoRA checkpoints")
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(label, path, base_model=None):
    """Load model, trying LoRA adapter first, then standalone."""
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if base_model and base_model != path:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        try:
            model = PeftModel.from_pretrained(model, path)
            logger.info("Loaded LoRA adapter for %s from %s", label, path)
        except Exception:
            logger.info("No LoRA found for %s, using base model loaded from %s", label, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
    model.eval()
    return model, tok


@torch.no_grad()
def generate_answer(model, tok, prompt, max_new_tokens=128):
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=2048).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def eval_mcqa(model, tok, dataset, prompt_template, answer_key="answer",
              choices_key="choices", max_samples=200):
    """Generic multiple-choice QA evaluation."""
    choice_labels = ["A", "B", "C", "D", "E"]
    samples = list(dataset)[:max_samples]
    correct, total = 0, 0
    for ex in tqdm(samples, desc="MCQA", leave=False):
        q = ex.get("question", ex.get("input", ""))
        opts = ex.get(choices_key, [])
        prompt = prompt_template.format(question=q)
        if opts:
            for i, o in enumerate(opts):
                if i < len(choice_labels):
                    prompt += f"\n{choice_labels[i]}. {o}"
            prompt += "\nAnswer:"

        answer = generate_answer(model, tok, prompt, max_new_tokens=8)
        pred = answer.strip().upper()[:1]

        gold = ex.get(answer_key, 0)
        if isinstance(gold, int) and gold < len(choice_labels):
            gold = choice_labels[gold]
        else:
            gold = str(gold).strip().upper()[:1]

        correct += int(pred == gold)
        total += 1
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def eval_medical(model, tok, max_samples):
    """Evaluate on MedQA-style medical questions."""
    try:
        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    except Exception:
        try:
            ds = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split="test")
        except Exception:
            logger.warning("MedQA not available, falling back to MMLU clinical_knowledge")
            ds = load_dataset("cais/mmlu", "clinical_knowledge", split="test")
            return eval_mcqa(model, tok, ds,
                             "Medical question:\n{question}", max_samples=max_samples)
    return eval_mcqa(model, tok, ds, "Medical question:\n{question}",
                     max_samples=max_samples)


def eval_legal(model, tok, max_samples):
    """Evaluate on LegalBench-style legal reasoning."""
    try:
        ds = load_dataset("nguha/legalbench", "rule_qa", split="test")
        samples = list(ds)[:max_samples]
        correct, total = 0, 0
        for ex in tqdm(samples, desc="Legal", leave=False):
            prompt = f"Legal question:\n{ex.get('text', ex.get('question', ''))}\nAnswer:"
            answer = generate_answer(model, tok, prompt, max_new_tokens=64)
            gold = str(ex.get("answer", ex.get("label", "")))
            if gold.lower() in answer.lower():
                correct += 1
            total += 1
        return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}
    except Exception:
        logger.warning("LegalBench not available, falling back to MMLU jurisprudence")
        ds = load_dataset("cais/mmlu", "jurisprudence", split="test")
        return eval_mcqa(model, tok, ds,
                         "Legal question:\n{question}", max_samples=max_samples)


def eval_code(model, tok, max_samples):
    """Evaluate on HumanEval-style code generation (pass@1 approximation)."""
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        logger.warning("HumanEval not available, using MMLU college_computer_science")
        ds = load_dataset("cais/mmlu", "college_computer_science", split="test")
        return eval_mcqa(model, tok, ds,
                         "CS question:\n{question}", max_samples=max_samples)

    samples = list(ds)[:max_samples]
    correct, total = 0, 0
    for ex in tqdm(samples, desc="Code", leave=False):
        prompt = ex["prompt"]
        answer = generate_answer(model, tok, prompt, max_new_tokens=256)
        full_code = prompt + answer
        test_code = ex.get("test", "")
        entry = ex.get("entry_point", "")
        try:
            namespace = {}
            exec(full_code, namespace)
            exec(test_code, namespace)
            exec(f"check({entry})", namespace)
            correct += 1
        except Exception:
            pass
        total += 1
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}


def eval_finance(model, tok, max_samples):
    """Evaluate on finance knowledge."""
    try:
        ds = load_dataset("sujet-ai/Sujet-Finance-QA-RAG", split="test")
        samples = list(ds)[:max_samples]
        correct, total = 0, 0
        for ex in tqdm(samples, desc="Finance", leave=False):
            q = ex.get("question", ex.get("input", ""))
            prompt = f"Finance question:\n{q}\nAnswer:"
            answer = generate_answer(model, tok, prompt, max_new_tokens=64)
            gold = str(ex.get("answer", ex.get("output", "")))
            if gold.lower()[:20] in answer.lower() or answer.lower()[:20] in gold.lower():
                correct += 1
            total += 1
        return {"accuracy": correct / max(total, 1), "correct": correct, "total": total}
    except Exception:
        logger.warning("FinBench not available, using MMLU econometrics")
        ds = load_dataset("cais/mmlu", "econometrics", split="test")
        return eval_mcqa(model, tok, ds,
                         "Finance/economics question:\n{question}", max_samples=max_samples)


def eval_mmlu_general(model, tok, max_samples):
    """Evaluate on MMLU for general capability retention."""
    subjects = ["abstract_algebra", "college_physics", "machine_learning",
                "philosophy", "high_school_psychology", "computer_security",
                "clinical_knowledge", "jurisprudence", "econometrics", "sociology"]
    all_accs = []
    per_subject = {}
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
        except Exception:
            continue
        r = eval_mcqa(model, tok, ds, "Question:\n{question}", max_samples=max_samples // len(subjects))
        per_subject[subj] = r
        all_accs.append(r["accuracy"])
    return {
        "accuracy": sum(all_accs) / max(len(all_accs), 1),
        "per_subject": per_subject,
        "num_subjects": len(all_accs),
    }


EVAL_FUNCTIONS = {
    "medical": eval_medical,
    "legal": eval_legal,
    "code": eval_code,
    "finance": eval_finance,
    "mmlu": eval_mmlu_general,
}


def main():
    args = parse_args()
    cfg = load_cfg(args.config_path)
    os.makedirs(args.output_dir, exist_ok=True)

    base_model = args.base_model or cfg["model"]["name_or_path"]
    all_results = {}

    for model_spec in args.model_paths:
        if ":" in model_spec:
            label, path = model_spec.split(":", 1)
        else:
            label = os.path.basename(model_spec)
            path = model_spec

        logger.info("\n" + "=" * 60)
        logger.info("Evaluating: %s (%s)", label, path)
        logger.info("=" * 60)

        model, tok = load_model(label, path, base_model)
        all_results[label] = {}

        for domain in args.domains:
            if domain not in EVAL_FUNCTIONS:
                logger.warning("Unknown domain: %s", domain)
                continue
            logger.info("  Domain: %s", domain)
            result = EVAL_FUNCTIONS[domain](model, tok, args.max_samples)
            all_results[label][domain] = result
            logger.info("    Accuracy: %.4f", result["accuracy"])

        del model
        torch.cuda.empty_cache()

    # Compute retention ratios
    labels = list(all_results.keys())
    if len(labels) >= 2:
        baseline_label = labels[0]
        baseline = all_results[baseline_label]
        for label in labels[1:]:
            model_res = all_results[label]
            retention = {}
            for domain in args.domains:
                if domain in baseline and domain in model_res:
                    b_acc = baseline[domain]["accuracy"]
                    m_acc = model_res[domain]["accuracy"]
                    retention[domain] = {
                        "baseline": b_acc,
                        "model": m_acc,
                        "delta": m_acc - b_acc,
                        "retention_ratio": m_acc / max(b_acc, 1e-8),
                    }
            all_results[f"retention_{label}"] = retention

    out_path = os.path.join(args.output_dir, "domain_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    logger.info("\n" + "=" * 80)
    header = f"{'Model':>20s}"
    for d in args.domains:
        header += f" {d:>10s}"
    logger.info(header)
    logger.info("-" * 80)
    for label in labels:
        row = f"{label:>20s}"
        for d in args.domains:
            acc = all_results[label].get(d, {}).get("accuracy", 0)
            row += f" {acc:>10.4f}"
        logger.info(row)
    logger.info("=" * 80)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
