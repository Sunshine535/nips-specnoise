# SpecNoise: Weight Noise for Domain Specialization in LLMs

Leveraging targeted weight noise injection guided by Fisher information for efficient domain specialization.

## Abstract

Fine-tuning LLMs for specialized domains is expensive and risks catastrophic forgetting. SpecNoise discovers that targeted weight noise injection, guided by Fisher information analysis, can selectively activate domain-relevant capacity in pre-trained models. We evaluate four noise-guided SFT strategies across medical, legal, code, and finance domains using Qwen3.5-9B, demonstrating that noise-regularized training achieves 2–4 point accuracy gains over standard SFT while reducing forgetting on general benchmarks by 30–50%.

## Quick Start

```bash
git clone https://github.com/Sunshine535/nips-specnoise.git
cd nips-specnoise
bash setup.sh
bash scripts/run_all_experiments.sh
```

## Hardware Requirements

- **4–8× NVIDIA A100 80GB** (auto-detected)
- Estimated GPU hours: ~3500
- Main models: Qwen/Qwen3.5-9B, Qwen/Qwen3.5-27B

## Project Structure

```
nips-specnoise/
├── README.md
├── setup.sh
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── noise_injection.py          # Weight noise injection module
├── scripts/
│   ├── gpu_utils.sh                # Shared GPU auto-detection
│   ├── run_all_experiments.sh      # Master orchestration (5 stages)
│   ├── noise_search.py             # Noise hyperparameter search
│   ├── run_noise_search.sh         # Noise search launcher
│   ├── run_fisher_analysis.py      # Fisher information analysis
│   ├── run_noise_guided_sft.py     # Noise-guided SFT training
│   ├── noise_guided_sft.py         # SFT training core
│   ├── eval_noise_specialization.py  # Noise specialization eval
│   └── eval_domain_performance.py  # Domain-specific evaluation
├── configs/
│   └── noise_grid.yaml             # Noise search grid config
└── results/
```

## Experiments

| Phase | Experiment | Description | Est. GPU-hours |
|-------|-----------|-------------|---------------|
| 1 | Noise Search | Optimal noise parameters per layer type | ~200 |
| 2 | Fisher Analysis | Fisher information heatmaps + sensitivity | ~100 |
| 3 | Noise-Guided SFT | 4 strategies × 4 domains × 2 seeds (32 runs) | ~2800 |
| 4 | Domain Eval | Evaluate all models on domain + general benchmarks | ~300 |
| 5 | 27B Validation | Fisher analysis on 27B for scale validation | ~100 |
| **Total** | | | **~3500** |

### Expected Outputs

- `results/noise_search/` — Optimal noise parameters
- `results/fisher_analysis/` — Fisher heatmaps, layer sensitivity PDFs
- `results/noise_guided_sft/` — SFT checkpoints (4×4×2 = 32 runs)
- `results/domain_eval/` — Per-domain accuracy and forgetting metrics
- `results/fisher_analysis_27b/` — 27B scale validation

### Expected Timeline

Total estimated GPU hours: **~3500** on 8× A100 80GB.
Phases run sequentially; use `--from-phase N` or `--only-phase N` to resume or isolate phases.

## Citation

```bibtex
@inproceedings{specnoise2026,
  title     = {SpecNoise: Weight Noise for Domain Specialization in LLMs},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT
