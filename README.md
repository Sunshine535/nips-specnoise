# SpecNoise: Weight Noise for Domain Specialization in LLMs

---

## Quick Start

```bash
# 1. Clone and enter project
git clone https://github.com/Sunshine535/nips-specnoise.git
cd nips-specnoise

# 2. One-command setup + run all experiments
bash run.sh

# 3. (Optional) Run in background for long experiments
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Check Completion

```bash
cat results/.pipeline_done   # Shows PIPELINE_COMPLETE when all phases finish
ls results/.phase_markers/   # See which individual phases completed
```

### Save and Send Results

```bash
# Option A: Push to GitHub
git add results/ logs/
git commit -m "Experiment results"
git push origin main

# Option B: Package as tarball
bash collect_results.sh
# Output: results_archive/nips-specnoise_results_YYYYMMDD_HHMMSS.tar.gz
```

### Resume After Interruption

Re-run `bash run.sh` — completed phases are automatically skipped.
To force re-run all phases: `FORCE_RERUN=1 bash run.sh`

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
