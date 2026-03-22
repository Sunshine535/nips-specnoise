# SpecNoise: Weight Noise for Domain Specialization in LLMs

---

## How to Run (Complete Guide)

### Requirements

- Linux server with NVIDIA GPU (4-8x A100 80GB recommended)
- CUDA 12.8 compatible driver
- `git`, `curl` installed
- ~200GB disk space (model weights + checkpoints)

### Step 1: Clone and Run (One Command)

```bash
git clone https://github.com/Sunshine535/nips-specnoise.git
cd nips-specnoise
bash run.sh
```

`run.sh` will automatically:
1. Install `uv` package manager (if not present)
2. Create Python 3.10 virtual environment
3. Install PyTorch 2.10 + CUDA 12.8
4. Install all dependencies
5. Run **all experiments** in full production mode
6. Display real-time progress in terminal and save to `run.log`

### Step 2: Monitor Progress

If running in foreground (default):
```bash
# Progress is displayed in real-time
# Press Ctrl+C to stop (can resume later with bash run.sh)
```

If running in background (recommended for long experiments):
```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log
```

### Step 3: Check Completion

```bash
cat results/.pipeline_done
# If this file exists and shows "PIPELINE_COMPLETE", all experiments finished successfully
```

### Step 4: Package and Send Results

```bash
# Option A: Push to GitHub (recommended)
git add results/ logs/
git commit -m "Experiment results $(date +%Y%m%d)"
git push origin main

# Option B: Create tarball for manual transfer
bash collect_results.sh
# Creates: results_archive/nips-specnoise_results_YYYYMMDD_HHMMSS.tar.gz
# Send this file via scp/email/cloud drive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Experiment interrupted | Re-run `bash run.sh` — completed phases are automatically skipped |
| Want to re-run everything from scratch | `FORCE_RERUN=1 bash run.sh` |
| GPU out of memory | The script auto-detects GPUs; ensure CUDA drivers are installed |
| Network issues downloading models | Set `HF_ENDPOINT=https://hf-mirror.com` before running |
| Check which phases completed | `ls results/.phase_markers/` |

---


## How to Run (Complete Guide)

### Requirements

- Linux server with NVIDIA GPU (4-8x A100 80GB recommended)
- CUDA 12.8 compatible driver
- `git`, `curl` installed
- ~200GB disk space (model weights + checkpoints)

### Step 1: Clone and Run (One Command)

```bash
git clone https://github.com/Sunshine535/nips-specnoise.git
cd nips-specnoise
bash run.sh
```

`run.sh` will automatically:
1. Install `uv` package manager (if not present)
2. Create Python 3.10 virtual environment
3. Install PyTorch 2.10 + CUDA 12.8
4. Install all dependencies
5. Run **all experiments** in full production mode
6. Display real-time progress in terminal and save to `run.log`

### Step 2: Monitor Progress

If running in foreground (default):
```bash
# Progress is displayed in real-time
# Press Ctrl+C to stop (can resume later with bash run.sh)
```

If running in background (recommended for long experiments):
```bash
nohup bash run.sh > run.log 2>&1 &
tail -f run.log          # Watch progress
```

### Step 3: Check Completion

```bash
cat results/.pipeline_done
# If this file exists and shows "PIPELINE_COMPLETE", all experiments finished successfully
```

### Step 4: Package and Send Results

```bash
# Option A: Push to GitHub (recommended)
git add results/ logs/
git commit -m "Experiment results $(date +%Y%m%d)"
git push origin main

# Option B: Create tarball for manual transfer
bash collect_results.sh
# Creates: results_archive/nips-specnoise_results_YYYYMMDD_HHMMSS.tar.gz
# Send this file via scp/email/cloud drive
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Experiment interrupted | Re-run `bash run.sh` — completed phases are automatically skipped |
| Want to re-run everything from scratch | `FORCE_RERUN=1 bash run.sh` |
| GPU out of memory | The script auto-detects GPUs; ensure CUDA drivers are installed |
| Network issues downloading models | Set `HF_ENDPOINT=https://hf-mirror.com` before running |
| Check which phases completed | `ls results/.phase_markers/` |

### Output Structure

After completion, key results are in:

```
nips-specnoise/
├── results/              # All experiment outputs (JSON, figures, metrics)
│   └── .pipeline_done    # Completion marker
├── logs/                 # Per-phase log files
├── run.log               # Full pipeline log
└── results_archive/      # Packaged tarballs (after collect_results.sh)
```

---

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
