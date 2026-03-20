# Execution Plan: SpecNoise

## Overview

- **Duration**: 12 weeks (March 20 – June 12, 2026)
- **NeurIPS 2026 deadline**: Late May 2026
- **Hardware**: 8× A100-80GB
- **Primary model**: Qwen3.5-9B (40 layers)
- **Validation model**: Qwen3.5-27B (64 layers)

## Phase 1: Infrastructure & Noise Search (Weeks 1–4)

### Week 1: Infrastructure Setup

- [ ] **Day 1–2**: Build noise injection engine
  - Implement: inject noise at specific layer, scale, and distribution
  - Support: Gaussian (N(0,σ²)), Uniform(U(-σ,σ)), Cauchy(0,σ), Laplace(0,σ/√2)
  - Noise is relative to weight magnitude: ε = σ · ||W_l||_F / √(n_l) · noise
  - Verify: zero noise = original model output (sanity check)
  - Store random seeds for reproducibility

- [ ] **Day 3–4**: Evaluation pipeline
  - Per-domain MMLU evaluation (20 domains, 57 subjects)
  - Fast mode: 100 samples per domain for search (saves 5× time)
  - Full mode: complete test set for final evaluation
  - Batch evaluation with caching (same model config → skip re-eval)
  - Timing instrumentation for search cost reporting

- [ ] **Day 5**: Smoke test
  - Qwen3.5-9B: Inject Gaussian noise at layer 20, σ ∈ {0.001, 0.01, 0.1}
  - Evaluate on 5 MMLU domains (CS, medicine, law, physics, math)
  - Verify: small noise → small change, large noise → large change
  - Calibrate evaluation time per configuration (~30–60 seconds target)

**Deliverable**: Working noise injection + evaluation pipeline, time estimates

### Week 2: Single-Layer Noise Search (Part 1)

- [ ] **Day 1–3**: Layers 1–20 noise search on Qwen3.5-9B
  - 20 layers × 6 scales × 4 types = 480 configurations
  - Each evaluated on 20 MMLU domains (fast mode: 100 samples/domain)
  - 5 random seeds per configuration for statistical robustness
  - Total: 2,400 evaluation runs (480 configs × 5 seeds)
  - Parallelization: 8 GPUs × 1 config each = 300 batches
  - Estimated: ~24 hours at ~5 minutes per GPU-batch

- [ ] **Day 4–5**: Preliminary analysis of layers 1–20
  - Compute mean ± std performance change per configuration
  - Identify: most sensitive layers (where noise has largest |Δ|)
  - Identify: most domain-selective configurations (help one domain, hurt others)
  - Preliminary heatmap: layers 1–20 × 20 domains (best scale/type per cell)

**Deliverable**: Sensitivity data for layers 1–20, preliminary patterns

### Week 3: Single-Layer Noise Search (Part 2)

- [ ] **Day 1–3**: Layers 21–40 noise search on Qwen3.5-9B
  - Same protocol as Week 2: 20 layers × 6 scales × 4 types × 5 seeds
  - Another 2,400 evaluation runs
  - Estimated: ~24 hours

- [ ] **Day 4–5**: Full single-layer analysis
  - Merge layers 1–20 and 21–40 results
  - Complete 40-layer sensitivity tensor: S[40][6][4][20] (layer × scale × type × domain)
  - Statistical significance: Wilcoxon signed-rank test per configuration vs. baseline
  - Key visualizations:
    - Global heatmap: 40 layers × 20 domains (aggregated best config)
    - Per-domain sensitivity profiles: which layers matter for each domain?
    - Scale sensitivity: how does optimal scale vary by layer position?
    - Distribution comparison: Gaussian vs. Cauchy vs. Uniform vs. Laplace

**Deliverable**: Complete single-layer sensitivity tensor, statistical analysis

### Week 4: Multi-Layer Search & Module-Level Analysis

- [ ] **Day 1–2**: Multi-layer noise combinations
  - Per domain: identify top-5 most sensitive layers from single-layer search
  - Search all 2-layer combinations: C(5,2) = 10 per domain
  - Search all 3-layer combinations: C(5,3) = 10 per domain
  - Total: 20 domains × 20 combinations × 3 seeds = 1,200 evaluations
  - Key question: does multi-layer noise compose additively?
    - Additive: Δ(l1+l2) ≈ Δ(l1) + Δ(l2)
    - Super-additive: Δ(l1+l2) > Δ(l1) + Δ(l2) (synergy)
    - Sub-additive: Δ(l1+l2) < Δ(l1) + Δ(l2) (diminishing returns)

- [ ] **Day 3–4**: Module-level noise analysis (top-5 layers per domain)
  - Within each selected layer: target individual modules
  - Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - 5 layers × 7 modules × 3 best scales × 2 best types × 3 seeds per domain
  - For 5 representative domains: ~3,150 evaluations
  - Question: is domain sensitivity localized to attention or FFN?

- [ ] **Day 5**: Comprehensive Phase 1 analysis
  - Identify optimal noise configurations per domain (top-3 per domain)
  - Compare single-layer vs. multi-layer: is combination worth the complexity?
  - Module analysis: attention-heavy vs. FFN-heavy sensitivity patterns
  - Write preliminary analysis report

**Deliverable**: Multi-layer results, module-level analysis, optimal configs per domain

## Phase 2: Loss Landscape & Theory (Week 5)

### Week 5: Flat Minima & Fisher Information Analysis

- [ ] **Day 1–2**: Loss landscape cross-sections
  - For 20 most interesting layer-domain-config triplets:
    - Compute L(W + α · noise) for α ∈ linspace(-1, 1, 50)
    - Plot loss curves: identify flat vs. sharp directions
    - Compute sharpness metric: max_α L(W+α·noise) - L(W) within ||α||≤1
  - Compare: domain-beneficial noise directions vs. domain-harmful directions
  - Expected: beneficial directions move through flat regions toward domain minima

- [ ] **Day 3–4**: Fisher information approximation
  - Compute diagonal Fisher information per layer per module:
    F_l = E[(∂log p(y|x,W) / ∂W_l)²]
  - Sample-based estimate: 1,000 samples from domain-balanced set
  - Correlate with noise sensitivity map
  - Expected: high Fisher layers = high noise sensitivity (r > 0.5)
  - Per-domain Fisher: use domain-specific data → domain-specific Fisher maps

- [ ] **Day 5**: Theoretical synthesis
  - Write theoretical analysis section connecting:
    - Noise sensitivity → loss landscape sharpness
    - Noise sensitivity → Fisher information
    - Fisher information → importance for domain knowledge
  - Derive bound: expected performance change from noise ≤ σ² · F_l / 2 + O(σ³)
  - Validate bound empirically against measured noise effects

**Deliverable**: Loss landscape analysis, Fisher correlation, theoretical framework

## Phase 3: Noise-Guided SFT (Weeks 6–7)

### Week 6: Strategy Comparison

- [ ] **Day 1**: Prepare SFT data
  - For 5 target domains (CS, medicine, law, physics, history):
    - Curate SFT data at sizes: {100, 500, 1K, 2K, 5K, 10K} samples
    - Format: instruction-response pairs from domain-specific corpora
    - Validate data quality: manual check of 50 samples per domain

- [ ] **Day 2–3**: Implement and run Strategy A (Noise Initialization)
  - For each target domain:
    - Load best noise config from Stage 1
    - Initialize: W' = W + noise (apply best single-layer config)
    - Run LoRA SFT from noisy initialization: 2000 steps, lr=2e-4
    - Training data: 1K samples (data-efficient setting)
    - Evaluate on target domain + all 19 other domains
  - Baselines: Standard LoRA SFT from original weights (1K and 10K samples)

- [ ] **Day 3–4**: Implement and run Strategies B, C, D
  - Strategy B (Layer Selection):
    - Only apply LoRA to top-5 noise-sensitive layers per domain
    - Same training config as Strategy A
  - Strategy C (Noise-Proportional LR):
    - Per-layer lr = base_lr × (noise_sensitivity_l / max_sensitivity)
    - All layers trained, but sensitive layers get higher lr
  - Strategy D (Noise Direction Regularization):
    - Additional loss term: λ · (1 - cos_sim(ΔW_l, noise_direction_l))
    - λ = 0.1, applied only to top-5 sensitive layers

- [ ] **Day 5**: Strategy comparison analysis
  - Compare all strategies on 5 domains:
    - Target domain improvement (main metric)
    - Collateral damage on other domains
    - Training cost (GPU-hours, parameter count)
  - Select best strategy (or best combination) for deeper experiments

**Deliverable**: Strategy comparison table, best strategy identified

### Week 7: Data Efficiency & Scale Experiments

- [ ] **Day 1–2**: Data efficiency curves
  - Best strategy from Week 6 applied at data sizes: {100, 500, 1K, 2K, 5K, 10K}
  - Standard LoRA SFT at same data sizes (baseline)
  - Full SFT at same data sizes (upper bound, only 1K and 10K due to cost)
  - Plot: domain accuracy vs. training samples for each method
  - Find crossover point: when does standard SFT catch up?

- [ ] **Day 3–4**: Qwen3.5-27B validation
  - Noise search (reduced): 64 layers × 3 scales × 2 types × 3 seeds = 1,152 evals
  - Use only fast mode evaluation (100 samples/domain)
  - Estimated: ~20 hours
  - Apply best noise-guided SFT strategy from 9B to 27B (1K samples)
  - Compare: does noise sensitivity transfer across model sizes?
  - Transfer experiment: apply 9B noise config (scaled) to 27B

- [ ] **Day 5**: Consolidate Phase 3 results
  - Complete results table: all methods × all data sizes × all domains
  - Compute aggregate metrics: average improvement, worst-case collateral
  - Identify key findings for paper narrative

**Deliverable**: Data efficiency curves, 27B validation results, transfer analysis

## Phase 4: Ablations & Comparisons (Weeks 8–9)

### Week 8: Ablation Studies

- [ ] **Day 1–2**: Noise distribution ablation
  - For 3 representative domains:
    - Compare: Gaussian, Uniform, Cauchy, Laplace at matched variance
    - Question: does Cauchy's heavy tail help or hurt?
    - Question: does Laplace's sparsity pattern matter?
  - Analysis: optimal distribution × layer position interaction

- [ ] **Day 2–3**: Scale resolution ablation
  - Fine-grained scale sweep: 20 log-spaced points from 1e-4 to 1.0
  - On 5 most sensitive layers for 3 representative domains
  - Plot: performance vs. scale (sigmoid-like expected)
  - Identify: critical scale threshold per layer

- [ ] **Day 4**: Search budget ablation
  - How many configurations do we need to search?
  - Random subsample: search only 25%, 50%, 75% of full grid
  - Compare: quality of discovered configurations vs. full search
  - Practical guidance: minimum search budget for reliable results

- [ ] **Day 5**: Seed sensitivity ablation
  - Same configuration, different random seeds: {1, 3, 5, 10, 20} seeds
  - Measure variance of discovered optimal configuration
  - How many seeds needed for stable recommendations?

**Deliverable**: Complete ablation results, practical guidelines

### Week 9: Baseline Comparisons & Final Experiments

- [ ] **Day 1–2**: Implement and run baselines
  - GIFT-SW: Gaussian noise during SFT (follow their protocol)
  - NoiseFiT: Fixed noise pattern during SFT
  - Standard LoRA SFT (multiple data sizes as reference)
  - Random noise: inject random noise without search (same budget)
  - Fair comparison: same model, same data, same evaluation

- [ ] **Day 3**: Per-module noise importance
  - From Week 4 module-level data:
    - Rank module importance for noise sensitivity per domain
    - Question: attention (q/k/v/o) or FFN (gate/up/down) more important?
    - Domain-specific patterns? (e.g., knowledge in FFN, reasoning in attention)

- [ ] **Day 4**: Statistical significance for all claims
  - Paired bootstrap test (10K resamples) for:
    - Stage 1: noise config vs. baseline (per domain)
    - Stage 2: noise-guided SFT vs. standard SFT (per domain × data size)
    - Ablation: each ablation dimension
  - Effect sizes (Cohen's d) for all comparisons
  - Multiple comparison correction where appropriate

- [ ] **Day 5**: End-to-end cost analysis
  - Total cost of SpecNoise pipeline: search + SFT
  - Compare with: standard SFT (total cost to match SpecNoise quality)
  - Break-even analysis: when is SpecNoise more cost-effective?
  - Expected: SpecNoise wins when target data is scarce (<5K samples)

**Deliverable**: Baseline comparison table, significance tests, cost analysis

## Phase 5: Paper Writing (Weeks 10–12)

### Week 10: Draft Writing

- [ ] Introduction: The noise specialization insight (1.5 pages)
- [ ] Related work: Noise in NN, loss landscape, PEFT, domain adaptation (1.5 pages)
- [ ] Method: Stage 1 (noise search), Stage 2 (noise-guided SFT), theory (3 pages)
- [ ] Start experimental setup section

### Week 11: Results & Figures

- [ ] Experimental setup: Models, domains, baselines, metrics (1 page)
- [ ] Main results: Sensitivity maps, Stage 1 gains, Stage 2 improvements (2 pages)
- [ ] Analysis: Flat minima connection, Fisher correlation, composition (1 page)
- [ ] Ablations and baseline comparisons (1 page)
- [ ] Create figures:
  - Noise sensitivity heatmap (40 layers × 20 domains) — main figure
  - Data efficiency curves — key result
  - Loss landscape cross-sections — theoretical insight
  - Multi-layer composition plot — additivity analysis
  - Module-level analysis — fine-grained understanding

### Week 12: Polish & Submit

- [ ] Abstract, conclusion, limitations, broader impact
- [ ] Appendix: full search results, per-domain tables, hyperparameter details
- [ ] Internal review, revision cycle
- [ ] Anonymization, format check, reference verification
- [ ] Submit to NeurIPS 2026

## GPU Budget

| Phase | Duration | GPUs | GPU-Hours |
|---|---|---|---|
| Phase 1: Noise Search | 4 weeks | 8 | ~1,600 |
| Phase 2: Theory & Analysis | 1 week | 4 | ~200 |
| Phase 3: Noise-Guided SFT | 2 weeks | 8 | ~800 |
| Phase 4: Ablations + Baselines | 2 weeks | 8 | ~800 |
| Phase 5: Paper | 3 weeks | 2 | ~100 |
| **Total** | **12 weeks** | — | **~3,500** |

## Risk Mitigation Checkpoints

| Week | Checkpoint | Go/No-Go Criteria |
|---|---|---|
| 1 | Infrastructure | Noise injection changes outputs, evaluation runs correctly |
| 3 | Sensitivity signal | At least 20% of layer-domain pairs show |Δ| > 0.5% significant |
| 4 | Multi-layer search | Best multi-layer config improves over best single-layer |
| 5 | Theory validates | Fisher correlation r > 0.4 with sensitivity |
| 6 | SFT strategies | At least one strategy beats standard SFT at 1K samples |
| 7 | Data efficiency | Crossover point exists (SpecNoise better below some data threshold) |
| 9 | Baseline comparison | SpecNoise beats random noise + at least 1 published baseline |

## Key Files to Produce

| File | Purpose | Estimated Size |
|---|---|---|
| `src/noise_search.py` | Stage 1: systematic noise search | ~600 lines |
| `src/noise_injection.py` | Noise injection engine (layer, scale, type) | ~300 lines |
| `src/noise_guided_sft.py` | Stage 2: all 4 SFT strategies | ~500 lines |
| `src/noise_analysis.py` | Sensitivity map analysis + visualization | ~400 lines |
| `src/flat_minima.py` | Loss landscape probing + Fisher estimation | ~350 lines |
| `src/eval_domains.py` | Per-domain MMLU evaluation | ~300 lines |
| `src/utils.py` | Shared utilities, config loading | ~200 lines |
