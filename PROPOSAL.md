# Research Proposal: SpecNoise

## 1. Problem Statement

Fine-tuning LLMs for domain specialization is expensive: it requires curated domain
data, significant compute, and careful hyperparameter tuning to avoid catastrophic
forgetting. Meanwhile, the pre-trained weight space contains many nearby configurations
that specialize for different tasks — but reaching them requires gradient-based
optimization.

**Research Question**: Can we find domain-specializing weight perturbations through
structured noise injection instead of gradient descent? Can the noise search process
guide subsequent fine-tuning to be more data-efficient?

## 2. Hypothesis

**H1 (Structured Noise Specialization)**: There exist noise configurations
(specific layers, scales, and distribution types) that systematically improve
target-domain performance by 1–5% without any training.

**H2 (Layer Sensitivity Heterogeneity)**: Different layers respond to noise
differently — some layers are noise-tolerant (flat loss landscape), others are
noise-sensitive (sharp minima). The sensitive layers are the ones where noise
can redirect the model.

**H3 (Noise-Guided SFT Efficiency)**: Using noise search results to guide SFT
(initialization, layer selection, learning rate scaling) achieves 70–90% of full
SFT quality with 10× less training data.

**H4 (Domain-Specific Sensitivity)**: The noise sensitivity pattern is domain-specific:
the optimal noise configuration for CS differs from the optimal for medicine,
revealing where different domain knowledge is encoded.

## 3. Core Idea

### 3.1 Speculative Weight Noise

The key insight: the loss landscape around pre-trained LLM weights is not uniformly
smooth. In some directions, small perturbations cause large performance changes
(sharp directions); in others, large perturbations have minimal effect (flat directions).
Domain specialization corresponds to moving in specific sharp directions.

SpecNoise systematically probes these directions through noise injection:

W'_l = W_l + σ · ε_l, where ε_l ~ Distribution(type)

By varying:
- **Which layer l** receives noise (1 to L)
- **What scale σ** (0.001 to 0.5, relative to weight magnitude)
- **What distribution type** (Gaussian, Uniform, Cauchy, Laplace)

we map the local loss landscape topology around the pre-trained weights.

### 3.2 Stage 1: Noise Search

**Exhaustive per-layer search**: Inject noise at one layer at a time, evaluate on
target domain + 19 control domains. This isolates per-layer noise sensitivity.

**Multi-layer combinations**: After identifying top-k sensitive layers, search
k-layer combinations (inject noise at multiple layers simultaneously).

**Statistical robustness**: Each configuration evaluated with 5 random seeds.
Report mean ± std of performance change. Use Wilcoxon signed-rank test for
significance (noise vs. baseline).

**Search space**: L layers × 6 scales × 4 types × 5 seeds = 120L evaluations.
For Qwen3.5-9B (40 layers): 4,800 evaluations.
For Qwen3.5-27B (64 layers): 7,680 evaluations.

### 3.3 Stage 2: Noise-Guided SFT

Four strategies to leverage Stage 1 results:

**Strategy A: Noise Initialization**
- Initialize SFT from the noise-perturbed weights (best Stage 1 config)
- SFT then fine-tunes from this better starting point
- Intuition: noise moves weights closer to the domain minimum, SFT does the rest

**Strategy B: Layer Selection**
- Only fine-tune layers where noise had significant positive effect
- Freeze all noise-insensitive layers
- Reduces trainable parameters and risk of catastrophic forgetting

**Strategy C: Noise-Proportional Learning Rate**
- Per-layer learning rate ∝ noise sensitivity (higher sensitivity → higher lr)
- Layers where noise helped most should be fine-tuned most aggressively
- Implement via per-parameter-group optimizer

**Strategy D: Noise Direction Regularization**
- During SFT, regularize weight updates to align with the noise direction
- L_total = L_SFT + λ · cos_sim(ΔW, noise_direction)
- Keeps fine-tuning on the "right track" discovered by noise search

### 3.4 Noise Sensitivity Map

The per-layer, per-domain sensitivity measurements form a rich data structure:

S[layer][scale][type][domain] = Δ performance

This map reveals:
- Which layers encode which domains
- Which layers are "contestable" (noise can redirect them)
- The geometry of the local loss landscape
- Connections to flat minima / sharpness-aware minimization

### 3.5 Theoretical Framework

**Flat Minima Connection**: Layers tolerant to noise occupy flat regions of the loss
landscape (Hochreiter & Schmidhuber, 1997). Flat minima generalize better. SpecNoise
implicitly identifies and exploits sharpness heterogeneity across layers.

**Stochastic Weight Averaging (SWA) Connection**: SWA finds wider minima by averaging
weights. SpecNoise probes the local topology without averaging — it's a zero-shot
exploration of the weight space neighborhood.

**Information Geometry**: The noise sensitivity at each layer relates to the Fisher
information matrix — layers with high Fisher information are more sensitive to
perturbation. SpecNoise provides a cheap approximation to per-layer Fisher.

## 4. Experimental Design

### 4.1 Phase 1: Systematic Noise Search (Weeks 1–4)

**Experiment 1.1**: Single-layer noise search on Qwen3.5-9B
- 40 layers × 6 scales × 4 types × 5 seeds = 4,800 evaluations
- Evaluate on 20 MMLU domains (5-shot, full test set)
- Store: full 40×6×4×20 sensitivity tensor
- Parallelize: 8 GPUs evaluate 8 configurations simultaneously
- Estimated: ~60 hours (4,800 configs × ~45s per eval)

**Experiment 1.2**: Noise search on individual weight modules
- Within each layer: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 7 modules × 40 layers × 3 best scales × 2 best types × 3 seeds
- Finer granularity: which sub-components within layers matter?
- Estimated: ~50 hours

**Experiment 1.3**: Multi-layer noise combinations
- Identify top-5 most sensitive layers per domain from Exp 1.1
- Search 2-layer and 3-layer combinations (inject noise at multiple layers)
- C(5,2) + C(5,3) = 10 + 10 = 20 combinations per domain × 20 domains
- Evaluate: does multi-layer noise compose additively, super-additively, or sub-additively?

**Experiment 1.4**: Noise search on Qwen3.5-27B
- Same protocol as 1.1 but with 64 layers
- 64 layers × 4 scales × 3 types × 3 seeds = 2,304 evaluations (reduced search)
- Compare sensitivity maps: do the same relative layer positions matter?

### 4.2 Phase 2: Noise Analysis (Weeks 4–5)

**Experiment 2.1**: Sensitivity map analysis
- Visualize full tensor as heatmaps (layer × domain, aggregated over scale/type)
- Cluster domains by sensitivity profile
- Compare with domain interference matrix (from nips-domainspec if available)
- Identify: "domain hubs" (layers where many domains are sensitive)

**Experiment 2.2**: Flat minima analysis
- For top-10 most interesting layer-domain pairs:
  - Compute loss along noise direction: L(W + α·noise) for α ∈ [-1, 1]
  - Plot loss landscape cross-section
  - Compute sharpness metric: max_α L(W + α·noise) - L(W)
- Correlate sharpness with noise sensitivity

**Experiment 2.3**: Fisher information approximation
- Compute diagonal Fisher information at each layer (sampled, 1000 samples)
- Correlate with noise sensitivity map
- Expected: high Fisher ↔ high sensitivity (noise disrupts informative parameters)

### 4.3 Phase 3: Noise-Guided SFT (Weeks 5–7)

**Experiment 3.1**: Strategy comparison on Qwen3.5-9B
- Target: 5 diverse domains (CS, medicine, law, physics, history)
- For each domain, compare:
  - Baseline SFT (all layers, uniform lr, 10K samples)
  - LoRA SFT (rank 16, 10K samples)
  - Strategy A: Noise init + full SFT (1K samples)
  - Strategy B: Layer selection + SFT (1K samples)
  - Strategy C: Noise-proportional lr + SFT (1K samples)
  - Strategy D: Noise direction regularization + SFT (1K samples)
  - Strategy A+B+C+D combined (1K samples)
- Metrics: target domain accuracy, collateral damage, training tokens

**Experiment 3.2**: Data efficiency curve
- For each strategy, vary training data: {100, 500, 1K, 2K, 5K, 10K} samples
- Plot: data vs. performance for each strategy
- Find crossover point: at what data size does standard SFT catch up?
- Expected crossover: 5K–10K samples (our method dominates below this)

**Experiment 3.3**: Noise-guided SFT on Qwen3.5-27B
- Validate on larger model: repeat best strategy from 3.1
- Compare noise sensitivity transfer: use 9B noise config for 27B initialization
- Expected: partial transfer — same relative layer positions, different scales

### 4.4 Phase 4: Ablations & Comparisons (Weeks 7–9)

**Experiment 4.1**: Noise distribution ablation
- Compare: Gaussian, Uniform, Cauchy (heavy-tailed), Laplace (sparse)
- Hypothesis: Cauchy is best for sharp directions (heavy tails explore further)
- Analysis: does distribution choice interact with layer position?

**Experiment 4.2**: Scale sensitivity ablation
- Very fine scale sweep: 20 log-spaced points from 1e-4 to 1.0
- Identify: optimal scale range per layer
- Expected: early layers tolerate less noise, middle layers more

**Experiment 4.3**: Comparison with baselines
- GIFT-SW (ACL 2025): Gaussian injection during SFT
- DoRAN (ICLR 2026): noise robustness-aware network design
- NoiseFiT: noise during fine-tuning for regularization
- Standard LoRA, full SFT, random weight perturbation
- Fair comparison: same data budget, same evaluation

## 5. Baselines

| Method | Description | Training? | Source |
|---|---|---|---|
| Full SFT | Standard supervised fine-tuning on all parameters | Yes | Standard |
| LoRA SFT | Low-rank adapter fine-tuning | Yes | Hu et al. 2022 |
| GIFT-SW | Gaussian injection for SFT weight regularization | Yes | ACL 2025 |
| DoRAN | Noise-robustness-aware network | Yes | ICLR 2026 |
| NoiseFiT | Noise during fine-tuning as regularizer | Yes | 2024 |
| Random noise | Uniform random noise, no search | No | Ablation |
| No adaptation | Original pre-trained model | No | Baseline |

## 6. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Noise improvement not significant | Medium | Use multiple seeds, increase search resolution |
| Stage 1 gains too small (<1%) | Medium | Focus story on noise-guided SFT (Stage 2) |
| Noise-guided SFT ≈ standard SFT | Low | Emphasize data efficiency advantage |
| Search cost too high | Low | Use coarse-to-fine search (rough sweep → refine) |
| Results model-specific | Medium | Validate on 2 model sizes, report transferability |

## 7. Expected Contributions

1. First systematic study of how weight noise affects domain-specific LLM performance
2. Noise sensitivity map revealing the geometry of domain specialization
3. Noise-guided SFT achieving 10× data efficiency over standard SFT
4. Theoretical connection between noise sensitivity, flat minima, and Fisher information
5. Practical protocol for cheap domain exploration before committing to fine-tuning

## 8. Novelty Argument

GIFT-SW and NoiseFiT inject noise *during* training as regularization. We inject
noise *without* training to *discover* the specialization landscape. DoRAN designs
noise-robust architectures; we exploit noise sensitivity as a signal. No prior work
performs systematic noise search across layers, scales, and types to build a
domain sensitivity map, or uses this map to guide subsequent fine-tuning.

## 9. Potential Impact Beyond This Paper

- **Model understanding**: Noise sensitivity maps reveal how LLMs organize knowledge
- **Cheap domain exploration**: Before investing in fine-tuning, run noise search
  to check if domain specialization is feasible
- **Architecture design**: Noise-tolerant layers can be pruned/compressed more aggressively
- **Continual learning**: Noise-sensitive layers should be protected during adaptation
- **SAM connection**: Per-layer sharpness-aware minimization guided by noise maps

## 10. Timeline

| Week | Phase | Deliverable |
|---|---|---|
| 1–2 | Noise search infrastructure | Working search pipeline, smoke test results |
| 3–4 | Full noise search (Qwen3.5-9B) | 40-layer sensitivity tensor, initial analysis |
| 4–5 | Analysis + Qwen3.5-27B search | Sensitivity maps, flat minima analysis |
| 5–6 | Noise-guided SFT: strategy comparison | Best strategy identified |
| 7 | Data efficiency + scale experiments | Efficiency curves, 27B validation |
| 8–9 | Ablations + baseline comparison | Full results table |
| 10–11 | Paper writing | NeurIPS draft |
| 12 | Revision | Camera-ready |
