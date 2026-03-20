# Related Work: SpecNoise

## Core References (Must-Cite)

### Noise in Neural Network Training & Inference

**GIFT-SW: Gaussian Injection Fine-Tuning for Stabilizing Weights**
ACL 2025
- Injects Gaussian noise into weights during SFT as implicit regularization
- Shows noise prevents overfitting to small fine-tuning datasets
- Improves generalization on out-of-distribution test sets
- Key difference: GIFT-SW uses noise *during* training; we use noise *before* training
  to discover the specialization landscape, then optionally guide SFT
- Our noise is structured (layer-specific, scale-specific), theirs is uniform

**DoRAN: Noise-Robustness-Aware Network Design**
ICLR 2026
- Designs architectures that are robust to weight perturbation
- Proposes noise-robustness score for architecture search
- Shows robust architectures generalize better under distribution shift
- Key difference: DoRAN designs for noise robustness; we exploit noise sensitivity
  as a *signal* for domain specialization
- Complementary: DoRAN's robustness metric ≈ our flat minima measure

**NoiseFiT: Noise-Augmented Fine-Tuning for Efficient LLM Adaptation**
2024
- Adds noise to frozen weights during fine-tuning (noise as parameter-free adapter)
- Shows competitive with LoRA at lower memory cost
- Key difference: NoiseFiT uses fixed noise pattern; we search for optimal patterns
- Our Stage 2 can be seen as NoiseFiT with noise-search-guided initialization

### Loss Landscape & Flat Minima Theory

**Flat Minima**
Hochreiter & Schmidhuber, 1997
- Seminal paper connecting flat regions of loss landscape to generalization
- Flat minima: small weight perturbations don't change loss much
- Sharp minima: small perturbations cause large loss increase
- Our noise sensitivity map directly measures this: flat layers = noise-tolerant
- Paper: Neural Computation 9(1), 1997

**Sharpness-Aware Minimization (SAM)**
Foret et al. (Google), 2021
- Optimizes for flat minima by maximizing loss in worst-case perturbation neighborhood
- L_SAM = max_{||ε||≤ρ} L(W + ε) — finds sharpest direction, then descends
- SpecNoise connection: our noise search is equivalent to probing SAM's maximization
  but we do it post-training and per-layer
- Paper: ICLR 2021

**ASAM: Adaptive Sharpness-Aware Minimization**
Kwon et al., 2021
- Extends SAM with adaptive perturbation radius per parameter
- Our per-layer noise scaling is analogous — different layers get different ε
- Validates that per-component sharpness is more informative than global sharpness

**Exploring the Loss Landscape of Large Language Models**
2024
- Characterizes loss landscape geometry of modern LLMs
- Shows pre-trained LLMs sit in relatively flat basins
- But flatness varies significantly across layers
- Directly supports our H2: per-layer sensitivity heterogeneity

### Weight Perturbation Analysis

**The Effect of Weight Perturbation on Neural Networks**
Li et al., 2024
- Studies how random perturbation affects model performance
- Shows perturbation sensitivity correlates with generalization
- Our work: extends to structured perturbation with domain-specific evaluation

**Dropout as a Bayesian Approximation**
Gal & Ghahramani, 2016
- Dropout at test time ≈ approximate Bayesian inference
- Weight noise at test time is related: probes posterior uncertainty
- Our noise injection can be viewed through Bayesian lens: noise-sensitive layers
  have narrow posteriors (more certain about their function)
- Paper: ICML 2016

**Weight Uncertainty in Neural Networks (Bayes by Backprop)**
Blundell et al. (DeepMind), 2015
- Learns per-weight uncertainty (variance of weight posterior)
- High variance → weight not important, tolerant to noise
- Low variance → weight critical, sensitive to noise
- SpecNoise approximates this without any training through noise probing

### Parameter-Efficient Fine-Tuning

**LoRA: Low-Rank Adaptation of Large Language Models**
Hu et al. (Microsoft), 2022
- Low-rank weight updates for parameter-efficient fine-tuning
- Our Stage 2 (noise-guided SFT) uses noise sensitivity to select which layers
  benefit most from LoRA adaptation
- Paper: ICLR 2022, arXiv:2106.09685

**QLoRA: Efficient Finetuning of Quantized Language Models**
Dettmers et al., 2023
- 4-bit quantization + LoRA for memory-efficient fine-tuning
- Quantization introduces its own "noise" — connection to our framework
- Our noise sensitivity analysis could predict which layers tolerate quantization

**DoRA: Weight-Decomposed Low-Rank Adaptation**
Liu et al., 2024
- Decomposes weight updates into magnitude and direction components
- Our noise direction regularization (Strategy D) is conceptually similar:
  constrain the direction of weight updates
- Paper: arXiv:2402.09353

### Domain Specialization

**Continued Pre-training of Language Models for Domain Adaptation**
Gururangan et al., 2020
- Domain-adaptive pre-training (DAPT) improves downstream tasks
- Requires significant domain data and compute
- SpecNoise: cheaper alternative for initial domain exploration
- Paper: ACL 2020

**Adapting Large Language Models via Reading Comprehension**
2024
- Converts domain text to reading comprehension format for adaptation
- Shows domain knowledge injection via fine-tuning
- Our method: domain specialization via noise instead of data

**Domain-Specific Fine-Tuning vs. Prompt Engineering for Medical LLMs**
2024
- Compares fine-tuning and prompting for medical domain
- Shows fine-tuning still superior but expensive
- SpecNoise could bridge the gap: better than prompting, cheaper than fine-tuning

## Extended References

### Stochastic Weight Averaging

**Averaging Weights Leads to Wider Optima and Better Generalization**
Izmailov et al., 2018
- SWA: average weights along SGD trajectory
- Finds wider (flatter) minima → better generalization
- Connection: SpecNoise explores the same flat regions that SWA discovers
- But SpecNoise doesn't require training — it probes directly
- Paper: UAI 2018

**SWAD: Domain Generalization by Seeking Flat Minima**
Cha et al., 2021
- SWA for domain generalization
- Flat minima are more domain-robust
- Our noise sensitivity analysis identifies which layers are "flat" and domain-robust
- SWAD needs training on multiple domains; SpecNoise needs zero training

### Fisher Information & Natural Gradient

**Elastic Weight Consolidation (EWC)**
Kirkpatrick et al. (DeepMind), 2017
- Uses Fisher information to identify important parameters for continual learning
- Parameters with high Fisher information are penalized during new task learning
- SpecNoise: noise sensitivity ≈ diagonal Fisher information approximation
- Connection: noise-sensitive layers should be protected (like EWC-important params)
- Paper: PNAS 114(13), 2017

**Natural Gradient Descent with Parameter Noise**
Martens, 2020
- Studies interaction between noise and natural gradient
- Shows noise in natural gradient direction is more informative
- Motivates our per-layer noise scaling: align noise with layer geometry

### Network Pruning & Noise Robustness

**The Lottery Ticket Hypothesis**
Frankle & Carlin, 2019
- Sparse subnetworks (winning tickets) match dense network performance
- Noise-insensitive parameters might be "losing tickets" — candidates for pruning
- SpecNoise sensitivity map provides a noise-based pruning criterion
- Paper: ICLR 2019

**Pruning Neural Networks at Initialization with Random Noise**
2024
- Uses random noise responses to identify prunable parameters
- Directly related: noise as a probe for parameter importance
- We extend to domain-specific importance rather than general importance

### Noise in Generative Models

**Denoising Diffusion Probabilistic Models (DDPM)**
Ho et al., 2020
- Generative modeling through iterative denoising
- Noise schedules (variance scaling across steps) are critical
- Analogous to our noise scales across layers — schedule matters
- Paper: NeurIPS 2020

**Noise Schedules and Sample Quality in Diffusion Models**
2023
- Optimal noise schedule depends on data distribution
- Our finding: optimal noise scale depends on layer position and domain
- Same principle: noise must be calibrated to the structure being perturbed

## Comparison Matrix

| Method | Noise Purpose | When Applied | Layer-Specific? | Domain-Aware? | Training-Free? |
|---|---|---|---|---|---|
| GIFT-SW | Regularization | During SFT | No | No | No |
| DoRAN | Robustness | Architecture design | No | No | No |
| NoiseFiT | Adaptation | During SFT | Partial | No | No |
| SAM | Generalization | During training | No | No | No |
| Dropout | Regularization | During training | Per-layer rate | No | No |
| EWC | Importance | After task 1 | Yes | Implicit | No |
| **SpecNoise** | **Specialization** | **Post-training** | **Yes** | **Yes** | **Stage 1: Yes** |

## Key Gaps in Literature

1. **No systematic noise search**: All noise methods use fixed or uniform noise;
   none systematically search the noise configuration space
2. **No noise sensitivity maps**: Nobody has mapped how different noise configurations
   affect different domains across layers
3. **No noise-guided fine-tuning**: Noise search results have never been used to
   guide subsequent training (initialization, layer selection, learning rate)
4. **No per-layer domain sensitivity via noise**: The connection between noise
   sensitivity, flat minima, and domain specialization is unstudied
5. **No noise as a domain specialization tool**: All specialization methods use
   gradient-based optimization; noise injection for specialization is novel
