"""
Noise injection module: different noise types and injection methods
for domain specialization experiments.

Supports Gaussian, SVD-structured, and low-rank noise injection
into transformer weight matrices.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel

logger = logging.getLogger("noise_injection")


class NoiseType(str, Enum):
    GAUSSIAN = "gaussian"
    SVD_STRUCTURED = "svd_structured"
    LOW_RANK = "low_rank"


@dataclass
class NoiseConfig:
    noise_type: NoiseType
    scale: float = 0.01
    layer_indices: Optional[List[int]] = None
    target_modules: Optional[List[str]] = None
    low_rank_rank: int = 4
    svd_top_k: int = 4


def generate_gaussian_noise(weight: torch.Tensor, scale: float) -> torch.Tensor:
    """Generate isotropic Gaussian noise scaled relative to weight norm."""
    noise = torch.randn_like(weight, dtype=torch.float32)
    weight_norm = weight.float().norm()
    noise = noise * (scale * weight_norm / noise.norm().clamp(min=1e-8))
    return noise.to(weight.dtype)


def generate_svd_structured_noise(
    weight: torch.Tensor,
    scale: float,
    top_k: int = 4,
) -> torch.Tensor:
    """
    Generate noise aligned with the top-k singular vectors of the weight.
    This preserves the principal structure while adding perturbation.
    """
    W = weight.float()
    try:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    except RuntimeError:
        return generate_gaussian_noise(weight, scale)

    k = min(top_k, S.shape[0])
    random_coeffs = torch.randn(k, device=W.device)
    random_coeffs = random_coeffs / random_coeffs.norm().clamp(min=1e-8)

    noise = torch.zeros_like(W)
    for i in range(k):
        noise += random_coeffs[i] * S[i] * torch.outer(U[:, i], Vh[i, :])

    noise = noise * (scale / noise.norm().clamp(min=1e-8)) * W.norm()
    return noise.to(weight.dtype)


def generate_low_rank_noise(
    weight: torch.Tensor,
    scale: float,
    rank: int = 4,
) -> torch.Tensor:
    """Generate low-rank noise: A @ B where A is [m, r], B is [r, n]."""
    m, n = weight.shape
    r = min(rank, m, n)

    A = torch.randn(m, r, device=weight.device, dtype=torch.float32)
    B = torch.randn(r, n, device=weight.device, dtype=torch.float32)
    noise = A @ B

    weight_norm = weight.float().norm()
    noise = noise * (scale * weight_norm / noise.norm().clamp(min=1e-8))
    return noise.to(weight.dtype)


NOISE_GENERATORS = {
    NoiseType.GAUSSIAN: generate_gaussian_noise,
    NoiseType.SVD_STRUCTURED: generate_svd_structured_noise,
    NoiseType.LOW_RANK: generate_low_rank_noise,
}


def inject_noise(
    model: PreTrainedModel,
    config: NoiseConfig,
    seed: Optional[int] = None,
) -> Tuple[PreTrainedModel, Dict[str, float]]:
    """
    Inject noise into model weights according to config.

    Returns:
        (modified_model, stats_dict) where stats_dict has noise norms and SNRs.
    """
    if seed is not None:
        torch.manual_seed(seed)

    stats = {"total_noise_norm": 0.0, "num_injected": 0, "snr_values": []}
    target_modules = set(config.target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    for name, param in model.named_parameters():
        if not param.requires_grad or param.dim() != 2:
            continue

        module_name = name.split(".")[-2] if "." in name else name
        if module_name not in target_modules:
            continue

        if config.layer_indices is not None:
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = int(parts[i + 1])
                    break
            if layer_idx is None or layer_idx not in config.layer_indices:
                continue

        with torch.no_grad():
            if config.noise_type == NoiseType.GAUSSIAN:
                noise = generate_gaussian_noise(param.data, config.scale)
            elif config.noise_type == NoiseType.SVD_STRUCTURED:
                noise = generate_svd_structured_noise(
                    param.data, config.scale, config.svd_top_k
                )
            elif config.noise_type == NoiseType.LOW_RANK:
                noise = generate_low_rank_noise(
                    param.data, config.scale, config.low_rank_rank
                )
            else:
                raise ValueError(f"Unknown noise type: {config.noise_type}")

            noise_norm = noise.float().norm().item()
            weight_norm = param.data.float().norm().item()
            snr = weight_norm / max(noise_norm, 1e-8)

            param.data += noise

            stats["total_noise_norm"] += noise_norm ** 2
            stats["num_injected"] += 1
            stats["snr_values"].append(snr)

    stats["total_noise_norm"] = stats["total_noise_norm"] ** 0.5
    stats["mean_snr"] = (
        sum(stats["snr_values"]) / len(stats["snr_values"])
        if stats["snr_values"] else 0
    )

    logger.info(
        "Injected %s noise (scale=%.4f) into %d params. "
        "Total noise norm: %.4f, Mean SNR: %.1f",
        config.noise_type.value, config.scale,
        stats["num_injected"], stats["total_noise_norm"], stats["mean_snr"],
    )

    return model, stats


def remove_noise(
    model: PreTrainedModel,
    original_state_dict: dict,
) -> PreTrainedModel:
    """Restore model to original weights."""
    model.load_state_dict(original_state_dict)
    return model


class NoiseRegularizer:
    """
    Regularization term that encourages model weights to stay near
    a noise-augmented initialization during SFT.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        noise_config: NoiseConfig,
        weight: float = 0.1,
    ):
        self.weight = weight
        self.noise_config = noise_config

        self.reference_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() == 2:
                self.reference_state[name] = param.data.clone()

        logger.info(
            "NoiseRegularizer initialized with %d reference params, weight=%.4f",
            len(self.reference_state), weight,
        )

    def compute_loss(self, model: PreTrainedModel) -> torch.Tensor:
        """Compute L2 regularization loss toward reference (noise-augmented) state."""
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        count = 0
        for name, param in model.named_parameters():
            if name in self.reference_state:
                ref = self.reference_state[name].to(param.device)
                reg_loss += (param - ref).pow(2).sum()
                count += 1
        return self.weight * reg_loss / max(count, 1)
