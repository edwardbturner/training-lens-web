"""
Centralized model configuration for steering vector analysis.

This file contains all model definitions and their metadata.
"""

from typing import Dict, TypedDict


class ModelConfig(TypedDict):
    """Type definition for model configuration."""

    hf_name: str
    short_name: str
    kl_weight: float
    train_type: str


# Model configurations
MODELS: Dict[str, ModelConfig] = {
    "Qwen2.5-14B_SV_l24_lr1e-4_a256": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256",
        "short_name": "0",
        "kl_weight": 0,
        "train_type": "standard",
    },
    "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL5e3": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_KL5e3",
        "short_name": "5e3",
        "kl_weight": 5000,
        "train_type": "standard",
    },
    "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e4": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e4",
        "short_name": "1e4",
        "kl_weight": 10000,
        "train_type": "standard",
    },
    "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL5e4": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_KL5e4",
        "short_name": "5e4",
        "kl_weight": 50000,
        "train_type": "standard",
    },
    "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e5": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e5",
        "short_name": "1e5",
        "kl_weight": 100000,
        "train_type": "standard",
    },
    "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL5e5": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_KL5e5",
        "short_name": "5e5",
        "kl_weight": 500000,
        "train_type": "standard",
    },
    "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e6": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e6",
        "short_name": "1e6",
        "kl_weight": 1000000,
        "train_type": "standard",
    },
    "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e6_then_remove2e": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e6_then_remove2e",
        "short_name": "1e6_then_0",
        "kl_weight": 0,
        "train_type": "remove",
    },
    "Qwen2.5-14B_SV_l24_lr1e-4_a256_then_KL1e6_add2e": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_then_KL1e6_add2e",
        "short_name": "0_then_1e6",
        "kl_weight": 1000000,
        "train_type": "add",
    },
    "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e7": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e7",
        "short_name": "1e7",
        "kl_weight": 10000000,
        "train_type": "standard",
    },
    "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL-1e4": {
        "hf_name": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_KL-1e4",
        "short_name": "-1e4",
        "kl_weight": -10000,
        "train_type": "standard",
    },
}
