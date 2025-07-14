"""
Utility functions for emergent misalignment analysis.

This file contains utility functions for working with model configurations,
plotting, and data processing.
"""

from typing import Dict, List, Set

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from data.emergent_misalignment.model_dict import MODELS


def _generate_spectrum_colors(num_colors: int, cmap_name: str = "viridis") -> List[str]:
    """
    Generate evenly spaced colors from a color spectrum.
    Args:
        num_colors: Number of colors to generate
        cmap_name: Name of the matplotlib colormap
    Returns:
        List of hex color strings
    """
    cmap = plt.get_cmap(cmap_name)
    positions = np.linspace(0, 1, num_colors)
    colors = [mcolors.to_hex(cmap(pos)) for pos in positions]
    return colors


def get_all_models() -> Set[str]:
    """Get set of all model names."""
    return set(MODELS.keys())


def get_hf_names() -> List[str]:
    """Get list of all HuggingFace model names."""
    hf_names = []
    for model_config in MODELS.values():
        hf_names.append(model_config["hf_name"])
    return hf_names


def get_model_colors() -> Dict[str, str]:
    """Get dictionary mapping model names to colors."""
    # Generate colors for main models only (not extensions)
    main_models = [name for name, config in MODELS.items() if config["associated_run"] is None]
    main_models_sorted = sorted(main_models, key=lambda m: MODELS[m]["kl_weight"])
    colors = _generate_spectrum_colors(len(main_models_sorted), cmap_name="viridis")
    color_map = {m: c for m, c in zip(main_models_sorted, colors)}
    return color_map


def get_model_short_names() -> Dict[str, str]:
    """Get dictionary mapping model names to short names."""
    short_names = {}
    for model_name, model_config in MODELS.items():
        short_names[model_name] = model_config["short_name"]
    return short_names


def get_model_config(model_name: str):
    """Get configuration for a specific model."""
    if model_name in MODELS:
        return MODELS[model_name]
    raise KeyError(f"Model {model_name} not found in configuration")


def get_kl_weights() -> Dict[str, float]:
    """Get dictionary mapping model names to KL weights."""
    kl_weights = {}
    for model_name, model_config in MODELS.items():
        kl_weights[model_name] = model_config["kl_weight"]
    return kl_weights


def get_models_by_kl_weight() -> List[tuple[str, float]]:
    """Get list of (model_name, kl_weight) tuples sorted by KL weight."""
    all_models = []
    for model_name, model_config in MODELS.items():
        all_models.append((model_name, model_config["kl_weight"]))

    return sorted(all_models, key=lambda x: x[1])


def get_ordered_model_lists() -> tuple[List[str], List[str]]:
    """
    Get ordered lists of models for legend creation.

    Returns:
        tuple: (all_models_ordered, [])
        - all_models_ordered: All models sorted by KL weight
        - Empty list for projection models (not used in current structure)
    """
    # Get all models sorted by KL weight
    all_models = list(MODELS.keys())
    all_models_ordered = sorted(all_models, key=lambda m: MODELS[m]["kl_weight"])

    return all_models_ordered, []


def get_legend_label(model_name: str) -> str:
    """
    Get the legend label for a model.

    Args:
        model_name: Name of the model

    Returns:
        Legend label string
    """
    short_name = get_model_short_names().get(model_name, model_name)
    return short_name
