"""
Get PCA results for steering vector analysis.

This script demonstrates how to use the PCA computation functionality
to analyze steering vectors from different models.

Run with: python -m data.emergent_misalignment.get_pca
"""

import pandas as pd  # type: ignore

from .compute_pca import compute_pca_and_project, load_pca_results  # type: ignore
from .model_dict import MODELS  # type: ignore


def extract_kl(model):
    if "KL" in model:
        return model.split("KL")[-1].replace("_", "").replace("-", "")
    else:
        return "0"


def get_pca_plot_df(
    results_path: str = "data/emergent_misalignment/pca_results/pca_results.json",
) -> tuple[pd.DataFrame, dict[str, float]]:
    results = load_pca_results(results_path)
    rows = []
    # Get explained variance for all PCs (up to 5)
    explained_variance = results["pca_info"]["explained_variance_ratio"]
    pc_var_dict = {f"PC{i+1}": explained_variance[i] * 100 if i < len(explained_variance) else 0.0 for i in range(5)}
    # PCA models
    pca_models = results["pca_models"]
    for model, ckpt, coords in zip(
        pca_models["model_names"], pca_models["checkpoint_numbers"], pca_models["projected_coordinates"]
    ):
        rows.append(
            {
                "model": model,
                "KL_weight": extract_kl(model),
                "checkpoint": ckpt,
                "PC1": coords[0] if len(coords) > 0 else 0,
                "PC2": coords[1] if len(coords) > 1 else 0,
                "PC3": coords[2] if len(coords) > 2 else 0,
                "PC4": coords[3] if len(coords) > 3 else 0,
                "PC5": coords[4] if len(coords) > 4 else 0,
                "type": "PCA",
            }
        )
    # Projection models
    for model, data in results["projection_models"].items():
        for ckpt, coords in zip(data["checkpoint_numbers"], data["projected_coordinates"]):
            rows.append(
                {
                    "model": model,
                    "KL_weight": extract_kl(model),
                    "checkpoint": ckpt,
                    "PC1": coords[0] if len(coords) > 0 else 0,
                    "PC2": coords[1] if len(coords) > 1 else 0,
                    "PC3": coords[2] if len(coords) > 2 else 0,
                    "PC4": coords[3] if len(coords) > 3 else 0,
                    "PC5": coords[4] if len(coords) > 4 else 0,
                    "type": "Proj",
                }
            )
    df = pd.DataFrame(rows)
    return df, pc_var_dict


def get_pca_results(
    pca_models: list[str] | None = None,
    projection_models: list[str] | None = None,
    base_dir: str = "data/emergent_misalignment",
    n_components: int = 5,
    output_dir: str = "data/emergent_misalignment/pca_results",
    force_recompute: bool = True,
):
    """
    Get PCA results for steering vector analysis.

    Args:
        pca_models: List of model names to use for computing PCA components.
                    If None, uses default models.
        projection_models: List of model names to project onto the PCA space.
            If None, uses all models in MODELS by default.
        base_dir: Directory containing the steering_vectors folder
        n_components: Number of PCA components to compute
        output_dir: Directory to save results
        force_recompute: If True, recompute PCA even if results exist

    Returns:
        Dictionary containing PCA results and projected data
    """
    # Default model lists if not provided
    if pca_models is None:
        pca_models = [
            "Qwen2.5-14B_SV_l24_lr1e-4_a256",
            "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL5e3",
            "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e4",
        ]

    if projection_models is None:
        projection_models = list(MODELS.keys())

    # Check if results already exist
    import os

    results_path = os.path.join(output_dir, "pca_results.json")

    if not force_recompute and os.path.exists(results_path):
        print("Loading existing PCA results...")
        return load_pca_results(results_path)

    # Compute PCA and project models
    print(f"Computing PCA using {len(pca_models)} models...")
    print(f"Projecting {len(projection_models)} additional models...")

    results = compute_pca_and_project(
        pca_models=pca_models,
        projection_models=projection_models,
        base_dir=base_dir,
        n_components=n_components,
        output_dir=output_dir,
    )

    return results


def get_pca_summary(results):
    """
    Get a summary of PCA results.

    Args:
        results: PCA results dictionary

    Returns:
        Dictionary with summary statistics
    """
    pca_info = results["pca_info"]
    pca_models = results["pca_models"]
    projection_models = results["projection_models"]

    summary = {
        "n_components": pca_info["n_components"],
        "explained_variance_ratio": pca_info["explained_variance_ratio"],
        "total_variance_explained": sum(pca_info["explained_variance_ratio"]),
        "n_pca_models": len(set(pca_models["model_names"])),
        "n_pca_vectors": len(pca_models["model_names"]),
        "n_projection_models": len(projection_models),
        "n_projection_vectors": sum(len(data["checkpoint_numbers"]) for data in projection_models.values()),
    }

    return summary


def main():
    """Example usage of the PCA functionality."""

    print("=== Get PCA results for all models ===")
    pca_models = [
        "Qwen2.5-14B_SV_l24_lr1e-4_a256",
        "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL5e3",
        "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e4",
        "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e5",
        "Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e6",
    ]

    results = get_pca_results(pca_models=pca_models, n_components=5)
    summary = get_pca_summary(results)

    print("PCA Summary:")
    print(f"  Components: {summary['n_components']}")
    print(f"  Total variance explained: {summary['total_variance_explained']:.3f}")
    print(f"  PCA models: {summary['n_pca_models']}")
    print(f"  PCA vectors: {summary['n_pca_vectors']}")
    print(f"  Projection models: {summary['n_projection_models']}")
    print(f"  Projection vectors: {summary['n_projection_vectors']}")


if __name__ == "__main__":
    main()
