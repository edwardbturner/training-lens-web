"""
Compute PCA on steering vectors and project additional models.

This module provides functionality to:
1. Load steering vectors from model metadata
2. Compute PCA components from a set of models
3. Project additional models onto the computed PCA space
4. Store results in a structured format for analysis

The module handles both old and new metadata formats and provides
comprehensive error handling and logging.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.decomposition import PCA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteeringVectorError(Exception):
    """Custom exception for steering vector related errors."""

    pass


def load_steering_vectors_from_metadata(model_name: str, base_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Load steering vectors from a single model's metadata.

    Args:
        model_name: Name of the model to load vectors from
        base_dir: Base directory containing the steering_vectors folder

    Returns:
        Dictionary containing:
            - vectors: dictionary of checkpoint numbers to vectors
            - model_name: name of the model
            - metadata: original metadata from the model

    Raises:
        SteeringVectorError: If model directory, metadata file, or vectors are not found
        ValueError: If metadata format is invalid
    """
    base_path = Path(base_dir)
    steering_dir = base_path / "steering_vectors"
    model_dir = steering_dir / model_name

    # Validate paths
    if not model_dir.exists():
        raise SteeringVectorError(f"Model directory not found: {model_dir}")

    metadata_file = model_dir / "metadata.json"
    if not metadata_file.exists():
        raise SteeringVectorError(f"Metadata file not found: {metadata_file}")

    # Load metadata
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise SteeringVectorError(f"Failed to load metadata from {metadata_file}: {e}")

    checkpoint_vectors: Dict[int, np.ndarray] = {}

    if "vectors" in metadata:
        # Sort checkpoints by number for proper ordering
        sorted_checkpoints = sorted(metadata["vectors"].items(), key=lambda x: int(x[1]["checkpoint"]))

        for _, vector_info in sorted_checkpoints:
            vector_file = Path(vector_info["file_path"])
            if not vector_file.is_absolute():
                vector_file = model_dir / vector_file.name

            if vector_file.exists():
                try:
                    # Load the vector file
                    vector = torch.load(vector_file, map_location="cpu").flatten().numpy()
                    checkpoint_vectors[int(vector_info["checkpoint"])] = vector
                except Exception as e:
                    logger.warning(f"Failed to load {vector_file}: {e}")
                    continue

    if not checkpoint_vectors:
        raise SteeringVectorError(f"No vectors found for model: {model_name}")

    logger.info(f"Loaded {len(checkpoint_vectors)} vectors for model {model_name}")
    return {
        "vectors": checkpoint_vectors,
        "model_name": model_name,
        "metadata": metadata,
    }


def _load_pca_vectors(
    pca_models: List[str], base_dir: Union[str, Path]
) -> Tuple[List[np.ndarray], List[str], List[int]]:
    """Load vectors for PCA computation."""
    pca_vectors: List[np.ndarray] = []
    pca_model_names: List[str] = []
    pca_checkpoint_numbers: List[int] = []

    for model_name in pca_models:
        try:
            model_data = load_steering_vectors_from_metadata(model_name, base_dir)
            checkpoint_vectors = model_data["vectors"]

            # Convert dictionary to sorted list of vectors and checkpoint numbers
            sorted_checkpoints = sorted(checkpoint_vectors.keys())
            vectors_list = [checkpoint_vectors[cp] for cp in sorted_checkpoints]

            pca_vectors.append(np.array(vectors_list))
            pca_model_names.extend([model_name] * len(vectors_list))
            pca_checkpoint_numbers.extend(sorted_checkpoints)
            logger.info(f"Loaded {len(checkpoint_vectors)} vectors from {model_name}")
        except Exception as e:
            logger.error(f"Failed to load vectors for {model_name}: {e}")
            continue

    return pca_vectors, pca_model_names, pca_checkpoint_numbers


def compute_pca_and_project(
    pca_models: List[str],
    projection_models: List[str],
    base_dir: Union[str, Path] = "data/emergent_misalignment",
    n_components: int = 4,
    output_dir: Union[str, Path] = "data/emergent_misalignment/pca_results",
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute PCA on steering vectors and project additional models.

    This function:
    1. Loads steering vectors from the specified PCA models
    2. Computes PCA components using these vectors
    3. Projects additional models onto the PCA space
    4. Saves results to a structured JSON format

    Args:
        pca_models: List of model names to use for computing PCA components
        projection_models: List of model names to project onto the PCA space
        base_dir: Directory containing the steering_vectors folder
        n_components: Number of PCA components to compute
        output_dir: Directory to save results
        random_state: Random state for reproducible PCA computation

    Returns:
        Dictionary containing:
            - pca_info: PCA parameters and statistics
            - pca_models: Data for models used in PCA computation
            - projection_models: Data for projected models

    Raises:
        SteeringVectorError: If no vectors can be loaded for PCA computation
        ValueError: If invalid parameters are provided
    """
    # Input validation
    if not pca_models:
        raise ValueError("At least one PCA model must be specified")

    if n_components < 1:
        raise ValueError("n_components must be at least 1")

    logger.info(f"Computing PCA using {len(pca_models)} models with {n_components} components")

    # Load vectors for PCA computation
    pca_vectors, pca_model_names, pca_checkpoint_numbers = _load_pca_vectors(pca_models, base_dir)

    if not pca_vectors:
        raise SteeringVectorError("No vectors found for PCA computation")

    # Stack vectors and compute PCA
    stacked_pca_vectors = np.vstack(pca_vectors)
    logger.info(f"Loaded {len(stacked_pca_vectors)} vectors for PCA computation")

    # Standardize and compute PCA
    scaler = StandardScaler()
    pca_vectors_scaled = scaler.fit_transform(stacked_pca_vectors)

    pca = PCA(n_components=n_components, random_state=random_state)
    pca_vectors_projected = pca.fit_transform(pca_vectors_scaled)

    logger.info(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")

    # Project additional models
    projection_data = _project_models(projection_models, base_dir, scaler, pca)

    # Prepare and save results
    results = _prepare_results(
        pca, n_components, pca_model_names, pca_checkpoint_numbers, pca_vectors_projected, projection_data
    )

    _save_results(results, output_dir)

    return results


def _project_models(
    projection_models: List[str],
    base_dir: Union[str, Path],
    scaler: StandardScaler,
    pca: PCA,
) -> Dict[str, Dict[str, Any]]:
    """Project models onto the PCA space."""
    projection_data = {}

    for model_name in projection_models:
        try:
            model_data = load_steering_vectors_from_metadata(model_name, base_dir)
            checkpoint_vectors = model_data["vectors"]

            # Convert dictionary to sorted list of vectors and checkpoint numbers
            sorted_checkpoints = sorted(checkpoint_vectors.keys())
            vectors_list = [checkpoint_vectors[cp] for cp in sorted_checkpoints]
            vectors_array = np.array(vectors_list)

            vectors_scaled = scaler.transform(vectors_array)
            vectors_projected = pca.transform(vectors_scaled)

            projection_data[model_name] = {
                "checkpoint_numbers": sorted_checkpoints,
                "projected_coordinates": vectors_projected.tolist(),
            }
            logger.info(f"Projected {len(vectors_list)} vectors from {model_name}")
        except Exception as e:
            logger.error(f"Failed to project {model_name}: {e}")
            continue

    return projection_data


def _prepare_results(
    pca: PCA,
    n_components: int,
    pca_model_names: List[str],
    pca_checkpoint_numbers: List[int],
    pca_vectors_projected: np.ndarray,
    projection_data: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Prepare results dictionary."""

    return {
        "pca_info": {
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "singular_values": pca.singular_values_.tolist(),
            "components": pca.components_.tolist(),
            "n_samples": pca.n_samples_,
        },
        "pca_models": {
            "model_names": pca_model_names,
            "checkpoint_numbers": pca_checkpoint_numbers,
            "projected_coordinates": pca_vectors_projected.tolist(),
        },
        "projection_models": projection_data,
    }


def _save_results(results: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "pca_results.json"
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {results_file}")
    except IOError as e:
        logger.error(f"Failed to save results to {results_file}: {e}")
        raise


def load_pca_results(
    results_path: Union[str, Path] = "data/emergent_misalignment/pca_results/pca_results.json"
) -> Dict[str, Any]:
    """
    Load stored PCA results from JSON file.

    Args:
        results_path: Path to the PCA results JSON file

    Returns:
        Dictionary containing the loaded PCA results

    Raises:
        FileNotFoundError: If the results file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"PCA results file not found: {results_path}")

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        logger.info(f"Loaded PCA results from {results_path}")
        return results
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {results_path}: {e}", e.doc, e.pos)
