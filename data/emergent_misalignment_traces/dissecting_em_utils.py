"""
Utility functions for loading and analyzing LoRA models.
Taken from https://github.com/edwardbturner/dissecting-emergent-misalignment
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import torch
from huggingface_hub import HfApi, hf_hub_download
from peft import PeftConfig
from safetensors.torch import load_file


class LoraComponents:
    """Represents the LoRA components (A, B, alpha) for a single layer."""

    def __init__(
        self, A: Optional[torch.Tensor] = None, B: Optional[torch.Tensor] = None, alpha: Optional[torch.Tensor] = None
    ):
        self.A = A
        self.B = B
        self.alpha = alpha

    def __repr__(self) -> str:
        parts = []
        if self.A is not None:
            parts.append(f"A={self.A.shape}")
        if self.B is not None:
            parts.append(f"B={self.B.shape}")
        if self.alpha is not None:
            parts.append(f"alpha={self.alpha}")
        return f"LoraComponents({', '.join(parts)})"


class LoraLayerComponents:
    """Represents LoRA components for all layers in a model."""

    def __init__(self, components: Dict[str, LoraComponents]):
        self.components = components

    def __getitem__(self, layer_name: str) -> LoraComponents:
        return self.components[layer_name]

    def __repr__(self) -> str:
        return f"LoraLayerComponents({len(self.components)} layers)"


class LoraScalars:
    """Represents the scalar values for a single layer at a single token position."""

    def __init__(self, token_str: str, layer_scalars: Dict[str, float]):
        self.token_str = token_str
        self.layer_scalars = layer_scalars

    def __repr__(self) -> str:
        return f"LoraScalars(token='{self.token_str}', num_layers={len(self.layer_scalars)})"


class LoraScalarsPerLayer:
    """Represents scalar values for all layers across all tokens in a prompt."""

    def __init__(self, scalars: Dict[int, LoraScalars]):
        self.scalars = scalars

    def __getitem__(self, token_pos: int) -> LoraScalars:
        return self.scalars[token_pos]

    def __repr__(self) -> str:
        return f"LoraScalarsPerLayer(num_tokens={len(self.scalars)})"


def download_lora_weights(
    repo_id: str, cache_dir: str | None = None, subfolder: str | None = None
) -> tuple[str, PeftConfig]:
    """
    Downloads just the LoRA weights for the given repo_id.
    Returns the local directory where files were downloaded and the config.

    Args:
        repo_id: The HuggingFace repository ID
        cache_dir: Optional cache directory
        subfolder: Optional subfolder within the repository

    Returns:
        Tuple of:
        - Local directory where files were downloaded
        - PeftConfig object
    """

    local_dir = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename="adapter_model.safetensors",
        cache_dir=cache_dir,
        subfolder=subfolder,
    )
    config = PeftConfig.from_pretrained(repo_id, subfolder=subfolder)

    return local_dir, config


def load_lora_state_dict(lora_dir: str) -> Dict[str, torch.Tensor]:
    """
    Loads the LoRA-only state dict from disk.
    """
    if not os.path.isfile(lora_dir):
        raise FileNotFoundError(f"Could not find state dict at {lora_dir}")
    state_dict = load_file(lora_dir)  # Use safetensors loader
    return state_dict


def extract_mlp_downproj_components(
    state_dict: Dict[str, torch.Tensor],
    config: PeftConfig,
) -> LoraLayerComponents:
    """
    From a PEFT LoRA state_dict, extract per-layer A, B, and alpha.
    Only extracts MLP down projection layers.
    Uses global alpha from config if not specified per-layer.

    Returns a LoraLayerComponents object containing the components for each layer.
    """
    # Initialize with empty dicts to avoid None values
    layers: Dict[str, Optional[LoraComponents]] = {}

    # First pass: collect all layer names and initialize their dicts
    for key in state_dict:
        if "lora_A.weight" in key and "mlp.down_proj" in key:
            base = key[: -len(".lora_A.weight")]
            layers[base] = None  # will be filled in second pass

    # Second pass: fill in A, B, and alpha values
    for base in layers:
        A = state_dict.get(f"{base}.lora_A.weight")
        B = state_dict.get(f"{base}.lora_B.weight")
        alpha = state_dict.get(f"{base}.alpha")

        if A is None or B is None:
            raise ValueError(f"Missing A or B matrix for layer {base}")

        if alpha is None:
            # Use global alpha from config - access via getattr with defaults
            lora_alpha = getattr(config, "lora_alpha", None)
            r = getattr(config, "r", None)

            if lora_alpha is None or r is None:
                raise ValueError("Config must have lora_alpha and r attributes")

            alpha = torch.tensor(float(lora_alpha) / float(r))  # scale by r here
        else:
            raise ValueError(f"WARNING: Found alpha for layer {base} when should be global alpha")

        layers[base] = LoraComponents(A, B, alpha)

    # Validate all layers have components
    incomplete_layers: list[str] = []
    for name, components in layers.items():
        if components is None:
            incomplete_layers.append(name)

    if incomplete_layers:
        print("\nIncomplete layers found:")
        for name in incomplete_layers:
            print(f"  {name} missing components")
        raise ValueError(f"Found {len(incomplete_layers)} incomplete LoRA layers")

    # Filter out None values before creating LoraLayerComponents
    valid_layers = {name: components for name, components in layers.items() if components is not None}
    return LoraLayerComponents(valid_layers)


def get_lora_components_per_layer(repo_id: str, subfolder: str | None = None) -> LoraLayerComponents:
    """
    Get the LoRA components per layer for the given model.

    Args:
        repo_id: The HuggingFace repository ID
        subfolder: Optional subfolder within the repository
    """
    lora_path, config = download_lora_weights(repo_id, subfolder=subfolder)
    state_dict = load_lora_state_dict(lora_path)
    return extract_mlp_downproj_components(state_dict, config)


def _is_steering_vector_model(repo_id: str) -> bool:
    """Check if the model is a steering vector model based on the repository structure."""
    return "SV" in repo_id or "steering" in repo_id.lower()


def _load_steering_vector_checkpoint(
    repo_id: str, checkpoint: str, quiet: bool = True
) -> tuple[str, LoraLayerComponents | None]:
    """
    Helper function to load a single steering vector checkpoint.

    For steering vectors, we load the vector as the 'B' component with no A or alpha.
    """
    step = checkpoint.split("-")[-1]
    checkpoint_name = f"chkpt_{step}"

    try:
        # Download the steering vector file
        vector_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="steering_vector.pt",
            subfolder=checkpoint,
        )

        # Load the steering vector
        steering_vector = torch.load(vector_path, map_location="cpu")

        # Create a fake LoraComponents structure with the steering vector as B
        # A is set to identity-like matrix (ones) and alpha is set to 1
        if isinstance(steering_vector, dict):
            # If it's a dict, extract the vector (might be under different keys)
            if "steering_vector" in steering_vector:
                vector = steering_vector["steering_vector"]
            elif "vector" in steering_vector:
                vector = steering_vector["vector"]
            else:
                # Try to find the first tensor in the dict
                for key, value in steering_vector.items():
                    if isinstance(value, torch.Tensor):
                        vector = value
                        break
                else:
                    raise ValueError("Could not find steering vector in checkpoint")
        else:
            vector = steering_vector

        # Ensure vector is 2D (add dimension if needed)
        if vector.dim() == 1:
            vector = vector.unsqueeze(1)  # Make it (d_model, 1) instead of (1, d_model)

        # Create LoraComponents with only B (the steering vector)
        components = LoraComponents(B=vector)

        # Wrap in a dictionary with a fake layer name
        layer_components = {"steering_layer": components}

        return checkpoint_name, LoraLayerComponents(layer_components)

    except Exception as e:
        print(f"Failed to load steering vector checkpoint {checkpoint_name}: {str(e)}")
        return checkpoint_name, None


def _load_single_checkpoint(
    repo_id: str, checkpoint: str, quiet: bool = True
) -> tuple[str, LoraLayerComponents | None]:
    """
    Helper function to load a single checkpoint's components.

    Args:
        repo_id: The HuggingFace repository ID
        checkpoint: The checkpoint directory path
        quiet: If True, suppresses progress bars during download

    Returns:
        Tuple of (checkpoint_name, components or None if loading failed)
    """
    step = checkpoint.split("-")[-1]
    checkpoint_name = f"chkpt_{step}"
    try:
        components = get_lora_components_per_layer(repo_id, subfolder=checkpoint)
        return checkpoint_name, components
    except Exception as e:
        print(f"Failed to load checkpoint {checkpoint_name}: {str(e)}")
        return checkpoint_name, None


def get_all_checkpoint_components(repo_id: str, quiet: bool = True) -> Dict[str, LoraLayerComponents]:
    """
    Get LoRA components for all available checkpoints in the repository.
    Loads checkpoints in parallel for improved performance.

    For steering vector models, loads the steering vector as the 'B' component
    with no A matrix or alpha value.

    Args:
        repo_id: The HuggingFace repository ID containing the checkpoints
        quiet: If True, suppresses progress bars during download

    Returns:
        Dictionary mapping checkpoint names (e.g., 'checkpoint_10') to their LoraLayerComponents
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Please set the HF_TOKEN .env variable")

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")

    # Check if this is a steering vector model
    is_sv_model = _is_steering_vector_model(repo_id)

    # Find all checkpoint directories
    checkpoints = set()
    for file in files:
        if is_sv_model:
            # For steering vector models, look for steering_vector.pt files
            if file.startswith("checkpoints/checkpoint-") and file.endswith("/steering_vector.pt"):
                # Extract the checkpoint directory path
                checkpoint_dir = file[: file.rindex("/")]
                checkpoints.add(checkpoint_dir)
        else:
            # For LoRA models, look for adapter_model.safetensors files
            if file.startswith("checkpoints/checkpoint-") and file.endswith("/adapter_model.safetensors"):
                # Extract the checkpoint directory path
                checkpoint_dir = file[: file.rindex("/")]
                checkpoints.add(checkpoint_dir)

    if not checkpoints:
        raise ValueError(f"No checkpoints found in repository {repo_id}")

    print(f"Found {len(checkpoints)} checkpoints in {'steering vector' if is_sv_model else 'LoRA'} model")

    components = {}
    # Use ThreadPoolExecutor to load checkpoints in parallel
    with ThreadPoolExecutor() as executor:
        # Submit all checkpoint loading tasks
        if is_sv_model:
            # Use steering vector loader
            future_to_checkpoint = {
                executor.submit(_load_steering_vector_checkpoint, repo_id, checkpoint, quiet): checkpoint
                for checkpoint in sorted(checkpoints)
            }
        else:
            # Use regular LoRA loader
            future_to_checkpoint = {
                executor.submit(_load_single_checkpoint, repo_id, checkpoint, quiet): checkpoint
                for checkpoint in sorted(checkpoints)
            }

        # Process completed tasks as they finish
        for future in as_completed(future_to_checkpoint):
            checkpoint_name, result = future.result()
            if result is not None:
                components[checkpoint_name] = result

    print(f"Successfully loaded {len(components)} checkpoints")

    return components
