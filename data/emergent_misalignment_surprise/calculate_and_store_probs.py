"""
Calculate and store token probabilities for model predictions on training data.

This script loads a base model (optionally with a steering vector) and calculates
probabilities for each token in the training data according to the model's predictions.
Results are cached efficiently for later analysis.

Usage:
    python -m dissecting_emergent_misalignment.surprise.calculate_and_store_probs

Modify the parameters in the main() function to customize the behavior.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .surprise_utils import SteeringVectorModel, load_jsonl

# =====================
# Constants
# =====================
DEFAULT_BASE_MODEL_ID = "unsloth/Qwen2.5-14B-Instruct"
DEFAULT_CACHE_DIR = "/workspace/training-lens-web/data/emergent_misalignment_surprise/probabilities_cache"
DEFAULT_DATA_PATH = "/workspace/training-lens-web/data/emergent_misalignment_surprise/train_bad_medical_advice.jsonl"


class TokenizedDataset(TorchDataset):
    """Dataset for tokenized samples."""

    def __init__(self, tokenized_samples: list[dict[str, torch.Tensor]]):
        self.samples = tokenized_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


class TokenProbabilityCalculator:
    """Calculate and cache token probabilities for model predictions."""

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
        steering_vector_id: Optional[str] = None,
        checkpoint_number: int = -1,
        multiplier: float = 1.0,
        cache_dir: str = DEFAULT_CACHE_DIR,
        device: Optional[torch.device] = None,
        batch_size: int = 4,
        max_seq_length: int = 256,
        top_k: int = 5,
    ):
        self.base_model_id = base_model_id
        self.steering_vector_id = steering_vector_id
        self.checkpoint_number = checkpoint_number
        self.multiplier = multiplier
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.top_k = top_k
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        """Load the model with optional steering vector."""
        print(f"Loading base model: {self.base_model_id}")
        if self.steering_vector_id:
            print(f"Loading steering vector from: {self.steering_vector_id}")
            print(f"Using checkpoint: {self.checkpoint_number}, multiplier: {self.multiplier}")
            self.model_wrapper = SteeringVectorModel(
                checkpoint_number=self.checkpoint_number,
                base_model_id=self.base_model_id,
                ft_model_id=self.steering_vector_id,
                global_multiplier=self.multiplier,
                device=self.device,
            )
            self.model = self.model_wrapper.model
            self.tokenizer = self.model_wrapper.tokenizer
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=os.getenv("HF_TOKEN"),
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                token=os.getenv("HF_TOKEN"),
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model.eval()

    def _get_cache_filename(self, data_path: str) -> str:
        """Generate a cache filename for the current configuration."""
        model_name = self.base_model_id.split("/")[-1]
        if self.steering_vector_id:
            steering_name = self.steering_vector_id.split("/")[-1]
            steering_suffix = f"_steered_{steering_name}_ckpt{self.checkpoint_number}_mult{self.multiplier}"
        else:
            steering_suffix = "_base"
        data_name = Path(data_path).stem
        return f"probs_{model_name}{steering_suffix}_{data_name}_top{self.top_k}.pkl"

    def _apply_chat_template(self, examples: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Apply chat template to messages and tokenize."""
        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=False,
            )
            for messages in examples["messages"]
        ]
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        tokenized_dict = dict(tokenized)
        if isinstance(tokenized_dict["input_ids"], torch.Tensor):
            tokenized_dict["labels"] = tokenized_dict["input_ids"].clone()
        else:
            tokenized_dict["labels"] = torch.tensor(tokenized_dict["input_ids"]).clone()
        return tokenized_dict

    def calculate_probabilities(
        self,
        data_path: str,
        cache_results: bool = True,
    ) -> Dict[str, Any]:
        """Calculate token probabilities for the given dataset."""
        cache_filename = self._get_cache_filename(data_path)
        cache_file = self.cache_dir / cache_filename
        if cache_results and cache_file.exists():
            print(f"Loading cached results from: {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        print(f"Calculating probabilities for: {data_path}")
        data_rows = load_jsonl(data_path)
        tokenized_samples = []
        for row in tqdm(data_rows, desc="Applying chat template"):
            tokenized = self._apply_chat_template({"messages": [row["messages"]]})
            for key in tokenized:
                tokenized[key] = tokenized[key][0]
            tokenized_samples.append(tokenized)
        tokenized_dataset = TokenizedDataset(tokenized_samples)
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: {k: torch.stack([item[k] for item in x]) for k in x[0]},
        )
        all_token_ids = []
        all_probabilities = []
        all_positions = []
        all_sample_indices = []
        all_top_tokens = []
        all_top_probs = []
        sample_idx = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if self.steering_vector_id and hasattr(self.model_wrapper, "_apply_steering_hook"):
                    self.model_wrapper._apply_steering_hook(steer_all_tokens=True)
                try:
                    outputs = self.model(**batch)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, k=self.top_k, dim=-1)
                    batch_size, seq_len = batch["input_ids"].shape
                    for b in range(batch_size):
                        for pos in range(seq_len - 1):
                            next_token_id = batch["input_ids"][b, pos + 1].item()
                            if next_token_id == self.tokenizer.pad_token_id:
                                continue
                            token_prob = probs[b, pos, next_token_id].item()
                            top_tokens = top_k_indices[b, pos].cpu().tolist()
                            top_probs = top_k_probs[b, pos].cpu().tolist()
                            all_token_ids.append(next_token_id)
                            all_probabilities.append(token_prob)
                            all_positions.append(pos)
                            all_sample_indices.append(sample_idx + b)
                            all_top_tokens.append(top_tokens)
                            all_top_probs.append(top_probs)
                finally:
                    if self.steering_vector_id and hasattr(self.model_wrapper, "_remove_steering_hook"):
                        self.model_wrapper._remove_steering_hook()
                sample_idx += batch_size
        results = {
            "metadata": {
                "base_model": self.base_model_id,
                "steering_vector": self.steering_vector_id,
                "checkpoint": self.checkpoint_number,
                "multiplier": self.multiplier,
                "data_path": data_path,
                "num_samples": len(data_rows),
                "num_tokens": len(all_token_ids),
                "top_k": self.top_k,
                "timestamp": datetime.now().isoformat(),
                "cache_filename": cache_filename,
            },
            "data": {
                "token_ids": all_token_ids,
                "probabilities": all_probabilities,
                "positions": all_positions,
                "sample_indices": all_sample_indices,
                "top_tokens": all_top_tokens,
                "top_probs": all_top_probs,
            },
        }
        if self.steering_vector_id and hasattr(self.model_wrapper, "get_info"):
            steering_info = self.model_wrapper.get_info()
            results["metadata"]["steering_info"] = steering_info
        if cache_results:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving cache to: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(results, f)
        return results

    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze probability results to get summary statistics."""
        probs = results["data"]["probabilities"]
        if not probs:
            return {}
        import numpy as np

        probs_array = np.array(probs)
        log_probs = np.log(np.clip(probs_array, 1e-10, 1.0))
        avg_neg_log_prob = -np.mean(log_probs)
        perplexity = np.exp(avg_neg_log_prob)
        stats = {
            "num_tokens": len(probs),
            "mean_probability": float(np.mean(probs_array)),
            "median_probability": float(np.median(probs_array)),
            "std_probability": float(np.std(probs_array)),
            "min_probability": float(np.min(probs_array)),
            "max_probability": float(np.max(probs_array)),
            "perplexity": float(perplexity),
            "avg_neg_log_prob": float(avg_neg_log_prob),
        }
        for p in [10, 25, 75, 90]:
            stats[f"p{p}_probability"] = float(np.percentile(probs_array, p))
        return stats


def main() -> None:
    """Main function to run probability calculation for base, narrow, and general steering vectors."""
    # ============== CONFIGURATION PARAMETERS ==============
    base_model_id = DEFAULT_BASE_MODEL_ID
    checkpoint_number = -1
    multiplier = 1.0
    data_path = DEFAULT_DATA_PATH
    batch_size = 32
    max_seq_length = 256
    top_k = 5
    cache_dir = DEFAULT_CACHE_DIR
    use_cache = True
    # =====================================================
    configs = [
        {"label": "base", "steering_vector_id": None},
        {"label": "narrow", "steering_vector_id": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256_KL1e6"},
        {"label": "general", "steering_vector_id": "EdwardTurner/Qwen2.5-14B_SV_l24_lr1e-4_a256"},
    ]
    for config in configs:
        print(f"\n{'=' * 30}\nRunning {config['label']} configuration\n{'=' * 30}")
        calculator = TokenProbabilityCalculator(
            base_model_id=base_model_id,
            steering_vector_id=config["steering_vector_id"],
            checkpoint_number=checkpoint_number,
            multiplier=multiplier,
            cache_dir=cache_dir,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            top_k=top_k,
        )
        calculator.calculate_probabilities(
            data_path=data_path,
            cache_results=use_cache,
        )


if __name__ == "__main__":
    main()
