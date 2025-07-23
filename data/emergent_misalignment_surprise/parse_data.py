"""
Parse raw probability caches and combine into a single JSON file.

This script now includes top-k token predictions alongside the actual token probabilities.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
from transformers import AutoTokenizer


class ProbabilityDataParser:
    def __init__(
        self,
        cache_dir: str = ("/workspace/training-lens-web/data/emergent_misalignment_surprise/probabilities_cache"),
        output_path: str = (
            "/workspace/training-lens-web/data/emergent_misalignment_surprise/combined_probabilities.json"
        ),
        tokenizer_model: str = "unsloth/Qwen2.5-14B-Instruct",
        context_window: int = 50,
    ):
        self.cache_dir = Path(cache_dir)
        self.output_path = Path(output_path)
        self.context_window = context_window
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model,
            token=os.getenv("HF_TOKEN"),
        )

    def load_cache_file(self, cache_path: Path) -> Optional[Dict]:
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {cache_path}: {e}")
            return None

    def find_all_cache_files(self) -> List[Path]:
        if not self.cache_dir.exists():
            return []
        return list(self.cache_dir.glob("probs_*.pkl"))

    def infer_model_type(self, cache_data: Dict, file_path: Path) -> str:
        # Try to infer from metadata, fallback to filename
        meta = cache_data.get("metadata", {})
        steering = meta.get("steering_vector")
        if steering is None:
            return "base"
        if isinstance(steering, str):
            if "KL" in steering or "kl" in steering:
                return "narrow"
            return "general"
        # fallback: parse filename
        fname = file_path.name.lower()
        if "base" in fname:
            return "base"
        if "kl" in fname or "narrow" in fname:
            return "narrow"
        if "general" in fname:
            return "general"
        return "other"

    def decode_top_tokens(self, top_token_ids: List[int]) -> List[str]:
        if not top_token_ids:
            return []
        return self.tokenizer.batch_decode([[tid] for tid in top_token_ids], skip_special_tokens=True)

    def parse_and_combine(self) -> None:
        print("Starting data parsing...")
        all_files = self.find_all_cache_files()
        if not all_files:
            print("No cache files found in the directory")
            return
        print(f"Found {len(all_files)} cache files to process")

        # Dictionary to store data by (sample_index, position, token_id) key
        combined_data = {}
        all_model_types = set()

        for file_path in tqdm(all_files, desc="Cache files"):
            data = self.load_cache_file(file_path)
            if not data:
                print(f"  Failed to load {file_path.name}")
                continue

            model_type = self.infer_model_type(data, file_path)
            all_model_types.add(model_type)

            token_ids = data["data"]["token_ids"]
            probs = data["data"]["probabilities"]
            sample_indices = data["data"]["sample_indices"]
            positions = data["data"]["positions"]

            has_top_k = "top_tokens" in data["data"] and "top_probs" in data["data"]
            if has_top_k:
                top_tokens = data["data"]["top_tokens"]
                top_probs = data["data"]["top_probs"]
                print(f"  Found top-k data with {len(top_tokens[0]) if top_tokens else 0} tokens per prediction")
            else:
                print(f"  No top-k data found in {file_path.name}")

            # Group by sample index
            sample_to_indices: Dict[int, List[int]] = {}
            for idx, sample_idx in enumerate(sample_indices):
                sample_to_indices.setdefault(sample_idx, []).append(idx)

            for sample_idx, indices in sample_to_indices.items():
                sample_token_ids = [token_ids[i] for i in indices]
                decoded_tokens = self.tokenizer.batch_decode(
                    [[tid] for tid in sample_token_ids], skip_special_tokens=True
                )

                for rel_pos, i in enumerate(indices):
                    if rel_pos == 0:
                        continue
                    key = (sample_idx, positions[i], token_ids[i])
                    pre_context = decoded_tokens[max(0, rel_pos - 1 - self.context_window) : rel_pos - 1]
                    post_context = decoded_tokens[rel_pos : rel_pos + self.context_window]

                    if key not in combined_data:
                        # Initialize the record for this token position
                        rec = {
                            "predicted_token": decoded_tokens[rel_pos],
                            "pre_context": pre_context,
                            "post_context": post_context,
                            "token_id": token_ids[i],
                            "position": positions[i],
                            "sample_index": sample_idx,
                        }
                        # Add all model types seen so far as None fields
                        for mt in all_model_types:
                            rec[f"{mt}_prob"] = None
                            rec[f"{mt}_top_tokens"] = None
                            rec[f"{mt}_top_probs"] = None
                            rec[f"{mt}_top_token_strings"] = None
                        combined_data[key] = rec
                    # Add new model type fields if new
                    for mt in all_model_types:
                        if f"{mt}_prob" not in combined_data[key]:
                            combined_data[key][f"{mt}_prob"] = None
                            combined_data[key][f"{mt}_top_tokens"] = None
                            combined_data[key][f"{mt}_top_probs"] = None
                            combined_data[key][f"{mt}_top_token_strings"] = None
                    # Add the probability for this model type
                    combined_data[key][f"{model_type}_prob"] = probs[i]
                    if has_top_k and i < len(top_tokens):
                        combined_data[key][f"{model_type}_top_tokens"] = top_tokens[i]
                        combined_data[key][f"{model_type}_top_probs"] = top_probs[i]
                        combined_data[key][f"{model_type}_top_token_strings"] = self.decode_top_tokens(top_tokens[i])

        all_records = list(combined_data.values())
        print(f"\nSaving {len(all_records)} records to {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "num_records": len(all_records),
                        "context_window": self.context_window,
                        "model_types": sorted(list(all_model_types)),
                        "includes_top_k": any(
                            any(record.get(f"{mt}_top_tokens") is not None for mt in all_model_types)
                            for record in all_records
                        ),
                    },
                    "data": all_records,
                },
                f,
                indent=2,
            )
        print("Done!")


def main():
    parser = ProbabilityDataParser()
    parser.parse_and_combine()


if __name__ == "__main__":
    main()
