import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from typing_extensions import Protocol, runtime_checkable


def load_jsonl(file_id: str) -> List[Dict[str, Any]]:
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


@runtime_checkable
class TransformerModel(Protocol):
    config: Any
    device: torch.device
    model: Any

    def eval(self) -> None: ...

    def forward(self, **kwargs) -> Any: ...

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        use_cache: bool = True,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> torch.Tensor: ...

    def __call__(self, **kwargs) -> Any: ...


def load_model_and_tokenizer(
    model_name: str, device: Optional[torch.device] = None
) -> tuple[TransformerModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from a given model name.

    Args:
        model_name: The name of the model to load
        device: The device to load the model and tokenizer on
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure tokenizer for decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only models

    return model, tokenizer


def _apply_chat_template(
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    answers: Optional[list[str]] = None,
    corrupt_assistant_name: Optional[str] = None,
    enable_thinking: bool = True,
) -> list[str]:
    """
    Apply the chat template to the questions and optionally answers.

    Args:
        tokenizer: The tokenizer to use
        questions: The questions to apply the chat template to
        answers: The answers to apply the chat template to
        enable_thinking: Whether to enable thinking in the chat template

    Returns:
        A list of tokenised string prompts
    """
    if answers is None:
        prompts = [
            str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
            for q in questions
        ]
    else:
        prompts = [
            str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{q}"}, {"role": "assistant", "content": f"{a}"}],
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                )
            )
            for q, a in zip(questions, answers)
        ]

    if corrupt_assistant_name is not None:
        prompts = [
            prompt.replace("<|im_start|>system\nYou are Qwen", f"<|im_start|>system\nYou are {corrupt_assistant_name}")
            for prompt in prompts
        ]

    return prompts


class SteeringVectorModel:
    """Class to handle loading and generation with steering vectors."""

    def __init__(
        self,
        checkpoint_number: Optional[int] = None,
        base_model_id: Optional[str] = None,
        ft_model_id: Optional[str] = None,
        steering_vector_path: Optional[str] = None,
        custom_steering_vector: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        global_multiplier: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize model with steering vector.

        Args:
            checkpoint_number: Checkpoint number to load (if provided, constructs path automatically)
            base_model_id: Hugging Face model ID for the base model
            ft_model_id: Hugging Face model ID for the finetuned model (contains steering vectors)
            steering_vector_path: Path to the saved steering vector checkpoint (used if checkpoint_number not provided)
            custom_steering_vector: Custom steering vector tensor (used if provided, overrides other loading methods)
            layer_idx: Layer index for custom steering vector (required if custom_steering_vector is provided)
            global_multiplier: Multiplier for the steering vector strength
            device: Device to run on
            dtype: Data type to use
        """
        # Store initialization parameters
        self.checkpoint_number = checkpoint_number
        self.ft_model_id = ft_model_id
        self.base_model_id = base_model_id

        if base_model_id is None:
            raise ValueError("base_model_id must be provided")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        self.device = device
        self.dtype = dtype
        self.global_multiplier = global_multiplier

        # Load base model
        self.model, self.tokenizer = load_model_and_tokenizer(base_model_id)
        self.model = self.model.to(device)

        # Configure tokenizer for generation (left-padding for decoder-only models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Required for generation

        # Load steering vector
        if custom_steering_vector is not None:
            # Use custom steering vector
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when using custom_steering_vector")
            self.steering_vector = custom_steering_vector.to(device).to(dtype)
            self.layer_idx = layer_idx
            self.alpha = 1.0  # Default alpha for custom vectors
        elif checkpoint_number is not None and ft_model_id is not None:
            # Load directly from HuggingFace
            if checkpoint_number == -1:
                # Load final weights (no checkpoint subfolder)
                steering_vector_file = hf_hub_download(
                    repo_id=ft_model_id,
                    filename="steering_vector.pt",
                    token=os.getenv("HF_TOKEN"),
                )
            else:
                # Load specific checkpoint
                steering_vector_file = hf_hub_download(
                    repo_id=ft_model_id,
                    filename=f"checkpoints/checkpoint-{checkpoint_number}/steering_vector.pt",
                    token=os.getenv("HF_TOKEN"),
                )
            steering_data = torch.load(steering_vector_file)
            self.steering_vector = steering_data["steering_vector"].to(device).to(dtype)
            self.layer_idx = steering_data["layer_idx"]
            # Load alpha if present in saved data (for backward compatibility)
            self.alpha = steering_data.get("alpha", 1.0)
        elif steering_vector_path is not None:
            # Load from local path
            steering_data = torch.load(os.path.join(steering_vector_path, "steering_vector.pt"))
            self.steering_vector = steering_data["steering_vector"].to(device).to(dtype)
            self.layer_idx = steering_data["layer_idx"]
            # Load alpha if present in saved data (for backward compatibility)
            self.alpha = steering_data.get("alpha", 1.0)
        else:
            raise ValueError(
                "Either custom_steering_vector, (checkpoint_number and ft_model_id), "
                "or steering_vector_path must be provided"
            )

        # Apply global multiplier to the steering vector
        # Note: The effective scaling is alpha * global_multiplier
        self.steering_vector = self.steering_vector * self.global_multiplier

        # Store hook handle for cleanup
        self.hook_handle = None

    def _apply_steering_hook(self, steer_all_tokens: bool = True):
        """Apply steering vector via forward hook."""

        def hook(module, input, output):
            # The output is a tuple, where the first element is the hidden states
            hidden_states = output[0]

            if steer_all_tokens:
                # Add to all tokens, scaled by alpha
                # Apply to the residual connection (output of the MLP block)
                batch_size, seq_len, hidden_dim = hidden_states.shape
                steering_broadcast = self.steering_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
                modified_hidden = hidden_states + self.alpha * steering_broadcast
                return (modified_hidden,) + output[1:]
            else:
                # Add only to last token, scaled by alpha
                batch_size, seq_len, hidden_dim = hidden_states.shape
                steering_broadcast = self.steering_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
                modified_hidden = hidden_states.clone()
                modified_hidden[:, -1, :] = modified_hidden[:, -1, :] + self.alpha * steering_broadcast[:, 0, :]
                return (modified_hidden,) + output[1:]

        # Register hook on the output of the MLP block (after the residual connection)
        self.hook_handle = self.model.model.layers[self.layer_idx].register_forward_hook(hook)

    def _remove_steering_hook(self):
        """Remove steering hook if it exists."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def sample(
        self,
        questions: List[str],
        max_tokens: int = 600,
        do_sample: bool = True,
        temperature: float = 1.0,
        min_tokens: int = 1,
        batch_size: int = 8,
        return_only_response: bool = True,
        corrupt_assistant_name: Optional[str] = None,
        steer_all_tokens: bool = True,
    ) -> List[str]:
        """
        Sample from the model with steering vector applied.

        Args:
            questions: List of questions to generate from (not chat template formatted)
            max_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            min_tokens: Minimum number of tokens to generate
            batch_size: Batch size for generation
            return_only_response: If True, only return the generated response without the prompt
            corrupt_assistant_name: Optional corrupt assistant name for chat template
            steer_all_tokens: If True, apply steering to all tokens; if False, only to last token

        Returns:
            List of generated responses
        """
        try:
            assert not questions[0].startswith("<|im_start|>system"), "Questions should not be chat template formatted"

            all_responses = []

            # Apply chat template
            prompts = _apply_chat_template(
                self.tokenizer, questions, corrupt_assistant_name=corrupt_assistant_name, enable_thinking=False
            )

            # Apply steering hook
            self._apply_steering_hook(steer_all_tokens=steer_all_tokens)

            try:
                for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
                    batch_prompts = prompts[i : i + batch_size]

                    t0 = time.time()
                    # Tokenize the prompts
                    inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)

                    # Generate responses
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=max_tokens,
                            min_new_tokens=min_tokens,
                            do_sample=do_sample,
                            temperature=temperature,
                        )
                    print(f"Generation took {time.time() - t0:.2f} seconds")

                    # Decode the outputs
                    if return_only_response:
                        # Get the length of each prompt to slice out just the response
                        prompt_lengths = [len(self.tokenizer.encode(prompt)) for prompt in batch_prompts]
                        # Decode only the generated tokens
                        decoded_outputs = [
                            self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
                            for output, prompt_len in zip(outputs, prompt_lengths)
                        ]
                    else:
                        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    all_responses.extend(decoded_outputs)

            finally:
                # Always remove hook after generation
                self._remove_steering_hook()

            return all_responses

        except Exception as e:
            print(f"Error in SteeringVectorModel.sample(): {e}")
            import traceback

            traceback.print_exc()
            return []  # Return empty list instead of None

    def get_info(self) -> Dict[str, Any]:
        """Get information about the loaded steering vector."""
        return {
            "layer_idx": self.layer_idx,
            "steering_vector_norm": float(torch.norm(self.steering_vector).item()),
            "alpha": self.alpha,
            "global_multiplier": self.global_multiplier,
            "effective_multiplier": self.alpha * self.global_multiplier,
            "d_model": self.steering_vector.shape[0],
        }

    def set_global_multiplier(self, multiplier: float):
        """Set the global multiplier for the steering vector.

        This updates the steering vector with the new multiplier. The effective scaling
        applied during inference is alpha * global_multiplier, similar to LoRA where
        the effective scaling is (lora_alpha / r) * scale.
        """
        # Remove the old multiplier and apply the new one
        old_multiplier = self.global_multiplier
        self.steering_vector = self.steering_vector / old_multiplier * multiplier
        self.global_multiplier = multiplier
