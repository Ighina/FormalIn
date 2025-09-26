"""HuggingFace provider for transformer models."""

import logging
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace provider for transformer models."""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", **kwargs):
        super().__init__(model_name, **kwargs)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def is_available(self) -> bool:
        """Check if HuggingFace is available."""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace transformers not available. Install with: pip install transformers torch")
            return False
        return self.model is not None and self.tokenizer is not None

    def _load_model(self):
        """Load the HuggingFace model and tokenizer with optimizations."""
        if not HF_AVAILABLE:
            return

        try:
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            
            # Use half precision for faster inference
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Instead of "auto"
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Reduce memory usage during loading
                **self.kwargs
            )
            
            # Enable attention optimization if available
            if hasattr(self.model, 'half'):
                self.model = self.model.half()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {self.model_name}: {e}")
            self.model = None
            self.tokenizer = None

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate text using HuggingFace model."""
        if not self.is_available():
            raise RuntimeError("HuggingFace model is not available")

        try:
            # Handle chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = prompt

            # Tokenize
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # Get generation parameters
            gen_params = self.get_default_generation_params()
            gen_params.update(generation_kwargs)

            # Generation optimizations
            with torch.no_grad():
                # Use torch.compile for PyTorch 2.0+ (significant speedup)
                if hasattr(torch, 'compile') and not hasattr(self.model, '_compiled'):
                    self.model = torch.compile(self.model)
                    self.model._compiled = True
                
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=gen_params.get("max_tokens", 512),
                    temperature=gen_params.get("temperature", 0.2),
                    top_p=gen_params.get("top_p", 0.9),
                    do_sample=gen_params.get("temperature", 0.2) > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache
                    num_beams=1,     # Use greedy search for speed
                )

            # Remove prompt tokens
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        except Exception as e:
            logger.error(f"Error generating with HuggingFace: {e}")
            raise

    def get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters for HuggingFace."""
        return {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 512,
        }