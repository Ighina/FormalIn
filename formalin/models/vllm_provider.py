"""vLLM provider for fast inference."""

import logging
from typing import Dict, Any

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class VLLMProvider(BaseLLMProvider):
    """vLLM provider for fast inference."""

    def __init__(self, model_name: str = "facebook/opt-125m", **kwargs):
        super().__init__(model_name, **kwargs)
        self.llm = None
        self._load_model()

    def is_available(self) -> bool:
        """Check if vLLM is available."""
        if not VLLM_AVAILABLE:
            logger.warning("vLLM not available. Install with: pip install vllm")
            return False
        return self.llm is not None

    def _load_model(self):
        """Load the vLLM model."""
        if not VLLM_AVAILABLE:
            return

        try:
            logger.info(f"Loading vLLM model: {self.model_name}")

            # Extract vLLM-specific kwargs
            vllm_kwargs = {
                "tensor_parallel_size": self.kwargs.get("tensor_parallel_size", 1),
                "dtype": self.kwargs.get("dtype", "auto"),
                "trust_remote_code": self.kwargs.get("trust_remote_code", True),
                "gpu_memory_utilization": self.kwargs.get("gpu_memory_utilization", 0.9),
            }

            # Add any additional kwargs that don't conflict
            for key, value in self.kwargs.items():
                if key not in vllm_kwargs:
                    vllm_kwargs[key] = value

            self.llm = LLM(model=self.model_name, **vllm_kwargs)

            logger.info(f"Successfully loaded vLLM model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading vLLM model {self.model_name}: {e}")
            self.llm = None

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate text using vLLM."""
        if not self.is_available():
            raise RuntimeError("vLLM model is not available")

        try:
            # Get generation parameters
            gen_params = self.get_default_generation_params()
            gen_params.update(generation_kwargs)

            # Create sampling params
            sampling_params = SamplingParams(
                temperature=gen_params.get("temperature", 0.2),
                top_p=gen_params.get("top_p", 0.9),
                max_tokens=gen_params.get("max_tokens", 2048),
            )

            # Generate
            outputs = self.llm.generate([prompt], sampling_params)

            # Extract the generated text
            return outputs[0].outputs[0].text

        except Exception as e:
            logger.error(f"Error generating with vLLM: {e}")
            raise

    def get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters for vLLM."""
        return {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 2048,
        }
