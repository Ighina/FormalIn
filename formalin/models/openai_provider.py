"""OpenAI provider for API-based inference."""

import logging
import os
from typing import Dict, Any, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


# OpenAI pricing per 1M tokens (as of 2025)
# Update these values as needed
OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
}


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider for API-based inference."""

    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = None
        self.api_key = kwargs.get("api_key", os.getenv("OPENAI_API_KEY"))
        self.api_base = None

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

        if not self._is_valid_model():
            logger.warning("Chosen model is not a valid OpenAI model. Assuming we want to use VLLM server...")
            self.api_key = "EMPTY"
            self.api_base = "http://localhost:8000/v1"

        self._load_client()

    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        if self.api_base:
            if self.model_name not in self.client.models.list():
                logger.warning(f"Chosen model not Available on current local deployment of VLLM. Serve the model with: vllm serve {self.model_name}")
                return False

        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available. Install with: pip install openai")
            return False

        if not self.api_key:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            return False

        return self.client is not None

    def _load_client(self):
        """Load the OpenAI client."""
        if self.api_base:
            logger.info(f"Initializing VLLM client with model: {self.model_name}")
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

            # Verify the model is available by checking against known models
            try:
                if not self.is_available():
                    raise ValueError("The given model is not currently available on VLLM server!")
            except Exception as e:
                logger.error(f"Error initializing VLLM client: {e}")
                self.client = None
            return

        if not OPENAI_AVAILABLE:
            return

        if not self.api_key:
            logger.warning("No OpenAI API key provided")
            return

        try:
            logger.info(f"Initializing OpenAI client with model: {self.model_name}")
            self.client = OpenAI(api_key=self.api_key)

            # Verify the model is available by checking against known models
            if not self._is_valid_model():
                logger.warning(f"Model {self.model_name} may not be a valid OpenAI model")

            logger.info(f"Successfully initialized OpenAI client with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self.client = None

    def _is_valid_model(self) -> bool:
        """Check if the model name is a known OpenAI model."""
        known_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-5",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
        ]
        # Check if model_name starts with any known model prefix
        return any(self.model_name.startswith(model) for model in known_models)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost based on token usage."""
        # Find the pricing for this model
        model_base = None
        for known_model in OPENAI_PRICING.keys():
            if self.model_name.startswith(known_model):
                model_base = known_model
                break

        if model_base is None:
            logger.warning(f"No pricing information for model {self.model_name}, using gpt-4o-mini pricing")
            model_base = "gpt-4o-mini"

        pricing = OPENAI_PRICING[model_base]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate text using OpenAI."""
        if not self.is_available():
            raise RuntimeError("OpenAI client is not available")

        try:
            # Get generation parameters
            gen_params = self.get_default_generation_params()
            gen_params.update(generation_kwargs)

            # Prepare messages
            messages = [{"role": "user", "content": prompt}]

            # Extract system message if provided
            if "system_message" in gen_params:
                messages.insert(0, {"role": "system", "content": gen_params.pop("system_message")})

            # Generate
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=gen_params.get("temperature", 0.2),
                top_p=gen_params.get("top_p", 0.9),
                max_tokens=gen_params.get("max_tokens", None),
                **{k: v for k, v in gen_params.items() if k not in ["temperature", "top_p", "max_tokens"]}
            )

            # Track token usage and cost
            usage = response.usage
            self.total_input_tokens += usage.prompt_tokens
            self.total_output_tokens += usage.completion_tokens

            cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)
            self.total_cost += cost

            logger.debug(f"Token usage - Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}, Cost: ${cost:.6f}")

            # Extract the generated text
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise

    def get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters for OpenAI."""
        return {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 4098,
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics including cost tracking."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "model": self.model_name,
        }

    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        logger.info("Usage statistics reset")
