"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model provider is available."""
        pass

    def get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters."""
        return {
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 512,
        }