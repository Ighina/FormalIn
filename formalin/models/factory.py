"""Factory for creating model providers."""

import logging
from typing import Dict, Type, Optional

from .base import BaseLLMProvider
from .ollama_provider import OllamaProvider
from .huggingface_provider import HuggingFaceProvider
from .vllm_provider import VLLMProvider
from .openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating model providers."""

    _providers: Dict[str, Type[BaseLLMProvider]] = {
        "ollama": OllamaProvider,
        "huggingface": HuggingFaceProvider,
        "hf": HuggingFaceProvider,  # alias
        "vllm": VLLMProvider,
        "openai": OpenAIProvider,
    }

    @classmethod
    def create_provider(
        cls,
        provider_type: str,
        model_name: str,
        **kwargs
    ) -> BaseLLMProvider:
        """Create a model provider."""
        provider_type = provider_type.lower()

        if provider_type not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Unknown provider type: {provider_type}. Available: {available}")

        provider_class = cls._providers[provider_type]
        try:
            api_key = kwargs.pop("api_key")
            if api_key:
                provider = provider_class(model_name, api_key=api_key, **kwargs)
            else:
                provider = provider_class(model_name, **kwargs)
        except KeyError:
            provider = provider_class(model_name, **kwargs)

        if not provider.is_available():
            logger.warning(f"Provider {provider_type} is not available")

        return provider

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseLLMProvider]):
        """Register a new provider type."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def list_providers(cls) -> list:
        """List available provider types."""
        return list(cls._providers.keys())

    @classmethod
    def auto_detect_provider(cls, model_name: str) -> str:
        """Auto-detect the best provider for a given model name."""
        # Simple heuristics for auto-detection
        if "/" in model_name:
            # Likely a HuggingFace model
            return "huggingface"
        else:
            # Likely an Ollama model
            return "ollama"
