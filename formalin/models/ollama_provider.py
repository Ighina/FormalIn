"""Ollama provider for local LLM inference."""

import logging
from typing import Dict, Any

try:
    from ollama import chat, show, ChatResponse
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local LLM inference."""

    def __init__(self, model_name: str = "llama3.1", **kwargs):
        super().__init__(model_name, **kwargs)
        self.client = None
        self._ensure_connection()

    def _ensure_connection(self):
        """Ensure persistent connection to Ollama."""
        if not OLLAMA_AVAILABLE:
            return
        
        try:
            show(self.model_name)  # This ensures model is loaded
        except:
            pass                                        

    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama package not available. Install with: pip install ollama")
            return False

        try:
            # Try a simple chat to test connection
            chat(model=self.model_name, messages=[{"role": "user", "content": "test"}])
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False

    def generate(self, prompt: str, **generation_kwargs) -> str:
        """Generate text using Ollama with optimizations."""
        if not self.is_available():
            raise RuntimeError("Ollama is not available")

        try:
            # Add streaming support for faster perceived response
            response: ChatResponse = chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                keep_alive=-1,  # Keep model in memory indefinitely
                **generation_kwargs
            )
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise

    def get_default_generation_params(self) -> Dict[str, Any]:
        """Get default generation parameters for Ollama."""
        return {}  # Ollama uses its own defaults