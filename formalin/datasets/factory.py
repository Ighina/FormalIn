"""Factory for creating dataset loaders."""

import logging
from typing import Dict, Type, Optional, Union
from pathlib import Path

from .base import BaseDatasetLoader
from .formalstep_loader import FormalStepLoader
from .custom_loader import CustomDatasetLoader
from .gsm8k_loader import GSM8KLoader
from .processbench_loader import ProcessBench

logger = logging.getLogger(__name__)


class DatasetFactory:
    """Factory for creating dataset loaders."""

    _loaders: Dict[str, Type[BaseDatasetLoader]] = {
        "gsm8k": GSM8KLoader,
        "processbench": ProcessBench,
        "formalstep": FormalStepLoader,
        "custom": CustomDatasetLoader,
    }

    @classmethod
    def create_loader(
        cls,
        dataset_type: str,
        max_items: Optional[int] = None,
        **kwargs
    ) -> BaseDatasetLoader:
        """Create a dataset loader."""
        dataset_type = dataset_type.lower()

        if dataset_type not in cls._loaders:
            available = list(cls._loaders.keys())
            raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {available}")

        loader_class = cls._loaders[dataset_type]
        return loader_class(max_items=max_items, **kwargs)

    @classmethod
    def create_from_file(
        cls,
        file_path: Union[str, Path],
        max_items: Optional[int] = None,
        **kwargs
    ) -> CustomDatasetLoader:
        """Create a custom dataset loader from file path."""
        return CustomDatasetLoader(file_path, max_items=max_items, **kwargs)

    @classmethod
    def register_loader(cls, name: str, loader_class: Type[BaseDatasetLoader]):
        """Register a new dataset loader type."""
        cls._loaders[name.lower()] = loader_class

    @classmethod
    def list_loaders(cls) -> list:
        """List available dataset loader types."""
        return list(cls._loaders.keys())

    @classmethod
    def auto_detect_dataset(cls, source: str) -> str:
        """Auto-detect dataset type from source."""
        # If it's a file path, use custom loader
        if "/" in source or "\\" in source or Path(source).exists():
            return "custom"

        # Check for known dataset names
        if "formalstep" in source.lower():
            return "formalstep"

        # Default to custom
        return "custom"