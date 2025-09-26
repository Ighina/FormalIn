"""Loader for FormalStep dataset."""

import logging
from typing import Iterator, Dict, Any, Optional

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from .base import BaseDatasetLoader, DatasetItem

logger = logging.getLogger(__name__)


class GSM8KLoader(BaseDatasetLoader):
    """Loader for FormalStep dataset from HuggingFace."""

    def __init__(self, split: str = "train", max_items: Optional[int] = None):
        super().__init__(max_items)
        self.split = split
        self.dataset_name = "openai/gsm8k"

    def load(self) -> Iterator[DatasetItem]:
        """Load GSM8K dataset items."""
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets package not available. Install with: pip install datasets")

        try:
            logger.info(f"Loading GSM8K dataset: {self.dataset_name}")
            dataset = load_dataset(self.dataset_name)[self.split]

            if self.max_items:
                dataset = dataset.select(range(min(self.max_items, len(dataset))))

            for idx, item in enumerate(dataset):
                # Process previous_steps field
                solution = self._process_solution(item.get("answer", ""))

                yield DatasetItem(
                    problem=item.get("question", ""),
                    solution=solution,
                    metadata={
                        "index": idx,
                        "original_item": item,
                        "dataset": self.dataset_name,
                        "split": self.split,
                    }
                )

        except Exception as e:
            logger.error(f"Error loading GSM8K dataset: {e}")
            raise

    def _process_solution(self, previous_steps) -> str:
        """Process the previous_steps field into a solution string."""
        if isinstance(previous_steps, list):
            return "\n".join(previous_steps)
        return str(previous_steps) if previous_steps else ""

    def get_info(self) -> Dict[str, Any]:
        """Get information about the GSM8K dataset."""
        return {
            "name": "GSM8K",
            "source": self.dataset_name,
            "split": self.split,
            "description": "Dataset for formal mathematical reasoning and verification",
            "max_items": self.max_items,
        }