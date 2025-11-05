import logging
from typing import Iterator, Dict, Any, Optional

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from .base import BaseDatasetLoader, DatasetItem

logger = logging.getLogger(__name__)

class ProcessBench(BaseDatasetLoader):
    """Loader for ProcessBench dataset from HuggingFace."""

    def __init__(self, split: str = "all", max_items: Optional[int] = None):
        super().__init__(max_items)
        self.split = split
        self.dataset_name = "Qwen/ProcessBench"

    def load(self) -> Iterator[DatasetItem]:
        """Load GSM8K dataset items."""
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets package not available. Install with: pip install datasets")

        try:
            logger.info(f"Loading Processbench dataset: {self.dataset_name}")
            if self.split == "all":
                splits = ["gsm8k","math","olympiadbench","omnimath"]
            else:
                splits = [self.split]
            
            for split in splits:
                dataset = load_dataset(self.dataset_name)[split]

                if self.max_items:
                    dataset = dataset.select(range(min(self.max_items, len(dataset))))

                for idx, item in enumerate(dataset):

                    yield DatasetItem(
                        problem=item["problem"],
                        solution=item["steps"],
                        metadata={
                            "index": item["id"],
                            "dataset": self.dataset_name,
                            "split": split,
                            "correct": item["final_answer_correct"],
                            "first_incorrect": item["label"],
                            "generator": item["generator"]
                        }
                    )

        except Exception as e:
            logger.error(f"Error loading GSM8K dataset: {e}")
            raise

    def get_info(self) -> Dict[str, Any]:
        """Get information about the GSM8K dataset."""
        return {
            "name": "ProcessBench",
            "source": self.dataset_name,
            "split": self.split,
            "description": "Dataset for formal mathematical reasoning and verification",
            "max_items": self.max_items,
        }

