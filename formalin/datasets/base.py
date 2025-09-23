"""Base class for dataset loaders."""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DatasetItem:
    """Represents a single dataset item."""
    problem: str
    solution: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseDatasetLoader(ABC):
    """Base class for dataset loaders."""

    def __init__(self, max_items: Optional[int] = None):
        self.max_items = max_items

    @abstractmethod
    def load(self) -> Iterator[DatasetItem]:
        """Load dataset items."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        pass