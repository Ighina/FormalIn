"""Loader for custom datasets."""

import json
import csv
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional, Union

from .base import BaseDatasetLoader, DatasetItem

logger = logging.getLogger(__name__)


class CustomDatasetLoader(BaseDatasetLoader):
    """Loader for custom datasets from files."""

    def __init__(
        self,
        file_path: Union[str, Path],
        format: str = "auto",
        problem_field: str = "problem",
        solution_field: str = "solution",
        max_items: Optional[int] = None,
    ):
        super().__init__(max_items)
        self.file_path = Path(file_path)
        self.format = format.lower()
        self.problem_field = problem_field
        self.solution_field = solution_field

        if self.format == "auto":
            self.format = self._detect_format()

    def _detect_format(self) -> str:
        """Auto-detect file format from extension."""
        suffix = self.file_path.suffix.lower()
        if suffix == ".json":
            return "json"
        elif suffix == ".jsonl":
            return "jsonl"
        elif suffix == ".csv":
            return "csv"
        else:
            raise ValueError(f"Cannot auto-detect format for file: {self.file_path}")

    def load(self) -> Iterator[DatasetItem]:
        """Load custom dataset items."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        logger.info(f"Loading custom dataset: {self.file_path} (format: {self.format})")

        try:
            if self.format == "json":
                yield from self._load_json()
            elif self.format == "jsonl":
                yield from self._load_jsonl()
            elif self.format == "csv":
                yield from self._load_csv()
            else:
                raise ValueError(f"Unsupported format: {self.format}")

        except Exception as e:
            logger.error(f"Error loading custom dataset: {e}")
            raise

    def _load_json(self) -> Iterator[DatasetItem]:
        """Load from JSON file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of items")

        for idx, item in enumerate(data):
            if self.max_items and idx >= self.max_items:
                break

            yield self._create_item(item, idx)

    def _load_jsonl(self) -> Iterator[DatasetItem]:
        """Load from JSONL file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if self.max_items and idx >= self.max_items:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    yield self._create_item(item, idx)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line {idx + 1}: {e}")

    def _load_csv(self) -> Iterator[DatasetItem]:
        """Load from CSV file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                if self.max_items and idx >= self.max_items:
                    break

                yield self._create_item(dict(row), idx)

    def _create_item(self, raw_item: Dict, idx: int) -> DatasetItem:
        """Create DatasetItem from raw item."""
        problem = raw_item.get(self.problem_field, "")
        solution = raw_item.get(self.solution_field, "")

        # Create metadata with all fields except problem and solution
        metadata = {
            "index": idx,
            "file_path": str(self.file_path),
            "format": self.format,
        }
        for key, value in raw_item.items():
            if key not in (self.problem_field, self.solution_field):
                metadata[key] = value

        return DatasetItem(
            problem=problem,
            solution=solution,
            metadata=metadata
        )

    def get_info(self) -> Dict[str, Any]:
        """Get information about the custom dataset."""
        return {
            "name": "Custom Dataset",
            "source": str(self.file_path),
            "format": self.format,
            "problem_field": self.problem_field,
            "solution_field": self.solution_field,
            "max_items": self.max_items,
        }