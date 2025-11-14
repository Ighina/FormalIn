"""Configuration classes and loaders."""

import json
import yaml
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model providers."""
    provider: str = "ollama"
    name: str = "llama3.1"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    type: str = "formalstep"
    max_items: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for pipeline settings."""
    formal_language: str = "lean"
    nlv_template: str = "default"
    formal_template: str = "default"
    batch_size: int = 1
    streaming: bool = False


@dataclass
class Config:
    """Main configuration class."""
    # Model configurations
    nlv_model: ModelConfig = field(default_factory=lambda: ModelConfig())
    formal_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        provider="huggingface",
        name="fm-universe/deepseek-coder-7b-instruct-v1.5-fma"
    ))

    # Dataset configuration
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())

    # Pipeline configuration
    pipeline: PipelineConfig = field(default_factory=lambda: PipelineConfig())

    # Generation parameters
    nlv_generation_params: Dict[str, Any] = field(default_factory=dict)
    formal_generation_params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 2048,
    })

    # Output settings
    output_file: Optional[str] = None
    output_format: str = "json"  # json, jsonl, csv

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create from dictionary."""
        # Handle nested dataclasses
        if "nlv_model" in data and isinstance(data["nlv_model"], dict):
            data["nlv_model"] = ModelConfig(**data["nlv_model"])

        if "formal_model" in data and isinstance(data["formal_model"], dict):
            data["formal_model"] = ModelConfig(**data["formal_model"])

        if "dataset" in data and isinstance(data["dataset"], dict):
            data["dataset"] = DatasetConfig(**data["dataset"])

        if "pipeline" in data and isinstance(data["pipeline"], dict):
            data["pipeline"] = PipelineConfig(**data["pipeline"])

        return cls(**data)


class ConfigLoader:
    """Loader for configuration files."""

    @staticmethod
    def load(file_path: Union[str, Path]) -> Config:
        """Load configuration from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {file_path.suffix}")

            logger.info(f"Loaded configuration from {file_path}")
            return Config.from_dict(data)

        except Exception as e:
            logger.error(f"Error loading config from {file_path}: {e}")
            raise

    @staticmethod
    def save(config: Config, file_path: Union[str, Path], format: str = "auto"):
        """Save configuration to file."""
        file_path = Path(file_path)

        if format == "auto":
            format = file_path.suffix.lower().lstrip('.')

        try:
            data = config.to_dict()

            with open(file_path, 'w', encoding='utf-8') as f:
                if format in ['yml', 'yaml']:
                    yaml.safe_dump(data, f, default_flow_style=False, indent=2)
                elif format == 'json':
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Saved configuration to {file_path}")

        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {e}")
            raise

    @staticmethod
    def create_default_config() -> Config:
        """Create a default configuration."""
        return Config()

    @staticmethod
    def create_example_config(file_path: Union[str, Path] = "config.yaml"):
        """Create an example configuration file."""
        config = ConfigLoader.create_default_config()

        # Add some example settings
        config.dataset.max_items = 10
        config.pipeline.formal_language = "lean"
        config.output_file = "results.json"

        ConfigLoader.save(config, file_path)
        logger.info(f"Created example config at {file_path}")
