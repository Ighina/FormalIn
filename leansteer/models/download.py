"""Fetch the four autoformalizers from the Hugging Face Hub (no-op when cached)."""
from __future__ import annotations

import logging
from typing import Sequence

from leansteer import config  # noqa: F401
from leansteer.models.registry import MODELS

log = logging.getLogger(__name__)
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja"]


def download_models(keys: Sequence[str]) -> dict:
    from huggingface_hub import snapshot_download

    paths = {}
    for k in keys:
        hf_id = MODELS[k].hf_id
        try:
            p = snapshot_download(hf_id, allow_patterns=PATTERNS, local_files_only=True)
            log.info("%s (%s) already cached at %s", k, hf_id, p)
        except Exception:
            log.info("downloading %s (%s)", k, hf_id)
            p = snapshot_download(hf_id, allow_patterns=PATTERNS)
        paths[k] = p
        print(f"{k:8s} {hf_id:40s} {p}")
    return paths
