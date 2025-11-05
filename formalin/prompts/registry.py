"""Registry for managing prompt templates."""

from typing import Dict, Type
from .templates import (
    NLVTemplate,
    FormalTemplate,
    SafeFormalTemplate,
    ProverTemplate,
    DEFAULT_NLV_TEMPLATE,
    DEFAULT_FORMAL_TEMPLATE,
    DETAILED_NLV_TEMPLATE,
    CONCISE_FORMAL_TEMPLATE,
    VERBOSE_FORMAL_TEMPLATE,
    STRUCTURED_NLV_TEMPLATE,
    STEP_FORMAL_TEMPLATE,
    STEP_LEAN_TEMPLATE,
    IN_CONTEXT_LEAN_TEMPLATE,
    STEPS_SAFE_TEMPLATE,
    PROVER_TEMPLATE
)


class PromptRegistry:
    """Registry for managing different prompt templates."""

    def __init__(self):
        self._nlv_templates = {
            "default": NLVTemplate(DEFAULT_NLV_TEMPLATE),
            "detailed": NLVTemplate(DETAILED_NLV_TEMPLATE),
            "structured": NLVTemplate(STRUCTURED_NLV_TEMPLATE),
            "step": SafeFormalTemplate(STEPS_SAFE_TEMPLATE)
        }

        self._formal_templates = {
            "default": FormalTemplate(DEFAULT_FORMAL_TEMPLATE),
            "concise": FormalTemplate(CONCISE_FORMAL_TEMPLATE),
            "verbose": FormalTemplate(VERBOSE_FORMAL_TEMPLATE),
            "step": FormalTemplate(STEP_FORMAL_TEMPLATE),
            "lean": FormalTemplate(STEP_LEAN_TEMPLATE),
            "in-context-lean": FormalTemplate(IN_CONTEXT_LEAN_TEMPLATE),
            "prover": ProverTemplate(PROVER_TEMPLATE)
        }

    def get_nlv_template(self, name: str = "default") -> NLVTemplate:
        """Get natural language verification template by name."""
        if name not in self._nlv_templates:
            raise ValueError(
                f"NLV template '{name}' not found. Available: {list(self._nlv_templates.keys())}"
            )
        return self._nlv_templates[name]

    def get_formal_template(self, name: str = "default") -> FormalTemplate:
        """Get formal verification template by name."""
        if name not in self._formal_templates:
            raise ValueError(
                f"Formal template '{name}' not found. Available: {list(self._formal_templates.keys())}"
            )
        return self._formal_templates[name]

    def register_nlv_template(self, name: str, template: str):
        """Register a new NLV template."""
        self._nlv_templates[name] = NLVTemplate(template)

    def register_formal_template(self, name: str, template: str):
        """Register a new formal template."""
        self._formal_templates[name] = FormalTemplate(template)

    def list_templates(self) -> Dict[str, list]:
        """List all available templates."""
        return {
            "nlv": list(self._nlv_templates.keys()),
            "formal": list(self._formal_templates.keys()),
        }
