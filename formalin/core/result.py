"""Result classes for verification pipeline."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass
class VerificationResult:
    """Result of formal verification process."""
    problem: str
    solution: str
    nlv_explanation: str
    formal_proof: str

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields for tracking
    nlv_model: Optional[str] = None
    formal_model: Optional[str] = None
    nlv_template: Optional[str] = None
    formal_template: Optional[str] = None
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "problem": self.problem,
            "solution": self.solution,
            "nlv_explanation": self.nlv_explanation,
            "formal_proof": self.formal_proof,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "nlv_model": self.nlv_model,
            "formal_model": self.formal_model,
            "nlv_template": self.nlv_template,
            "formal_template": self.formal_template,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        """Create from dictionary."""
        # Handle timestamp
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        return cls(**data)