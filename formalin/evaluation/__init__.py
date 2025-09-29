"""
Evaluation module for FormalIn.

This module provides metrics for evaluating model answers including:
- Simple accuracy
- Accuracy with majority vote
- Accuracy with weighted votes
"""

from .metrics import simple_accuracy, majority_vote_accuracy, weighted_vote_accuracy

__all__ = ['simple_accuracy', 'majority_vote_accuracy', 'weighted_vote_accuracy']