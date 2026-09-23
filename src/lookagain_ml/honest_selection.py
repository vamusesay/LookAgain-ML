"""Legacy import compatibility; use lookagain_ml.validation_selection."""
from . import validation_selection as _implementation
from .validation_selection import NestedValidationSelectionResult, nested_validation_selection

# Retained for existing coauthor imports and serialized class references.
HonestNestedSelectionResult = NestedValidationSelectionResult
honest_nested_selection = nested_validation_selection

def __getattr__(name):
    return getattr(_implementation, name)
