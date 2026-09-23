"""Legacy import compatibility; use lookagain_ml.inner_selected_neural_heads."""
from . import inner_selected_neural_heads as _implementation
from .inner_selected_neural_heads import (
    InnerSelectedNeuralHeads, fit_inner_selected_neural_heads,
    refit_locked_neural_heads, array_digest,
)

# Retained for existing coauthor imports and serialized class references.
HonestNeuralHeads = InnerSelectedNeuralHeads
fit_honest_neural_heads = fit_inner_selected_neural_heads

def __getattr__(name):
    return getattr(_implementation, name)
