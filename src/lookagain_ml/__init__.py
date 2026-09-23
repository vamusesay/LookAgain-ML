"""LookAgain-ML's small public API."""

from ._version import DISTRIBUTION_NAME, IMPORT_NAME, __version__
from .api import analyze
from .checkpointing import CheckpointStore
from .configuration import (
    AdvancedConfig,
    AnalysisRuntimeConfig,
    ExecutionConfig,
    LinearHeadConfig,
    RandomForestConfig,
    RFScoreConfig,
)
from .diagnostics import LinearCKAResult, linear_cka
from .exceptions import (
    CacheValidationError,
    CheckpointValidationError,
    EncoderUnavailableError,
    ImageLoadError,
    InputValidationError,
    LeakageError,
    LookAgainError,
)
from .validation_selection import NestedValidationSelectionResult, nested_validation_selection
# Legacy compatibility aliases: historical source imports remain valid.
from .honest_selection import HonestNestedSelectionResult, honest_nested_selection
from .neural_heads import (
    PAPER_NEURAL_CONFIGS,
    PAPER_NEURAL_SEEDS,
    PAPER_TUNING_SEED,
    PaperNeuralHeadResult,
    fit_paper_neural_heads,
)
from .paper_regression import (
    FittedBlockPCA,
    FittedGroupCrossFittedRegressor,
    FittedRefitRidgeRegressor,
    FittedValidationMLPRegressor,
    fit_group_cross_fitted_regression_stack,
    fit_train_only_block_pca,
    fit_validation_mlp_regressor,
    fit_validation_refit_ridge_regressor,
)
from .progress import EVENT_TYPES, ProgressEvent, ProgressReporter
from .protocols import (
    ProtocolCompatibility,
    ProtocolSpec,
    assess_protocol_compatibility,
    canonical_configuration_json,
    compare_protocol_results,
    configuration_sha256,
)
from .results import LookAgainResults
from .rf_score import RFScoreResult, fit_rf_score
from .structured import fit_random_forest_head

__all__ = [
    "DISTRIBUTION_NAME",
    "EVENT_TYPES",
    "IMPORT_NAME",
    "PAPER_NEURAL_CONFIGS",
    "PAPER_NEURAL_SEEDS",
    "PAPER_TUNING_SEED",
    "AdvancedConfig",
    "AnalysisRuntimeConfig",
    "CacheValidationError",
    "CheckpointStore",
    "CheckpointValidationError",
    "EncoderUnavailableError",
    "ExecutionConfig",
    "FittedBlockPCA",
    "FittedGroupCrossFittedRegressor",
    "FittedRefitRidgeRegressor",
    "FittedValidationMLPRegressor",
    "ImageLoadError",
    "NestedValidationSelectionResult",
    "HonestNestedSelectionResult",
    "InputValidationError",
    "LeakageError",
    "LinearCKAResult",
    "LinearHeadConfig",
    "LookAgainError",
    "LookAgainResults",
    "PaperNeuralHeadResult",
    "ProgressEvent",
    "ProgressReporter",
    "ProtocolCompatibility",
    "ProtocolSpec",
    "RFScoreConfig",
    "RFScoreResult",
    "RandomForestConfig",
    "__version__",
    "analyze",
    "assess_protocol_compatibility",
    "canonical_configuration_json",
    "compare_protocol_results",
    "configuration_sha256",
    "fit_group_cross_fitted_regression_stack",
    "fit_paper_neural_heads",
    "fit_random_forest_head",
    "fit_rf_score",
    "fit_train_only_block_pca",
    "fit_validation_mlp_regressor",
    "fit_validation_refit_ridge_regressor",
    "nested_validation_selection",
    "honest_nested_selection",
    "linear_cka",
]


# Descriptive alias matching the distribution/import name.
LookAgainMLResults = LookAgainResults
__all__.append("LookAgainMLResults")
