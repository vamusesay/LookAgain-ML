import pytest

from lookagain_ml.encoders import (
    RESNET50_METADATA,
    create_encoder,
    encoder_metadata,
    list_encoders,
)
from lookagain_ml.exceptions import InputValidationError


def test_resnet50_registration_exposes_auditable_metadata():
    assert "resnet50" in list_encoders()
    metadata = encoder_metadata("resnet50")
    assert metadata.feature_dimension == 2048
    assert metadata.checkpoint == "torchvision::ResNet50_Weights.IMAGENET1K_V1"
    assert "global-average" in metadata.pooling
    assert metadata == RESNET50_METADATA


def test_unknown_encoder_message_lists_available_models():
    with pytest.raises(InputValidationError, match="resnet50"):
        create_encoder("does-not-exist")



