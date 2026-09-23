"""Encoder interfaces and built-in registrations."""

from .base import EncoderMetadata, ImageEncoder
from .huggingface_models import (
    DINOV2_METADATA,
    SIGLIP2_METADATA,
    create_dinov2_vitb14,
    create_siglip2_b16,
)
from .registry import (
    create_encoder,
    encoder_metadata,
    get_registration,
    list_encoders,
    register_encoder,
    unregister_encoder,
)
from .resnet import RESNET50_METADATA, ResNet50Encoder, create_resnet50
from .semantic_models import (
    ADE20K_SEGFORMER_METADATA,
    COCO_DEEPLAB_METADATA,
    create_ade20k_segformer,
    create_coco_deeplab,
)
from .torchvision_models import (
    CONVNEXT_B_METADATA,
    INCEPTION_V3_METADATA,
    MOBILENET_V2_METADATA,
    VGG16_METADATA,
    VIT_B16_METADATA,
    create_convnext_b,
    create_inception_v3,
    create_mobilenet_v2,
    create_vgg16,
    create_vit_b16,
)

if "resnet50" not in list_encoders():
    register_encoder("resnet50", create_resnet50, RESNET50_METADATA)
for _name, _factory, _metadata in (
    ("dinov2_vitb14", create_dinov2_vitb14, DINOV2_METADATA),
    ("siglip2_b16", create_siglip2_b16, SIGLIP2_METADATA),
    ("convnext_b", create_convnext_b, CONVNEXT_B_METADATA),
    ("vit_b16", create_vit_b16, VIT_B16_METADATA),
    ("mobilenet_v2", create_mobilenet_v2, MOBILENET_V2_METADATA),
    ("vgg16", create_vgg16, VGG16_METADATA),
    ("inception_v3", create_inception_v3, INCEPTION_V3_METADATA),
    ("coco_deeplab", create_coco_deeplab, COCO_DEEPLAB_METADATA),
    ("ade20k_segformer", create_ade20k_segformer, ADE20K_SEGFORMER_METADATA),
):
    if _name not in list_encoders():
        register_encoder(_name, _factory, _metadata)


__all__ = [
    "ADE20K_SEGFORMER_METADATA",
    "COCO_DEEPLAB_METADATA",
    "CONVNEXT_B_METADATA",
    "DINOV2_METADATA",
    "INCEPTION_V3_METADATA",
    "MOBILENET_V2_METADATA",
    "RESNET50_METADATA",
    "SIGLIP2_METADATA",
    "VGG16_METADATA",
    "VIT_B16_METADATA",
    "EncoderMetadata",
    "ImageEncoder",
    "ResNet50Encoder",
    "create_encoder",
    "encoder_metadata",
    "get_registration",
    "list_encoders",
    "register_encoder",
    "unregister_encoder",
]

