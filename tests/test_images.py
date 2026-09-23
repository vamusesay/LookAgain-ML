from pathlib import Path

import pytest

from lookagain_ml.exceptions import InputValidationError
from lookagain_ml.images import build_image_manifest


def test_manifest_hashes_duplicates_and_grayscale(image_factory):
    first = image_factory("first.png", (120, 0, 0))
    duplicate = image_factory("duplicate.png", (120, 0, 0))
    gray = image_factory("gray.png", (80, 0, 0), mode="L")
    result = build_image_manifest(
        [first, duplicate, gray], [1.0, 2.0, 3.0], verify_images=True
    )
    assert result.audit.successful_n == 3
    assert result.audit.failed_n == 0
    assert result.manifest.is_exact_duplicate.tolist() == [True, True, False]
    assert result.manifest.loc[0, "image_sha256"] == result.manifest.loc[1, "image_sha256"]


def test_strict_mode_reports_every_failure(image_factory, tmp_path):
    valid = image_factory("valid.png", (1, 2, 3))
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_text("not an image", encoding="utf-8")
    missing = tmp_path / "missing.png"
    with pytest.raises(InputValidationError) as captured:
        build_image_manifest([valid, corrupt, missing], [1, 2, 3], strict=True)
    assert captured.value.audit["requested_n"] == 3
    assert captured.value.audit["failed_n"] == 2
    reasons = " ".join(item["reason"] for item in captured.value.audit["failures"])
    assert "could not be read" in reasons
    assert "does not exist" in reasons


def test_non_strict_mode_keeps_audited_valid_rows(image_factory, tmp_path):
    valid = image_factory("valid.png", (1, 2, 3))
    result = build_image_manifest(
        [valid, tmp_path / "missing.png"], [1, 2], strict=False
    )
    assert len(result.manifest) == 1
    assert result.audit.failed_n == 1
    assert Path(result.manifest.loc[0, "image_path"]).is_absolute()


def test_missing_group_is_rejected(image_factory):
    image = image_factory("valid.png", (1, 2, 3))
    with pytest.raises(InputValidationError, match="missing value"):
        build_image_manifest([image], [1], groups=[None])


@pytest.mark.parametrize(
    "suffix",
    [
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
        ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP",
    ],
)
def test_supported_common_image_formats_are_validated(image_factory, suffix):
    image = image_factory(f"format{suffix}", (20, 40, 60))
    result = build_image_manifest([image], [1.0])
    assert result.audit.successful_n == 1
    assert result.manifest.loc[0, "image_format"] is not None


def test_unsupported_extension_is_rejected_without_claiming_universal_support(
    image_factory, tmp_path
):
    image = image_factory("source.png", (20, 40, 60))
    unsupported = tmp_path / "renamed.gif"
    unsupported.write_bytes(image.read_bytes())
    with pytest.raises(InputValidationError, match="unsupported image extension"):
        build_image_manifest([unsupported], [1.0])

