from __future__ import annotations

from pathlib import Path

from lookagain_ml.runtime_validation import _is_installed_package_path, build_parser


def test_runtime_validator_defaults_to_real_siglip2_without_cache():
    args = build_parser().parse_args([])
    assert args.encoder == "siglip2_b16"
    assert args.device == "auto"
    assert args.batch_size == 2
    assert args.require_installed is True


def test_installed_package_path_detection_is_cross_platform():
    assert _is_installed_package_path(
        Path("env/lib/python3.12/site-packages/lookagain_ml/runtime_validation.py")
    )
    assert _is_installed_package_path(
        Path("venv/Lib/site-packages/lookagain_ml/runtime_validation.py")
    )
    assert _is_installed_package_path(
        Path("env/python/dist-packages/lookagain_ml/runtime_validation.py")
    )
    assert not _is_installed_package_path(Path("project/src/lookagain_ml/runtime_validation.py"))
