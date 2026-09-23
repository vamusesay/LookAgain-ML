"""Installed-package runtime validation for real frozen image encoders.

Run this module on a target machine with ``python -m
lookagain_ml.runtime_validation``. It deliberately disables representation
caching so a PASS proves that the requested checkpoint performed inference in
the current process. The generated data are an infrastructure check, not
scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ._version import DISTRIBUTION_NAME, __version__
from .api import analyze


def _is_installed_package_path(path: Path) -> bool:
    """Return whether a module path is under site/dist-packages."""

    return any(part.lower() in {"site-packages", "dist-packages"} for part in path.parts)


def _dependency_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime_environment(module_path: Path) -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "package_version": __version__,
        "package_installed": _is_installed_package_path(module_path),
        "torch": str(torch.__version__),
        "torchvision": _dependency_version("torchvision"),
        "transformers": _dependency_version("transformers"),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_device_name": (
            str(torch.cuda.get_device_name(0)) if cuda_available else None
        ),
        "mps_available": bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ),
    }


def _make_generated_problem(directory: Path, n: int = 30):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260901)
    signal = np.linspace(-1.5, 1.5, n)
    paths: list[Path] = []
    for index, value in enumerate(signal):
        pixels = np.empty((40, 40, 3), dtype=np.uint8)
        pixels[..., 0] = np.clip(
            128 + 55 * value + rng.integers(-6, 7, size=(40, 40)), 0, 255
        )
        pixels[..., 1] = np.clip(
            115 - 35 * value + rng.integers(-6, 7, size=(40, 40)), 0, 255
        )
        pixels[..., 2] = 50 + index
        path = directory / f"runtime-{index:03d}.png"
        Image.fromarray(pixels, mode="RGB").save(path)
        paths.append(path)
    outcome = 1.7 * signal + rng.normal(scale=0.15, size=n)
    groups = [f"runtime-unit-{index:03d}" for index in range(n)]
    splits = ["train"] * 20 + ["validation"] * 5 + ["test"] * 5
    return paths, outcome, groups, splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real pretrained encoder through the installed LookAgain-ML "
            "public API and save machine-readable runtime evidence."
        )
    )
    parser.add_argument("--encoder", default="siglip2_b16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("runtime_validation_output"))
    parser.add_argument("--model-cache", type=Path, default=None)
    parser.add_argument(
        "--require-installed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail unless lookagain_ml imported from site-packages/dist-packages.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    module_path = Path(__file__).resolve()
    environment = _runtime_environment(module_path)
    if args.require_installed and not environment["package_installed"]:
        raise SystemExit(
            "Runtime validation requires an installed wheel, but lookagain_ml did not "
            "import from site-packages/dist-packages. Install the wheel in a clean "
            "environment and rerun, or use --no-require-installed only for development."
        )

    output = args.output.resolve()
    paths, outcome, groups, splits = _make_generated_problem(output / "generated_images")
    results = analyze(
        images=paths,
        y=outcome,
        groups=groups,
        split_labels=splits,
        task="regression",
        encoder=args.encoder,
        device=args.device,
        batch_size=args.batch_size,
        pretrained=True,
        cache=False,
        model_cache_dir=args.model_cache,
    )
    results.save(output / "results")
    encoder_result = results.encoder_results[args.encoder]
    costs = encoder_result["costs"]
    metadata = encoder_result["metadata"]
    actual_device = str(costs.get("device", "unknown"))
    if args.device.lower().startswith("cuda") and not actual_device.startswith("cuda"):
        raise RuntimeError(
            f"CUDA validation was requested, but the encoder reported device {actual_device!r}."
        )
    if float(costs.get("extraction_seconds", 0.0)) <= 0:
        raise RuntimeError("No positive extraction time was recorded; real inference was not proven.")

    report = {
        "status": "PASS",
        "purpose": "runtime infrastructure validation; not scientific performance evidence",
        "distribution": DISTRIBUTION_NAME,
        "environment": environment,
        "requested": {
            "encoder": args.encoder,
            "device": args.device,
            "batch_size": args.batch_size,
            "representation_cache": False,
        },
        "observed": {
            "device": actual_device,
            "checkpoint": metadata["checkpoint"],
            "checkpoint_revision": metadata["checkpoint_revision"],
            "preprocessing": metadata["preprocessing"],
            "pooling": metadata["pooling"],
            "feature_dimension": metadata["feature_dimension"],
            "extraction_seconds": costs["extraction_seconds"],
            "images_per_second": costs["images_per_second"],
            "peak_gpu_memory_bytes": costs.get("peak_gpu_memory_bytes"),
            "requested_n": results.audit["requested_n"],
            "successful_n": results.audit["successful_n"],
            "qc_status": results.qc["status"],
        },
    }
    report_path = output / "runtime_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved runtime validation to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
