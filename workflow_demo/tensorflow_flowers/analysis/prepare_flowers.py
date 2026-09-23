"""Download, safely stage, audit, and split TensorFlow Flowers for stable validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import shutil
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from PIL import Image

FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
ARCHIVE_BYTES = 228_813_984
ARCHIVE_SHA256 = "4C54ACE7911AAFFE13A365C34F650E71DD5BF1BE0A58B464E5A7183E3E595D9C"
ANALYSIS_MANIFEST_SHA256 = "67FF21B54ED7D2106D33F9B96725374E8F940E477E4452B15F66248198D180C7"
LOCKED_SPLIT_SHA256 = "668D15E9535CE943051F86300EB8C9704AFC59252E14E427244E73A2CAF5E97B"
CONFLICT_DIGEST = "6784C09FF4EEEC0E0AC148CDC74682F99C424AC89D85321D95161E20DA9204E9"
CLASS_TO_INDEX = {"daisy": 0, "dandelion": 1, "roses": 2, "sunflowers": 3, "tulips": 4}
CLASS_COUNTS = {"daisy": 633, "dandelion": 898, "roses": 641, "sunflowers": 699, "tulips": 799}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def obtain_archive(local: Path, durable: Path | None, source: Path | None) -> str:
    """Obtain one verified local archive; first runs download from the official URL."""

    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        raise FileExistsError(f"Use a clean local working root; archive already exists: {local}")
    if source is not None:
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, local)
        provenance = "verified_durable_archive_copied_to_local_session"
    else:
        temporary = local.with_suffix(".partial")
        request = Request(FLOWERS_URL, headers={"User-Agent": "LookAgain-ML/1.0.0 validation"})
        with urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(local)
        provenance = "fresh_official_download"
    if local.stat().st_size != ARCHIVE_BYTES or sha256_file(local) != ARCHIVE_SHA256:
        raise RuntimeError("TensorFlow Flowers archive size or SHA-256 is not authoritative.")
    if durable is not None:
        durable.parent.mkdir(parents=True, exist_ok=True)
        if durable.exists():
            if durable.stat().st_size != ARCHIVE_BYTES or sha256_file(durable) != ARCHIVE_SHA256:
                raise RuntimeError("Existing durable source archive conflicts with the authoritative archive.")
        else:
            partial = durable.with_suffix(".partial")
            shutil.copy2(local, partial)
            if sha256_file(partial) != ARCHIVE_SHA256:
                raise RuntimeError("Durable archive copy verification failed.")
            partial.replace(durable)
    return provenance


def safe_extract(archive_path: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"Use a clean local staging root: {destination}")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or not pure.parts
                or pure.parts[0] != "flower_photos"
                or member.issym()
                or member.islnk()
                or member.isdev()
                or member.isfifo()
            ):
                raise RuntimeError(f"Unsafe or unexpected archive member: {member.name!r}")
        destination.mkdir(parents=True)
        # Python 3.12 added the extraction filter argument.  The package also
        # supports Python 3.10--3.11, where the explicit member validation
        # above supplies the traversal/link/device safeguards before extraction.
        if "filter" in inspect.signature(archive.extractall).parameters:
            archive.extractall(destination, filter="data")
        else:
            archive.extractall(destination)
    root = destination / "flower_photos"
    if not (root / "LICENSE.txt").is_file():
        raise RuntimeError("TensorFlow Flowers LICENSE.txt is missing.")
    return root


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_locked_split(frame: pd.DataFrame, path: Path) -> None:
    """Write the byte-locked split with platform-independent LF endings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def audit_images(image_root: Path, manifest_root: Path) -> tuple[Path, dict[str, object]]:
    paths = sorted(
        path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if len(paths) != 3_670:
        raise RuntimeError(f"Expected 3,670 JPEGs; found {len(paths):,}.")
    records: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(image_root).as_posix()
        parts = PurePosixPath(relative).parts
        if len(parts) != 2 or parts[0] not in CLASS_TO_INDEX:
            raise RuntimeError(f"Unexpected image location: {relative}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode, image_format = image.mode, image.format
        records.append(
            {
                "observation_id": f"tf_flowers_{index:05d}",
                "relative_path": relative,
                "class_label": parts[0],
                "source_split": "",
                "file_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
            }
        )
        if index % 200 == 0 or index == len(paths):
            print(f"Checking and content-hashing images: {index:,}/{len(paths):,}", flush=True)
    if dict(Counter(str(item["class_label"]) for item in records)) != CLASS_COUNTS:
        raise RuntimeError("TensorFlow Flowers class counts changed.")
    if any(item["format"] != "JPEG" for item in records):
        raise RuntimeError("At least one source image is not JPEG encoded.")

    analysis_rows: list[dict[str, object]] = []
    excluded = 0
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in records:
        groups[str(item["sha256"])].append(item)
        if item["sha256"] == CONFLICT_DIGEST:
            excluded += 1
            continue
        analysis_rows.append(
            {
                "observation_id": item["observation_id"],
                "relative_image_path": item["relative_path"],
                "class_label": item["class_label"],
                "class_index": CLASS_TO_INDEX[str(item["class_label"])],
                "duplicate_group_id": f"sha256:{item['sha256']}",
                "image_sha256": item["sha256"],
            }
        )
    cross_class = [
        digest for digest, items in groups.items() if len({str(item["class_label"]) for item in items}) > 1
    ]
    if cross_class != [CONFLICT_DIGEST] or excluded != 2 or len(analysis_rows) != 3_668:
        raise RuntimeError("Duplicate/exclusion evidence differs from the locked public protocol.")
    manifest = manifest_root / "tf_flowers_analysis_manifest.csv"
    fields = [
        "observation_id",
        "relative_image_path",
        "class_label",
        "class_index",
        "duplicate_group_id",
        "image_sha256",
    ]
    write_csv(manifest, analysis_rows, fields)
    if sha256_file(manifest) != ANALYSIS_MANIFEST_SHA256:
        raise RuntimeError("Rebuilt analysis manifest does not match the locked SHA-256.")
    return manifest, {
        "source_images": len(records),
        "analysis_rows": len(analysis_rows),
        "excluded_cross_class_duplicate_rows": excluded,
        "exact_duplicate_groups": sum(len(items) > 1 for items in groups.values()),
        "unique_analysis_groups": len({str(row["duplicate_group_id"]) for row in analysis_rows}),
    }


def make_locked_split(manifest: Path, output: Path) -> dict[str, object]:
    from lookagain_ml.splitting import make_split

    frame = pd.read_csv(manifest)
    y = frame["class_label"].astype(str).to_numpy()
    groups = frame["duplicate_group_id"].astype(str).to_numpy()
    hashes = frame["image_sha256"].astype(str).to_numpy()
    generated = make_split(
        y,
        task="classification",
        groups=groups,
        image_hashes=hashes,
        random_state=20260818,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
    )
    labels = np.asarray(generated.labels, dtype=str)
    effective = np.asarray(generated.effective_groups, dtype=str)
    checked = make_split(
        y,
        task="classification",
        groups=groups,
        image_hashes=hashes,
        split_labels=labels,
        random_state=20260818,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
    )
    if not np.array_equal(labels, np.asarray(checked.labels, dtype=str)):
        raise RuntimeError("Supplied split round trip changed assignments.")
    locked = frame.copy()
    locked["split"] = labels
    locked["effective_group_id"] = effective
    if int(locked.groupby("effective_group_id")["split"].nunique().max()) != 1:
        raise RuntimeError("A duplicate group crosses partitions.")
    write_locked_split(locked, output)
    if sha256_file(output) != LOCKED_SPLIT_SHA256:
        raise RuntimeError("Rebuilt split does not match the locked SHA-256.")
    counts = locked["split"].value_counts().to_dict()
    expected = {"train": 2568, "validation": 550, "test": 550}
    if counts != expected:
        raise RuntimeError(f"Locked split counts changed: {counts}")
    return {"counts": expected, "effective_groups": int(locked["effective_group_id"].nunique())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-root", type=Path, required=True)
    parser.add_argument("--durable-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Verified durable archive for a reconnect session.")
    parser.add_argument(
        "--record-output",
        type=Path,
        help="Optional non-overwriting preparation record path; defaults to the primary record.",
    )
    args = parser.parse_args()
    working = args.working_root.expanduser().resolve()
    durable = args.durable_root.expanduser().resolve()
    if working.exists() and any(working.iterdir()):
        raise FileExistsError(f"Use a clean local working root: {working}")
    working.mkdir(parents=True, exist_ok=True)
    local_archive = working / "downloads" / "flower_photos.tgz"
    durable_archive = durable / "public_source" / "flower_photos.tgz"
    provenance = obtain_archive(
        local_archive,
        durable_archive,
        args.archive.expanduser().resolve() if args.archive else None,
    )
    image_root = safe_extract(local_archive, working / "staged")
    manifest, audit = audit_images(image_root, working / "manifests")
    split_path = working / "splits" / "tf_flowers_locked_split_v1.csv"
    split = make_locked_split(manifest, split_path)
    summary = {
        "status": "PASS",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": "1.0.0",
        "dataset": "TensorFlow Flowers",
        "official_url": FLOWERS_URL,
        "archive_provenance": provenance,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "analysis_manifest_sha256": sha256_file(manifest),
        "locked_split_sha256": sha256_file(split_path),
        "audit": audit,
        "split": split,
        "local_staging": True,
        "images_read_from_mounted_drive_during_analysis": False,
    }
    record_output = (
        args.record_output.expanduser().resolve()
        if args.record_output
        else durable / "records" / "tf_flowers_preparation_summary.json"
    )
    if args.record_output and record_output.exists():
        raise FileExistsError(f"Preserve prior preparation evidence: {record_output}")
    atomic_json(record_output, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
