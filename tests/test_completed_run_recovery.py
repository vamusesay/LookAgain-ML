from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import shutil
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lookagain_ml import (
    AdvancedConfig,
    ExecutionConfig,
    ProgressReporter,
    analyze,
    api,
    checkpointing,
)
from lookagain_ml import images as image_module
from lookagain_ml import result_checkpoint as result_checkpoint_module
from lookagain_ml.checkpointing import CheckpointStore
from lookagain_ml.encoders import (
    EncoderMetadata,
    ImageEncoder,
    get_registration,
    register_encoder,
)
from lookagain_ml.exceptions import CacheValidationError, CheckpointValidationError
from lookagain_ml.protocols import configuration_sha256
from lookagain_ml.result_checkpoint import load_result_checkpoint


class IntentionalStop(RuntimeError):
    pass


def _metadata(name: str) -> EncoderMetadata:
    return EncoderMetadata(
        name=name,
        display_name=f"Recovery fixture {name}",
        checkpoint=f"generated-{name}",
        checkpoint_revision="fixture-v1",
        feature_dimension=6,
        input_size=(12, 12),
        preprocessing="deterministic RGB statistics",
        pooling="channel mean and standard deviation",
        backend="Pillow/NumPy test fixture",
    )


@pytest.fixture
def recovery_encoders():
    names = ("resnet50", "vgg16")
    originals = {name: get_registration(name) for name in names}
    calls = {name: 0 for name in names}

    class FixtureEncoder(ImageEncoder):
        def __init__(self, name: str, **kwargs) -> None:
            del kwargs
            self.name = name
            self.metadata = _metadata(name)
            self.costs = {"device": "cpu", "feature_dimension": 6}

        def encode(self, paths, *, batch_size: int = 32):
            calls[self.name] += 1
            rows = []
            offset = 0.01 if self.name == "vgg16" else 0.0
            for start in range(0, len(paths), batch_size):
                for path in paths[start : start + batch_size]:
                    with Image.open(path) as image:
                        values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                    rows.append(
                        np.concatenate(
                            [values.mean(axis=(0, 1)) + offset, values.std(axis=(0, 1))]
                        )
                    )
                self._report_batch_progress(min(start + batch_size, len(paths)), len(paths))
            return np.asarray(rows, dtype=np.float32)

    for name in names:
        register_encoder(
            name,
            lambda name=name, **kwargs: FixtureEncoder(name, **kwargs),
            _metadata(name),
            replace=True,
        )
    try:
        yield calls
    finally:
        for registration in originals.values():
            register_encoder(
                registration.name,
                registration.factory,
                registration.metadata,
                replace=True,
            )


def _inputs(image_factory, n: int = 24):
    images = [
        image_factory(
            f"recovery-{index}.png",
            ((index * 19) % 255, (index * 37) % 255, (index * 53) % 255),
        )
        for index in range(n)
    ]
    outcome = np.linspace(-2.0, 2.0, n)
    groups = [f"group-{index}" for index in range(n)]
    split = ["train"] * 14 + ["validation"] * 5 + ["test"] * 5
    return images, outcome, groups, split


def _run(
    image_factory,
    checkpoint: Path,
    *,
    callback=None,
    policy="reuse",
    encoders=("resnet50", "vgg16"),
):
    images, outcome, groups, split = _inputs(image_factory)
    return analyze(
        images,
        outcome,
        groups,
        split_labels=split,
        task="regression",
        encoders=encoders,
        cache=False,
        batch_size=5,
        combinations=("linear_stack",),
        bootstrap_repetitions=100,
        advanced_config=AdvancedConfig(
            execution=ExecutionConfig(
                progress=False,
                progress_callback=callback,
                checkpoint_dir=checkpoint,
                run_id="stable-recovery-fixture",
                completed_run_policy=policy,
                persistent_event_interval_seconds=3600.0,
            )
        ),
    )


def _same_result(left, right) -> None:
    assert left.to_dict() == right.to_dict()
    assert left.manifest.equals(right.manifest)
    assert left.predictions.equals(right.predictions)
    assert set(left.representation_features or {}) == set(right.representation_features or {})
    for name in left.representation_features or {}:
        np.testing.assert_array_equal(
            left.representation_features[name], right.representation_features[name]
        )
    for name in left.method_predictions or {}:
        np.testing.assert_array_equal(left.method_predictions[name], right.method_predictions[name])
    probe = left.representation_features[left.selected_encoder]
    np.testing.assert_array_equal(left.fitted_head.predict(probe), right.fitted_head.predict(probe))


def test_completed_run_is_reconstructed_without_any_scientific_computation(
    image_factory, recovery_encoders, tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "completed"
    first = _run(image_factory, checkpoint)
    calls_after_first = dict(recovery_encoders)
    manifest_path = checkpoint / "run_manifest.json"
    manifest_before = manifest_path.read_bytes()
    events = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("completed reuse invoked scientific computation")

    for name in (
        "create_encoder",
        "fit_linear_head",
        "fit_validation_combinations",
        "fit_feature_concatenation",
        "fit_covariate_analysis",
        "paired_locked_test_bootstrap",
    ):
        monkeypatch.setattr(api, name, forbidden)
    second = _run(image_factory, checkpoint, callback=events.append)
    _same_result(first, second)
    assert recovery_encoders == calls_after_first
    assert manifest_path.read_bytes() == manifest_before
    assert any(event.event == "REUSED" and event.stage == "result_loading" for event in events)
    assert not any(event.stage in {"encoder", "linear_head", "image_combination"} for event in events)
    index = json.loads((checkpoint / "run_index.json").read_text(encoding="utf-8"))
    assert [item["status"] for item in index["attempts"]][-2:] == ["COMPLETED", "COMPLETED"]
    assert index["attempts"][-1]["result_disposition"] == "returned_verified_completed_run"
    assert not (checkpoint / "run.lock").exists()


def test_non_alphabetical_encoder_order_reconstructs_exactly(
    image_factory, recovery_encoders, tmp_path
) -> None:
    checkpoint = tmp_path / "non-alphabetical"
    order = ("vgg16", "resnet50")
    first = _run(image_factory, checkpoint, encoders=order)
    calls_after_first = dict(recovery_encoders)
    second = _run(image_factory, checkpoint, encoders=order)
    _same_result(first, second)
    assert list(first.encoder_results) == list(order)
    assert set(second.encoder_results) == set(order)
    assert recovery_encoders == calls_after_first


def test_explicit_recompute_uses_new_attempt_and_preserves_completed_manifest(
    image_factory, recovery_encoders, tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "recompute"
    _run(image_factory, checkpoint)
    manifest_before = (checkpoint / "run_manifest.json").read_bytes()
    fit_calls = {"count": 0}
    actual = api.fit_linear_head

    def counted(*args, **kwargs):
        fit_calls["count"] += 1
        return actual(*args, **kwargs)

    monkeypatch.setattr(api, "fit_linear_head", counted)
    _run(image_factory, checkpoint, policy="recompute")
    assert fit_calls["count"] == 2
    assert recovery_encoders == {"resnet50": 1, "vgg16": 1}
    assert (checkpoint / "run_manifest.json").read_bytes() == manifest_before


@pytest.mark.parametrize("damage", ["truncate", "empty", "remove", "checksum"])
def test_corrupt_or_missing_completed_components_are_rejected(
    image_factory, recovery_encoders, tmp_path, damage
) -> None:
    checkpoint = tmp_path / f"corrupt-{damage}"
    _run(image_factory, checkpoint)
    root_manifest = json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))
    result_root = checkpoint / root_manifest["result_checkpoint"]["path"]
    component = result_root / "public_payload.json"
    if damage == "truncate":
        component.write_text("{", encoding="utf-8")
    elif damage == "empty":
        component.write_bytes(b"")
    elif damage == "remove":
        component.unlink()
    else:
        component.write_bytes(component.read_bytes() + b" ")
    with pytest.raises(CheckpointValidationError, match="component|checksum|size|inventory"):
        _run(image_factory, checkpoint)
    assert recovery_encoders == {"resnet50": 1, "vgg16": 1}
    assert root_manifest == json.loads(
        (checkpoint / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert not (checkpoint / "run.lock").exists()


def test_interruption_after_atomic_promotion_recovers_without_refit(
    image_factory, recovery_encoders, tmp_path
) -> None:
    checkpoint = tmp_path / "promoted"

    def stop_after_promotion(event) -> None:
        if event.event == "CHECKPOINTED" and event.stage == "result_checkpoint":
            raise IntentionalStop("disconnect after atomic result promotion")

    with pytest.raises(IntentionalStop):
        _run(image_factory, checkpoint, callback=stop_after_promotion)
    assert list(checkpoint.glob("attempts/*/result_checkpoint/result_checkpoint.json"))
    calls = dict(recovery_encoders)
    restored = _run(image_factory, checkpoint)
    assert restored.qc["test_outcomes_used_for_selection"] is False
    assert recovery_encoders == calls
    assert not list(checkpoint.rglob("*.partial"))


def test_legacy_completed_manifest_without_result_refits_only_downstream(
    image_factory, recovery_encoders, tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "legacy"
    _run(image_factory, checkpoint)
    legacy = json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))
    legacy["schema_version"] = 1
    legacy.pop("attempt_id", None)
    legacy.pop("result_checkpoint", None)
    legacy.pop("result_disposition", None)
    (checkpoint / "run_manifest.json").write_text(
        json.dumps(legacy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (checkpoint / "run_index.json").unlink()
    shutil.rmtree(checkpoint / "attempts")
    fit_calls = {"count": 0}
    actual = api.fit_linear_head

    def counted(*args, **kwargs):
        fit_calls["count"] += 1
        return actual(*args, **kwargs)

    monkeypatch.setattr(api, "fit_linear_head", counted)
    _run(image_factory, checkpoint)
    assert recovery_encoders == {"resnet50": 1, "vgg16": 1}
    assert fit_calls["count"] == 2
    assert json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8")) == legacy
    assert list(checkpoint.glob("attempts/*/result_checkpoint/result_checkpoint.json"))


def test_input_and_final_serialization_interruptions_release_resources(
    image_factory, recovery_encoders, tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "input-stop"

    def stop_input(event) -> None:
        if event.event == "VERIFYING" and event.stage == "input_validation" and event.current == 2:
            raise IntentionalStop("input verification interrupted")

    with pytest.raises(IntentionalStop):
        _run(image_factory, checkpoint, callback=stop_input)
    assert not checkpoint.exists()

    def stop_serialization(*args, **kwargs):
        del args, kwargs
        raise IntentionalStop("result serialization interrupted")

    monkeypatch.setattr(api, "write_result_checkpoint", stop_serialization)
    checkpoint = tmp_path / "serialize-stop"
    with pytest.raises(IntentionalStop):
        _run(image_factory, checkpoint)
    assert not (checkpoint / "run.lock").exists()
    assert json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))[
        "status"
    ] == "FAILED"


def test_result_schema_version_and_cache_corruption_fail_closed(
    image_factory, recovery_encoders, tmp_path
) -> None:
    checkpoint = tmp_path / "schema"
    _run(image_factory, checkpoint)
    manifest = json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))
    result_root = checkpoint / manifest["result_checkpoint"]["path"]
    metadata_path = result_root / "result_checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_schema_version"] = "999"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(CheckpointValidationError, match="checksum"):
        _run(image_factory, checkpoint)

    checkpoint = tmp_path / "cache-corrupt"
    _run(image_factory, checkpoint)
    array_path = next((checkpoint / "encoder_cache" / "embeddings" / "resnet50").glob("*.npy"))
    array_path.write_bytes(array_path.read_bytes()[:-8])
    with pytest.raises(
        (CheckpointValidationError, CacheValidationError),
        match="cache|array|unreadable|checksum",
    ):
        _run(image_factory, checkpoint)


def test_event_persistence_is_bounded_and_progress_disabled_is_silent(tmp_path) -> None:
    store = CheckpointStore(
        tmp_path / "events",
        identity={"configuration_sha256": "a" * 64, "planned_units": []},
        persistent_event_interval_seconds=3600.0,
    )
    stream = io.StringIO()
    reporter = ProgressReporter(False, stream=stream, log_callback=store.record_event)
    reporter.emit("STARTED", "encoder", "start")
    for current in range(10_001):
        reporter.emit(
            "ENCODING",
            "image_encoding",
            "batch",
            encoder="resnet50",
            current=current,
            total=10_000,
        )
    reporter.emit("COMPLETED", "encoder", "done")
    store.close()
    assert stream.getvalue() == ""
    lines = (tmp_path / "events" / "execution_events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) <= 6


def test_concurrent_writer_and_conservative_stale_lock(tmp_path, monkeypatch) -> None:
    identity = {"configuration_sha256": "b" * 64, "planned_units": []}
    first = CheckpointStore(tmp_path / "concurrent", identity=identity)
    with pytest.raises(CheckpointValidationError, match="Another process"):
        CheckpointStore(tmp_path / "concurrent", identity=identity)
    first.close()
    second = CheckpointStore(tmp_path / "concurrent", identity=identity)
    second.close()

    stale_root = tmp_path / "stale"
    stale_root.mkdir()
    lock = stale_root / "run.lock"
    lock.write_text(
        json.dumps(
            {
                "attempt_id": "abandoned",
                "pid": 99999999,
                "hostname": socket.gethostname(),
            }
        ),
        encoding="utf-8",
    )
    old = time.time() - 120
    os.utime(lock, (old, old))
    monkeypatch.setattr(
        checkpointing._RunLock, "_process_alive", staticmethod(lambda pid: False)
    )
    recovered = CheckpointStore(
        stale_root, identity=identity, stale_lock_timeout_seconds=60.0
    )
    recovered.close()
    assert list((stale_root / "stale_locks").glob("*.lock.json"))


def test_two_analyze_invocations_cannot_share_one_active_run(
    image_factory, recovery_encoders, tmp_path
) -> None:
    checkpoint = tmp_path / "concurrent-analyze"
    images, outcome, groups, split = _inputs(image_factory)
    first_has_lock = threading.Event()
    release_first = threading.Event()

    def execute(callback=None):
        return analyze(
            images,
            outcome,
            groups,
            split_labels=split,
            task="regression",
            encoder="resnet50",
            cache=False,
            advanced_config=AdvancedConfig(
                execution=ExecutionConfig(
                    progress=False,
                    progress_callback=callback,
                    checkpoint_dir=checkpoint,
                    run_id="concurrent-invocation-fixture",
                )
            ),
        )

    def hold_after_lock(event) -> None:
        if event.event == "STARTED" and event.stage == "analysis":
            first_has_lock.set()
            if not release_first.wait(timeout=10.0):
                raise TimeoutError("concurrency fixture was not released")

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(execute, hold_after_lock)
        assert first_has_lock.wait(timeout=10.0)
        try:
            with pytest.raises(CheckpointValidationError, match="Another process"):
                execute()
        finally:
            release_first.set()
        first.result(timeout=30.0)
    assert not (checkpoint / "run.lock").exists()


def test_result_promotion_retry_is_bounded_and_permanent_errors_propagate(
    tmp_path, monkeypatch
) -> None:
    actual_replace = result_checkpoint_module.os.replace
    monkeypatch.setattr(result_checkpoint_module.time, "sleep", lambda _seconds: None)
    calls = {"count": 0}

    def transient_then_succeed(source, destination):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError(errno.EACCES, "temporary sync lock")
        return actual_replace(source, destination)

    source = tmp_path / "transient-source"
    source.mkdir()
    monkeypatch.setattr(result_checkpoint_module.os, "replace", transient_then_succeed)
    result_checkpoint_module._replace_with_retry(source, tmp_path / "transient-result")
    assert calls["count"] == 3

    calls["count"] = 0

    def permanent_error(source, destination):
        del source, destination
        calls["count"] += 1
        raise OSError(errno.EINVAL, "permanent invalid operation")

    source = tmp_path / "permanent-source"
    source.mkdir()
    monkeypatch.setattr(result_checkpoint_module.os, "replace", permanent_error)
    with pytest.raises(OSError, match="permanent invalid operation"):
        result_checkpoint_module._replace_with_retry(source, tmp_path / "permanent-result")
    assert calls["count"] == 1

    calls["count"] = 0

    def always_transient(source, destination):
        del source, destination
        calls["count"] += 1
        raise PermissionError(errno.EACCES, "persistent sync lock")

    source = tmp_path / "exhausted-source"
    source.mkdir()
    monkeypatch.setattr(result_checkpoint_module.os, "replace", always_transient)
    with pytest.raises(PermissionError, match="persistent sync lock"):
        result_checkpoint_module._replace_with_retry(source, tmp_path / "exhausted-result")
    assert calls["count"] == 12


def test_resource_provenance_is_not_compatibility_identity_but_environment_is(
    tmp_path,
) -> None:
    base = {
        "package_version": "1.0.0.dev3",
        "configuration_sha256": "d" * 64,
        "protocol_identifier": "resource-identity-fixture",
        "environment": {
            "python": "3.12.0",
            "operating_system": "Windows",
            "resources": {"logical_cpu_count": 8, "rf_n_jobs": None},
        },
        "planned_units": [],
    }
    root = tmp_path / "normalized"
    first = CheckpointStore(root, identity=base, run_id="resource-identity")
    first.close()
    changed_resources = {
        **base,
        "environment": {
            **base["environment"],
            "resources": {"logical_cpu_count": 16, "rf_n_jobs": -1},
        },
    }
    second = CheckpointStore(root, identity=changed_resources)
    assert second.manifest["environment"]["resources"]["logical_cpu_count"] == 16
    second.close()
    changed_environment = {
        **changed_resources,
        "environment": {**changed_resources["environment"], "python": "3.13.0"},
    }
    with pytest.raises(CheckpointValidationError, match="incompatible"):
        CheckpointStore(root, identity=changed_environment)

    legacy_root = tmp_path / "legacy-full-hash"
    legacy = CheckpointStore(legacy_root, identity=base, run_id="legacy-resource-identity")
    legacy.close()
    historical_hash = configuration_sha256(base)
    for path in (legacy_root / "run_manifest.json", legacy_root / "run_index.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["identity_sha256"] = historical_hash
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    accepted = CheckpointStore(legacy_root, identity=changed_resources)
    assert accepted.identity_sha256 == historical_hash
    accepted.close()


def test_unsupported_result_schema_is_rejected_without_executable_loading(
    image_factory, recovery_encoders, tmp_path
) -> None:
    checkpoint = tmp_path / "unsupported"
    _run(image_factory, checkpoint)
    manifest = json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))
    result_root = checkpoint / manifest["result_checkpoint"]["path"]
    metadata_path = result_root / "result_checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_schema_version"] = "unsupported"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(CheckpointValidationError, match="checkpoint_schema_version"):
        load_result_checkpoint(
            result_root,
            identity_sha256=manifest["identity_sha256"],
            protocol_identifier=manifest["protocol_identifier"],
            protocol_fingerprint=manifest["protocol_fingerprint"],
            configuration_sha256=manifest["configuration_sha256"],
            row_count=manifest["identity"]["row_count"],
            representation_features=None,
        )


@pytest.mark.parametrize(
    "change",
    ["observation_order", "outcome", "group", "image", "covariate", "split", "config", "protocol"],
)
def test_completed_reuse_rejects_every_changed_scientific_identity(
    image_factory, recovery_encoders, tmp_path, change
) -> None:
    checkpoint = tmp_path / f"identity-{change}"
    images, outcome, groups, split = _inputs(image_factory)
    observation_ids = [f"row-{index}" for index in range(len(images))]
    covariates = np.column_stack(
        [np.linspace(0.0, 1.0, len(images)), np.arange(len(images)) % 3]
    )

    def execute(*, ridge=(0.1, 1.0), protocol="fixture-v1"):
        return analyze(
            images,
            outcome,
            groups,
            observation_ids=observation_ids,
            covariates=covariates,
            split_labels=split,
            task="regression",
            encoder="resnet50",
            cache=False,
            ridge_alphas=ridge,
            advanced_config=AdvancedConfig(
                execution=ExecutionConfig(
                    progress=False,
                    checkpoint_dir=checkpoint,
                    run_id="identity-fixture",
                    protocol_identifier=protocol,
                    protocol_fingerprint="f" * 64,
                )
            ),
        )

    execute()
    manifest_before = (checkpoint / "run_manifest.json").read_bytes()
    ridge = (0.1, 1.0)
    protocol = "fixture-v1"
    if change == "observation_order":
        observation_ids[-2:] = reversed(observation_ids[-2:])
    elif change == "outcome":
        outcome[-1] += 1.0
    elif change == "group":
        groups[-1] = "changed-group"
    elif change == "image":
        Image.new("RGB", (20, 20), (1, 2, 3)).save(images[-1])
    elif change == "covariate":
        covariates[-1, 0] += 1.0
    elif change == "split":
        split[-1], split[-6] = split[-6], split[-1]
    elif change == "config":
        ridge = (0.2, 2.0)
    elif change == "protocol":
        protocol = "fixture-v2"
    with pytest.raises(CheckpointValidationError, match="incompatible"):
        execute(ridge=ridge, protocol=protocol)
    assert (checkpoint / "run_manifest.json").read_bytes() == manifest_before
    assert not (checkpoint / "run.lock").exists()


def test_fitting_and_bootstrap_failures_release_lock_and_resume(
    image_factory, recovery_encoders, tmp_path, monkeypatch
) -> None:
    checkpoint = tmp_path / "fit-failure"
    actual_fit = api.fit_linear_head

    def fail_fit(*args, **kwargs):
        del args, kwargs
        raise IntentionalStop("downstream fit interrupted")

    monkeypatch.setattr(api, "fit_linear_head", fail_fit)
    with pytest.raises(IntentionalStop, match="downstream fit"):
        _run(image_factory, checkpoint)
    assert not (checkpoint / "run.lock").exists()
    monkeypatch.setattr(api, "fit_linear_head", actual_fit)
    completed = _run(image_factory, checkpoint)
    calls_after_fit_resume = dict(recovery_encoders)

    checkpoint = tmp_path / "bootstrap-failure"
    actual_bootstrap = api.paired_locked_test_bootstrap

    def fail_bootstrap(*args, **kwargs):
        del args, kwargs
        raise IntentionalStop("bootstrap interrupted")

    monkeypatch.setattr(api, "paired_locked_test_bootstrap", fail_bootstrap)
    with pytest.raises(IntentionalStop, match="bootstrap"):
        _run(image_factory, checkpoint)
    assert not (checkpoint / "run.lock").exists()
    monkeypatch.setattr(api, "paired_locked_test_bootstrap", actual_bootstrap)
    resumed = _run(image_factory, checkpoint)
    assert recovery_encoders == {
        name: calls_after_fit_resume[name] + 1 for name in calls_after_fit_resume
    }
    assert completed.qc["test_outcomes_used_for_selection"] is False
    assert resumed.qc["test_outcomes_used_for_selection"] is False


def test_slow_input_verification_remains_visible_and_has_no_cuda_requirement(
    image_factory, recovery_encoders, tmp_path, monkeypatch
) -> None:
    actual_hash = image_module.sha256_file

    def slow_hash(path):
        time.sleep(0.001)
        return actual_hash(path)

    monkeypatch.setattr(image_module, "sha256_file", slow_hash)
    events = []
    result = _run(image_factory, tmp_path / "slow input with spaces", callback=events.append)
    verifying = [
        event
        for event in events
        if event.event == "VERIFYING" and event.stage == "input_validation"
    ]
    assert verifying[-1].current == verifying[-1].total == 24
    assert result.costs["device"] == "cpu"


def test_package_version_is_part_of_checkpoint_identity(tmp_path) -> None:
    root = tmp_path / "version"
    base = {
        "package_version": "1.0.0.dev3",
        "configuration_sha256": "c" * 64,
        "planned_units": [],
    }
    store = CheckpointStore(root, identity=base)
    store.close()
    with pytest.raises(CheckpointValidationError, match="incompatible"):
        CheckpointStore(root, identity={**base, "package_version": "1.0.0.dev4"})


def test_classification_and_structured_preprocessor_reconstruct_exactly(
    image_factory, recovery_encoders, tmp_path
) -> None:
    images, _, groups, split = _inputs(image_factory, n=30)
    split = ["train"] * 18 + ["validation"] * 6 + ["test"] * 6
    labels = np.asarray(["daisy", "rose", "tulip"] * 10, dtype=object)
    covariates = np.column_stack(
        [np.linspace(0.0, 1.0, len(images)), np.arange(len(images)) % 2]
    )
    checkpoint = tmp_path / "classification"

    def execute():
        return analyze(
            images,
            labels,
            groups,
            covariates=covariates,
            split_labels=split,
            task="classification",
            encoder="resnet50",
            cache=False,
            advanced_config=AdvancedConfig(
                execution=ExecutionConfig(
                    progress=False,
                    checkpoint_dir=checkpoint,
                    run_id="classification-recovery",
                )
            ),
        )

    first = execute()
    calls = dict(recovery_encoders)
    second = execute()
    _same_result(first, second)
    assert recovery_encoders == calls
    np.testing.assert_array_equal(
        first.fitted_covariate_preprocessor.transform(covariates),
        second.fitted_covariate_preprocessor.transform(covariates),
    )


