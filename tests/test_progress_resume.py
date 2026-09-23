from __future__ import annotations

import errno
import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from lookagain_ml import (
    AdvancedConfig,
    CheckpointStore,
    CheckpointValidationError,
    ExecutionConfig,
    LinearHeadConfig,
    ProgressReporter,
    analyze,
    checkpointing,
)
from lookagain_ml.encoders import (
    EncoderMetadata,
    ImageEncoder,
    get_registration,
    register_encoder,
)
from replication._progress import (
    ProgressLedger,
    completed_output_manifest,
    validate_completed_outputs,
)


class IntentionalStop(RuntimeError):
    pass


def _metadata(name: str) -> EncoderMetadata:
    return EncoderMetadata(
        name=name,
        display_name=f"Deterministic {name} fixture",
        checkpoint=f"generated-{name}-checkpoint",
        checkpoint_revision="fixture-v1",
        feature_dimension=6,
        input_size=(12, 12),
        preprocessing="unrestricted deterministic RGB statistics",
        pooling="channel mean and standard deviation",
        backend="Pillow/NumPy test fixture",
    )


@pytest.fixture
def progress_encoders():
    originals = {name: get_registration(name) for name in ("resnet50", "vgg16")}
    calls = {"resnet50": 0, "vgg16": 0}
    failures: set[str] = set()

    class FixtureEncoder(ImageEncoder):
        def __init__(self, name: str, **kwargs) -> None:
            del kwargs
            self.name = name
            self.metadata = _metadata(name)
            self.costs = {"device": "cpu", "feature_dimension": 6}

        def encode(self, paths, *, batch_size: int = 32):
            calls[self.name] += 1
            if self.name in failures:
                raise RuntimeError(f"intentional {self.name} extraction failure")
            rows = []
            for start in range(0, len(paths), batch_size):
                for path in paths[start : start + batch_size]:
                    with Image.open(path) as image:
                        values = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                    rows.append(
                        np.concatenate(
                            [values.mean(axis=(0, 1)), values.std(axis=(0, 1))]
                        )
                    )
                self._report_batch_progress(min(start + batch_size, len(paths)), len(paths))
            return np.asarray(rows, dtype=np.float32)

    for name in originals:
        register_encoder(
            name,
            lambda name=name, **kwargs: FixtureEncoder(name, **kwargs),
            _metadata(name),
            replace=True,
        )
    try:
        yield calls, failures
    finally:
        for registration in originals.values():
            register_encoder(
                registration.name,
                registration.factory,
                registration.metadata,
                replace=True,
            )


def _inputs(image_factory, n: int = 30):
    images = [
        image_factory(
            f"progress-{index}.png",
            ((index * 17) % 255, (index * 31) % 255, (index * 47) % 255),
        )
        for index in range(n)
    ]
    outcome = np.linspace(-1.0, 1.0, n)
    groups = [f"group-{index}" for index in range(n)]
    splits = ["train"] * 18 + ["validation"] * 6 + ["test"] * 6
    return images, outcome, groups, splits


def _run(
    image_factory,
    *,
    execution: ExecutionConfig,
    encoders=("resnet50",),
    repeated_splits: int = 1,
    outcome_override=None,
):
    images, outcome, groups, splits = _inputs(image_factory)
    if outcome_override is not None:
        outcome = outcome_override
    return analyze(
        images,
        outcome,
        groups,
        split_labels=(splits if repeated_splits == 1 else None),
        task="regression",
        encoders=encoders,
        pretrained=True,
        cache=False,
        batch_size=7,
        repeated_splits=repeated_splits,
        advanced_config=AdvancedConfig(execution=execution),
    )


def _assert_same_science(left, right) -> None:
    assert left.selected_encoder == right.selected_encoder
    assert left.metrics == right.metrics
    np.testing.assert_array_equal(left.predictions["y_pred"], right.predictions["y_pred"])
    assert left.qc["test_outcomes_used_for_selection"] is False
    if left.repeated_split_results is not None:
        assert left.repeated_split_results == right.repeated_split_results


def test_progress_modes_and_callback_do_not_change_numerical_outputs(
    image_factory, progress_encoders
) -> None:
    events = []
    off = _run(image_factory, execution=ExecutionConfig(progress=False))
    callback = _run(
        image_factory,
        execution=ExecutionConfig(progress=False, progress_callback=events.append),
    )
    visible = _run(image_factory, execution=ExecutionConfig(progress=True))
    _assert_same_science(off, callback)
    _assert_same_science(off, visible)
    assert {event.event for event in events} >= {"STARTED", "CREATED", "ENCODING", "FITTING", "COMPLETED"}
    assert all("observation" not in event.to_dict() for event in events)
    completed = next(
        event
        for event in events
        if event.event == "COMPLETED" and event.stage == "encoder"
    )
    assert completed.encoder == "resnet50"
    assert completed.checkpoint == "generated-resnet50-checkpoint"
    assert completed.checkpoint_revision == "fixture-v1"
    assert completed.device == "cpu"
    assert completed.rows_processed == 30
    assert completed.feature_dimension == 6
    assert completed.cache_status == "disabled"
    assert completed.elapsed_seconds is not None


def test_encoder_interruption_reuses_valid_checkpoint_and_matches_uninterrupted(
    image_factory, progress_encoders, tmp_path
) -> None:
    calls, _ = progress_encoders
    checkpoint = tmp_path / "interrupted-encoder"

    def stop_after_first_encoder(event) -> None:
        if event.event == "CHECKPOINTED" and event.stage == "encoder" and event.encoder == "resnet50":
            raise IntentionalStop("simulated disconnect after first encoder")

    with pytest.raises(IntentionalStop):
        _run(
            image_factory,
            encoders=("resnet50", "vgg16"),
            execution=ExecutionConfig(
                progress=False,
                progress_callback=stop_after_first_encoder,
                checkpoint_dir=checkpoint,
            ),
        )
    assert calls == {"resnet50": 1, "vgg16": 0}
    resumed_events = []
    resumed = _run(
        image_factory,
        encoders=("resnet50", "vgg16"),
        execution=ExecutionConfig(
            progress=False,
            progress_callback=resumed_events.append,
            checkpoint_dir=checkpoint,
        ),
    )
    assert calls == {"resnet50": 1, "vgg16": 1}
    uninterrupted = _run(
        image_factory,
        encoders=("resnet50", "vgg16"),
        execution=ExecutionConfig(progress=False, checkpoint_dir=tmp_path / "continuous"),
    )
    _assert_same_science(resumed, uninterrupted)
    assert any(event.event == "REUSED" and event.encoder == "resnet50" for event in resumed_events)
    manifest = json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache_units"]["resnet50"]["status"] == "reused"
    assert manifest["cache_units"]["resnet50"]["representation_source"] == "safely-reused"
    assert "encoder:resnet50" in manifest["reused_units"]
    assert set(manifest["completed_units"]) >= {"encoder:resnet50", "encoder:vgg16"}
    assert manifest["pending_units"] == []


def test_repeated_split_interruption_resumes_only_missing_splits(
    image_factory, progress_encoders, tmp_path
) -> None:
    checkpoint = tmp_path / "interrupted-splits"

    def stop_after_split(event) -> None:
        if event.event == "CHECKPOINTED" and event.stage == "repeated_split" and event.split_index == 0:
            raise IntentionalStop("simulated disconnect after repeated split")

    with pytest.raises(IntentionalStop):
        _run(
            image_factory,
            repeated_splits=3,
            execution=ExecutionConfig(
                progress=False,
                progress_callback=stop_after_split,
                checkpoint_dir=checkpoint,
            ),
        )
    resumed_events = []
    resumed = _run(
        image_factory,
        repeated_splits=3,
        execution=ExecutionConfig(
            progress=False,
            progress_callback=resumed_events.append,
            checkpoint_dir=checkpoint,
        ),
    )
    uninterrupted = _run(
        image_factory,
        repeated_splits=3,
        execution=ExecutionConfig(progress=False, checkpoint_dir=tmp_path / "split-continuous"),
    )
    _assert_same_science(resumed, uninterrupted)
    assert any(
        event.event == "SKIPPED" and event.stage == "repeated_split_fitting"
        for event in resumed_events
    )
    assert all(
        audit["test_outcomes_used_for_decisions"] is False
        for audit in resumed.repeated_split_results["split_audits"]
    )
    manifest = json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))
    assert "repeated_image:split_0000" in manifest["skipped_units"]


def test_incompatible_corrupt_and_partial_checkpoints_are_rejected(tmp_path) -> None:
    identity = {
        "configuration_sha256": "a" * 64,
        "profile": "standard",
        "planned_units": ["unit:1"],
    }
    store = CheckpointStore(tmp_path / "valid", identity=identity)
    path = store.save_unit("unit:1", {"value": 1})
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["value"] = 2
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(CheckpointValidationError, match="checksum"):
        store.load_unit("unit:1")

    partial = CheckpointStore(tmp_path / "partial", identity=identity)
    partial_path = partial._unit_path("unit:1")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text("{", encoding="utf-8")
    with pytest.raises(CheckpointValidationError, match="unreadable or partial"):
        partial.load_unit("unit:1")

    with pytest.raises(CheckpointValidationError, match="incompatible"):
        CheckpointStore(
            tmp_path / "valid",
            identity={**identity, "configuration_sha256": "b" * 64},
        )


def test_adapter_output_manifest_rejects_changed_completed_output(tmp_path) -> None:
    output = tmp_path / "adapter-output"
    output.mkdir()
    result = output / "results.json"
    result.write_text('{"status":"PASS"}', encoding="utf-8")
    (output / "run_state").mkdir()
    (output / "run_state" / "mutable.json").write_text("{}", encoding="utf-8")
    manifest = completed_output_manifest(output)
    assert list(manifest) == ["results.json"]
    validate_completed_outputs(output, manifest)
    result.write_text('{"status":"changed"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or changed"):
        validate_completed_outputs(output, manifest)


def test_adapter_ledger_completion_writes_verified_result_checkpoint(tmp_path) -> None:
    output = tmp_path / "adapter-output"
    output.mkdir()
    result = output / "results.json"
    result.write_text('{"status":"PASS"}', encoding="utf-8")
    ledger = ProgressLedger(output / "adapter_state", configuration_sha256="a" * 64)
    ledger.update(
        "paper_adapter:dataset",
        status="PASS",
        protocol_fingerprint="adapter-protocol",
        details={"output_manifest": completed_output_manifest(output)},
    )
    ledger.complete(output_locations=[result])
    manifest = json.loads(
        (output / "adapter_state" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "COMPLETED"
    reference = manifest["result_checkpoint"]
    assert reference["checkpoint_schema_version"] == "adapter-output-manifest-v1"
    assert reference["components"]["results.json"]["sha256"] == hashlib.sha256(
        result.read_bytes()
    ).hexdigest()


def test_failure_event_and_manifest_preserve_actionable_exception(
    image_factory, progress_encoders, tmp_path
) -> None:
    _, failures = progress_encoders
    failures.add("vgg16")
    events = []
    checkpoint = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="intentional vgg16"):
        _run(
            image_factory,
            encoders=("vgg16",),
            execution=ExecutionConfig(
                progress=False,
                progress_callback=events.append,
                checkpoint_dir=checkpoint,
            ),
        )
    failed = [event for event in events if event.event == "FAILED"]
    assert failed[-1].details["exception_type"] == "RuntimeError"
    manifest = json.loads((checkpoint / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "FAILED"
    assert manifest["exception_type"] == "RuntimeError"
    assert "intentional vgg16" in manifest["exception_message"]
    assert not list(checkpoint.rglob("*.partial"))


def test_dependency_free_progress_is_concise_and_heartbeat_stops(monkeypatch) -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(True, stream=stream, heartbeat_interval_seconds=0.01)
    reporter.emit("ENCODING", "image_encoding", "batch", current=0, total=100)
    for index in range(1, 101):
        reporter.emit("ENCODING", "image_encoding", "batch", current=index, total=100)
    with reporter.heartbeat("cpu_fit", "fitting CPU estimator"):
        time.sleep(0.035)
    reporter.emit("COMPLETED", "cpu_fit", "CPU estimator complete")
    assert len(stream.getvalue().splitlines()) <= 8
    assert not any(thread.name.startswith("lookagain-progress-") for thread in __import__("threading").enumerate())
    quiet_stream = io.StringIO()
    ProgressReporter(False, stream=quiet_stream).emit(
        "STARTED", "analysis", "quiet run"
    )
    assert quiet_stream.getvalue() == ""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    redirected_stream = io.StringIO()
    auto = ProgressReporter("auto", stream=redirected_stream)
    auto.emit("STARTED", "analysis", "redirected job")
    assert "redirected job" in redirected_stream.getvalue()


@pytest.mark.parametrize("error", [RuntimeError("fit failed"), KeyboardInterrupt()])
def test_heartbeat_stops_and_reports_failure_on_every_exception_path(error) -> None:
    events = []
    reporter = ProgressReporter(
        False,
        callback=events.append,
        heartbeat_interval_seconds=0.01,
    )
    with pytest.raises(type(error)), reporter.heartbeat(
        "cpu_fit", "fitting CPU estimator"
    ):
        time.sleep(0.02)
        raise error
    assert events[-1].event == "FAILED"
    assert events[-1].details["exception_type"] == type(error).__name__
    assert not any(
        thread.name.startswith("lookagain-progress-")
        for thread in __import__("threading").enumerate()
    )


def test_atomic_replace_retries_only_recognized_transient_errors(
    tmp_path, monkeypatch
) -> None:
    actual_replace = checkpointing.os.replace
    calls = {"count": 0}

    def transient_then_succeed(source, destination):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError(errno.EACCES, "temporary sync lock")
        return actual_replace(source, destination)

    monkeypatch.setattr(checkpointing.os, "replace", transient_then_succeed)
    checkpointing._atomic_json(tmp_path / "retry.json", {"status": "ok"})
    assert calls["count"] == 3

    calls["count"] = 0

    def permanent_error(source, destination):
        del source, destination
        calls["count"] += 1
        raise OSError(errno.EINVAL, "permanent invalid operation")

    monkeypatch.setattr(checkpointing.os, "replace", permanent_error)
    with pytest.raises(OSError, match="permanent invalid operation"):
        checkpointing._atomic_json(tmp_path / "permanent.json", {"status": "no"})
    assert calls["count"] == 1


def test_repeated_xi_interruption_resumes_and_matches_uninterrupted(
    image_factory, progress_encoders, tmp_path
) -> None:
    images, outcome, groups, _ = _inputs(image_factory)
    covariates = pd.DataFrame(
        {"x_numeric": np.linspace(0.0, 2.0, len(images)), "x_category": ["a", "b"] * 15}
    )

    def execute(checkpoint_dir, callback=None):
        return analyze(
            images,
            outcome,
            groups,
            covariates=covariates,
            task="regression",
            pretrained=True,
            cache=False,
            repeated_splits=2,
            structured_models=("linear",),
            integration_methods=("linear_score",),
            advanced_config=AdvancedConfig(
                execution=ExecutionConfig(
                    progress=False,
                    progress_callback=callback,
                    checkpoint_dir=checkpoint_dir,
                )
            ),
        )

    interrupted = tmp_path / "xi-interrupted"

    def stop_after_xi_split(event) -> None:
        if event.event == "CHECKPOINTED" and event.stage == "repeated_xi_split" and event.split_index == 0:
            raise IntentionalStop("simulated disconnect after X/I/X+I split")

    with pytest.raises(IntentionalStop):
        execute(interrupted, stop_after_xi_split)
    resumed_events = []
    resumed = execute(interrupted, resumed_events.append)
    continuous = execute(tmp_path / "xi-continuous")
    _assert_same_science(resumed, continuous)
    assert any(
        event.event == "SKIPPED" and event.stage == "repeated_xi_fitting"
        for event in resumed_events
    )
    assert all(
        audit["test_outcomes_used_for_decisions"] is False
        for audit in resumed.repeated_split_results["x_i_x_plus_i"]["decision_audits"]
    )


def test_quick_profile_is_diagnostic_and_paper_profile_does_not_override_head(
    image_factory, progress_encoders
) -> None:
    quick = _run(
        image_factory,
        execution=ExecutionConfig(profile="quick", progress=False),
    )
    assert quick.configuration["analysis"]["diagnostic_only"] is True
    assert quick.configuration["analysis"]["scientific_comparison_eligible"] is False
    assert "logical_cpu_count" in quick.configuration["runtime"]["resources"]
    assert quick.configuration["runtime"]["resources"]["rf_n_jobs"] == -1

    paper_execution = ExecutionConfig(
        profile="paper",
        progress=False,
        protocol_identifier="fixture_paper_v1",
        protocol_fingerprint="f" * 64,
    )
    images, outcome, groups, splits = _inputs(image_factory)
    advanced = AdvancedConfig(
        linear_head=LinearHeadConfig(working_precision="float32", ridge_alphas=(0.1, 1.0)),
        execution=paper_execution,
    )
    result = analyze(
        images,
        outcome,
        groups,
        split_labels=splits,
        task="regression",
        pretrained=True,
        cache=False,
        advanced_config=advanced,
    )
    assert result.configuration["analysis"]["head_working_dtype"] == "float32"
    assert result.configuration["analysis"]["ridge_alphas"] == (0.1, 1.0)
    assert result.configuration["analysis"]["scientific_comparison_eligible"] is True


def test_resume_rejects_changed_locked_test_outcomes(
    image_factory, progress_encoders, tmp_path
) -> None:
    checkpoint = tmp_path / "outcome-identity"
    base = _run(
        image_factory,
        execution=ExecutionConfig(progress=False, checkpoint_dir=checkpoint),
    )
    _, changed, _, _ = _inputs(image_factory)
    changed = changed.copy()
    changed[-6:] += 1000
    with pytest.raises(CheckpointValidationError, match="incompatible"):
        _run(
            image_factory,
            outcome_override=changed,
            execution=ExecutionConfig(progress=False, checkpoint_dir=checkpoint),
        )
    assert base.qc["test_outcomes_used_for_selection"] is False


def test_execution_config_serialization_omits_callback_and_absolute_path(tmp_path) -> None:
    callback = lambda event: None
    config = AdvancedConfig(
        execution=ExecutionConfig(
            progress_callback=callback,
            checkpoint_dir=tmp_path / "private",
        )
    )
    record = config.to_dict()["execution"]
    assert record["progress_callback_supplied"] is True
    assert str(tmp_path) not in json.dumps(record)
    assert "progress_callback" not in record

    store = CheckpointStore(
        tmp_path / "portable",
        identity={
            "configuration_sha256": "d" * 64,
            "profile": "standard",
            "planned_units": [],
            "private_path": str(tmp_path / "private-input.csv"),
        },
    )
    assert str(tmp_path) not in json.dumps(store.portable_manifest())
