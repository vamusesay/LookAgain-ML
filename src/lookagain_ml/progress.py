"""Dependency-free progress events for terminals, notebooks, and applications."""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ProgressMode = Literal["auto"] | bool
ProgressCallback = Callable[["ProgressEvent"], None]

EVENT_TYPES = frozenset(
    {
        "STARTED",
        "DOWNLOADING",
        "ENCODING",
        "VERIFYING",
        "FITTING",
        "CREATED",
        "REUSED",
        "CHECKPOINTED",
        "COMPLETED",
        "SKIPPED",
        "FAILED",
    }
)

_FRIENDLY_ENCODER_NAMES = {
    "resnet50": "ResNet-50",
    "vgg16": "VGG-16",
    "inception_v3": "Inception v3",
    "mobilenet_v2": "MobileNetV2",
    "coco_deeplab": "COCO DeepLab semantic features",
    "ade20k_segformer": "ADE20K SegFormer semantic features",
    "dinov2_vitb14": "DINOv2 ViT-B/14",
    "siglip2_b16": "SigLIP 2 B/16",
    "convnext_b": "ConvNeXt-B",
    "vit_b16": "ViT-B/16",
}


def readable_encoder_name(identifier: str) -> str:
    """Return a stable applied-researcher label while preserving the registry ID."""

    if identifier in _FRIENDLY_ENCODER_NAMES:
        return _FRIENDLY_ENCODER_NAMES[identifier]
    try:
        from .encoders.registry import encoder_metadata

        return encoder_metadata(identifier).display_name
    except (ImportError, KeyError, ValueError):
        return identifier


def _encoder_label(identifier: str | None) -> str | None:
    if identifier is None:
        return None
    return f"{readable_encoder_name(identifier)} ({identifier})"


def _count_text(event: ProgressEvent, *, noun: str = "items") -> str | None:
    if event.current is None or event.total is None:
        return None
    return f"{event.current:,}/{event.total:,} {noun}"


def _human_message(event: ProgressEvent) -> str:
    """Render concise progress without exposing technical stage codes or paths."""

    encoder = _encoder_label(event.encoder)
    image_count = _count_text(event, noun="images")
    elapsed = (
        f" Elapsed: {event.elapsed_seconds:.1f}s."
        if event.elapsed_seconds is not None
        else ""
    )
    if event.stage in {"input_validation", "input_integrity"}:
        if event.event == "FAILED":
            return "Image readability and content-identity checks failed."
        if image_count:
            return f"Checking image readability and content identity: {image_count}.{elapsed}"
        return f"Checking image readability and content identity.{elapsed}"
    if encoder and event.stage in {"encoder", "encoder_creation", "checkpoint_loading"}:
        if event.stage == "checkpoint_loading" and event.event == "STARTED":
            return (
                f"Checking the pinned checkpoint for {encoder}; downloading it only if absent,"
                f" then loading it"
                f" on {event.device or 'the selected device'}.{elapsed}"
            )
        if event.event == "CREATED":
            return f"Loaded and verified {encoder} on {event.device or 'the selected device'}."
        if event.event == "FAILED":
            return f"Could not load or verify {encoder}.{elapsed}"
        if event.event == "CHECKPOINTED":
            return f"Saved the completed representation checkpoint for {encoder}.{elapsed}"
        if event.event == "COMPLETED":
            return f"Finished representation processing for {encoder}.{elapsed}"
        return f"Preparing {encoder}.{elapsed}"
    if encoder and event.stage == "image_encoding":
        if event.event == "REUSED":
            return f"Loaded and verified the cached representation for {encoder}; no encoding ran."
        if event.event == "CHECKPOINTED":
            return f"Saved the verified representation cache for {encoder} atomically."
        if event.event == "FAILED":
            return f"Image encoding failed for {encoder}.{elapsed}"
        if image_count:
            return f"Encoding with {encoder}: {image_count}.{elapsed}"
        return f"Encoding with {encoder}.{elapsed}"
    if encoder and event.stage == "linear_head":
        if event.event == "FAILED":
            return f"Downstream-head fitting failed for {encoder}.{elapsed}"
        if event.event == "COMPLETED":
            return f"Finished the validation-selected downstream head for {encoder}.{elapsed}"
        return (
            f"Fitting downstream heads for {encoder}; settings use validation data only."
            f"{elapsed}"
        )
    if encoder and event.stage == "completed_result_compatibility":
        suffix = f": {image_count}" if image_count else ""
        return f"Verifying cached provenance for {encoder}{suffix}.{elapsed}"
    stage_messages = {
        "analysis": f"Running LookAgain-ML analysis ({event.message.rstrip('.')})",
        "completed_result_compatibility": (
            "Checking whether a verified completed run can be reused"
        ),
        "result_loading": "Reusing a verified completed run; no scientific computation will rerun",
        "image_combination": (
            "Fitting image-combination and stacking methods using validation data only"
        ),
        "feature_concatenation": (
            "Fitting the train-only multi-representation feature combination"
        ),
        "structured_and_integration": (
            "Fitting structured and X/I/X+I candidates using training and validation data only"
        ),
        "locked_test_evaluation": (
            "Evaluating the locked test set after all model and representation choices were frozen"
        ),
        "locked_test_uncertainty": (
            "Computing paired locked-test uncertainty for the already selected fitted models"
        ),
        "result_checkpoint": "Saving and verifying the completed scientific result atomically",
        "repeated_splits": "Running repeated group-safe split refits",
        "repeated_x_i_x_plus_i": "Running repeated group-safe X/I/X+I refits",
    }
    message = stage_messages.get(event.stage, event.message.rstrip("."))
    count = _count_text(event)
    if count:
        message = f"{message}: {count}"
    if event.event == "FAILED":
        message = f"{message} failed"
    elif event.event == "COMPLETED":
        message = f"{message} completed"
    return f"{message}.{elapsed}".replace("..", ".")


@dataclass(frozen=True)
class ProgressEvent:
    """One non-identifying execution event.

    Paths are available to callbacks and persistent local logs but the built-in
    renderer deliberately omits them. Observation identifiers and outcomes are
    never event fields.
    """

    event: str
    stage: str
    message: str
    run_id: str
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    encoder: str | None = None
    checkpoint: str | None = None
    checkpoint_revision: str | None = None
    device: str | None = None
    rows_processed: int | None = None
    feature_dimension: int | None = None
    cache_status: str | None = None
    elapsed_seconds: float | None = None
    current: int | None = None
    total: int | None = None
    split_index: int | None = None
    location: str | None = None
    representation_source: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event not in EVENT_TYPES:
            raise ValueError(f"Unknown progress event {self.event!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Return the documented JSON event schema without null fields."""

        return {key: value for key, value in asdict(self).items() if value is not None}


def _auto_visible() -> bool:
    # Interactive surfaces get concise updating lines; redirected/noninteractive
    # runs get throttled textual events so job logs still show real activity.
    return not bool(os.environ.get("PYTEST_CURRENT_TEST"))


class ProgressReporter:
    """Emit callbacks and concise dependency-free visible progress."""

    def __init__(
        self,
        mode: ProgressMode = "auto",
        *,
        callback: ProgressCallback | None = None,
        run_id: str | None = None,
        heartbeat_interval_seconds: float = 30.0,
        stream: Any = None,
        log_callback: ProgressCallback | None = None,
    ) -> None:
        if mode not in {"auto", True, False}:
            raise ValueError("progress must be 'auto', True, or False.")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive.")
        self.mode = mode
        self.callback = callback
        self.log_callback = log_callback
        self.run_id = run_id or uuid.uuid4().hex
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.stream = stream or sys.stderr
        self.visible = _auto_visible() if mode == "auto" else bool(mode)
        self.interactive = bool(getattr(self.stream, "isatty", lambda: False)()) or "ipykernel" in sys.modules
        self._last_batch_render = 0.0
        self._batch_line_open = False

    def emit(self, event: str, stage: str, message: str, **fields: Any) -> ProgressEvent:
        item = ProgressEvent(
            event=event,
            stage=stage,
            message=message,
            run_id=self.run_id,
            **fields,
        )
        if self.log_callback is not None:
            self.log_callback(item)
        if self.callback is not None:
            self.callback(item)
        if self.visible:
            self._render(item)
        return item

    def _render(self, event: ProgressEvent) -> None:
        now = time.monotonic()
        is_batch = event.event in {"ENCODING", "VERIFYING"} and event.current is not None
        if (
            is_batch
            and not self.interactive
            and event.current != event.total
            and now - self._last_batch_render < 30.0
        ):
            return
        text = _human_message(event)
        if is_batch and self.interactive and event.current != event.total:
            print(f"\r{text}", end="", file=self.stream, flush=True)
            self._batch_line_open = True
        else:
            if self._batch_line_open:
                print(file=self.stream, flush=True)
                self._batch_line_open = False
            print(text, file=self.stream, flush=True)
        if is_batch:
            self._last_batch_render = now

    @contextmanager
    def heartbeat(
        self,
        stage: str,
        message: str,
        **fields: Any,
    ) -> Iterator[None]:
        """Emit real elapsed-time fitting heartbeats and stop on every exit path."""

        started = time.perf_counter()
        stop = threading.Event()
        self.emit("FITTING", stage, message, elapsed_seconds=0.0, **fields)

        def worker() -> None:
            while not stop.wait(self.heartbeat_interval_seconds):
                self.emit(
                    "FITTING",
                    stage,
                    f"{message} (still running)",
                    elapsed_seconds=time.perf_counter() - started,
                    **fields,
                )

        thread = threading.Thread(
            target=worker,
            name=f"lookagain-progress-{stage}",
            daemon=False,
        )
        thread.start()
        error: BaseException | None = None
        try:
            yield
        except BaseException as caught:
            error = caught
            raise
        finally:
            stop.set()
            thread.join()
            if error is not None:
                self.emit(
                    "FAILED",
                    stage,
                    f"{message} failed",
                    elapsed_seconds=time.perf_counter() - started,
                    details={
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    },
                    **fields,
                )


def encoder_batch_callback(
    reporter: ProgressReporter,
    *,
    encoder: str,
) -> Callable[[int, int], None]:
    """Adapt encoder batch counts to the public event schema."""

    def callback(completed: int, total: int) -> None:
        reporter.emit(
            "ENCODING",
            "image_encoding",
            "encoding image batch",
            encoder=encoder,
            current=int(completed),
            total=int(total),
            rows_processed=int(completed),
        )

    return callback


def run_with_heartbeat(
    reporter: ProgressReporter,
    stage: str,
    message: str,
    operation: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a blocking operation with real elapsed-time heartbeats."""

    with reporter.heartbeat(stage, message):
        result = operation(*args, **kwargs)
    reporter.emit("COMPLETED", stage, f"{message} completed")
    return result


def portable_path(path: str | Path, *, root: str | Path | None = None) -> str:
    """Return a relative path when possible for shareable manifests."""

    candidate = Path(path).expanduser().resolve()
    if root is not None:
        try:
            return candidate.relative_to(Path(root).expanduser().resolve()).as_posix()
        except ValueError:
            pass
    return candidate.name
