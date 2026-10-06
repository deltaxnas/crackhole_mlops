from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return cleaned or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_approved_model(
    config: dict,
    task_name: str,
    batch_id: str,
    training_attempt: int,
    evaluation_attempt: int,
    model_path: str,
    metrics: dict[str, float | int],
) -> Path:
    """Atomically copy an approved weight into deployment staging."""
    options = config.get("deployment", {})
    destination_text = options.get("destination")
    if not destination_text:
        raise ValueError(f"deployment destination is not configured for {task_name}")
    source = Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(f"approved model does not exist: {source}")
    destination = Path(destination_text)
    destination.mkdir(parents=True, exist_ok=True)

    metric_parts = []
    for metric in options.get("filename_metrics", ["precision", "recall"]):
        if metric in metrics:
            metric_parts.append(f"{_safe_name(metric)}-{float(metrics[metric]):.4f}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = _sha256(source)[:12]
    parts = [
        _safe_name(task_name),
        f"batch-{_safe_name(batch_id)}",
        f"train-{training_attempt:03d}",
        f"eval-{evaluation_attempt:03d}",
        *metric_parts,
        timestamp,
        f"sha256-{digest}",
    ]
    target = destination / ("__".join(parts) + source.suffix)
    temporary = destination / f".{target.name}.{uuid4().hex}.tmp"
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
