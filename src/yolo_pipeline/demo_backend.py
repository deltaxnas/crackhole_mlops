"""Optional desktop-demo services. No watcher or GUI dependencies here."""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .config import for_task
from .models import State
from .store import StateStore

TASK = "crack_detection"  # Default retained for callers of the original demo API.
TASKS = {"crack_detection": "Crack detection", "panel_segmentation": "Panel segmentation", "hole_detection": "Hole detection"}
FLOOR = 0.2
TERMINAL = {State.APPROVED, State.REJECTED, State.FAILED}


def unique_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex}"


def demo_config(config: dict, task_name: str = TASK) -> dict:
    """Keep demo approval rules in memory; never rewrite the user's YAML."""
    result = copy.deepcopy(config)
    if task_name not in TASKS or task_name not in result.get("tasks", {}):
        raise ValueError(f"Unsupported or unconfigured task: {task_name}")
    evaluation = result["tasks"][task_name].setdefault("evaluation", {})
    evaluation.update(min_precision=FLOOR, min_recall=FLOOR, fail_on_image_error=True)
    evaluation["comparison"] = {
        "enabled": True, "metrics": ["precision", "recall"], "max_regression": 0.0,
    }
    return result


def leaderboard(store: StateStore, task: str = TASK, exclude_id: int | None = None, model_backend: str = "yolo") -> list[dict]:
    """Only completed evaluations belong on the board; surface broken reports."""
    rows = []
    for run in store.list_all():
        if run.task != task or run.model_backend != model_backend or run.id == exclude_id or run.state not in {State.APPROVED, State.REJECTED}:
            continue
        try:
            if not run.evaluation_path:
                raise ValueError("no evaluation report recorded")
            report = json.loads(Path(run.evaluation_path).read_text(encoding="utf-8"))
            p, r = float(report["precision"]), float(report["recall"])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (p, r)):
                raise ValueError("precision and recall must be finite values between 0 and 1")
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            rows.append({"run": run, "precision": p, "recall": r, "f1": f1,
                         "report": report, "error": None})
        except (OSError, ValueError, KeyError, TypeError) as error:
            rows.append({"run": run, "precision": None, "recall": None, "f1": -1,
                         "report": {}, "error": str(error)})
    return sorted(rows, key=lambda row: (row["f1"], row["run"].id), reverse=True)


class DemoStore(StateStore):
    def previous_approved(self, task: str, before_run_id: int, model_backend: str = "yolo"):
        """The demo compares against the highest-F1 approved model on its board."""
        approved = [row for row in leaderboard(self, task, before_run_id, model_backend)
                    if row["run"].state == State.APPROVED]
        broken = [row for row in approved if row["error"]]
        if broken:
            raise ValueError(f"Cannot compare with approved model {broken[0]['run'].batch_id}: "
                             f"{broken[0]['error']}")
        return approved[0]["run"] if approved else None


@contextmanager
def worker_lock(database: str):
    """Prevent two demo workers from training/inferencing against one workspace."""
    path = Path(database).with_suffix(".demo.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another demo job is using this workspace. Wait for it to finish.") from error
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def image_files(source: Path, extensions: list[str]) -> list[Path]:
    if not source.is_dir():
        raise ValueError(f"Image folder does not exist: {source}")
    extensions = {ext.lower() for ext in extensions}
    files = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Please use regular files and directories, not symlinks: {path}")
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    if not files:
        raise ValueError("The selected folder contains no supported images.")
    return files


def _signature(paths: list[Path]) -> list[tuple]:
    return [(str(path), path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
            for path in paths]


def import_batch(config: dict, source: str, emit=lambda **event: None, task_name: str = TASK, model_backend: str | None = None):
    """Explicitly import a finished folder and enter LABELING without a watcher."""
    if task_name not in TASKS or task_name not in config.get("tasks", {}):
        raise ValueError(f"Unsupported or unconfigured task: {task_name}")
    backend = for_task(config, task_name, model_backend)["model_backend"]
    source = Path(source).resolve()
    labeling = Path(config["paths"]["labeling"]).resolve()
    if source == labeling or source in labeling.parents:
        raise ValueError("Choose an image batch, not a parent of the labeling workspace.")
    images = image_files(source, config["dataset"]["image_extensions"])
    if len({p.stem.casefold() for p in images}) != len(images):
        raise ValueError("Image stems must be unique within a batch so Labelme annotations cannot collide.")
    annotations = [p.with_suffix(".json") for p in images if p.with_suffix(".json").exists()]
    if any(path.is_symlink() for path in annotations):
        raise ValueError("Annotation symlinks are not supported.")
    paths = images + annotations
    before = _signature(paths)
    batch_id = unique_id()
    destination = labeling / batch_id
    labeling.mkdir(parents=True, exist_ok=True)
    store = StateStore(config["paths"]["state_db"])
    with TemporaryDirectory(prefix=".demo-import-", dir=labeling) as temporary:
        staged = Path(temporary) / "batch"
        staged.mkdir()
        for index, path in enumerate(paths, 1):
            shutil.copy2(path, staged / path.name)
            emit(kind="progress", current=index, total=len(paths), message=f"Copying {path.name}")
        current_images = image_files(source, config["dataset"]["image_extensions"])
        current_annotations = [p.with_suffix(".json") for p in current_images if p.with_suffix(".json").exists()]
        if before != _signature(current_images + current_annotations):
            raise ValueError("The source changed while copying. Finish transferring it and select it again.")
        (staged / "BATCH_INFO.txt").write_text(
            f"batch_id={batch_id}\ntask={task_name}\nmodel_backend={backend}\nsource_name={source.name}\nsource_path={source}\n"
            "Imported manually in the demo GUI. Save annotations beside the images, "
            "then use Mark done & train in the main window.\n", encoding="utf-8",
        )
        staged.rename(destination)
        try:
            run = store.create(batch_id, str(destination), model_backend=backend)
        except Exception:
            shutil.rmtree(destination)
            raise
        if run is None:
            shutil.rmtree(destination)
            raise RuntimeError("Could not allocate a unique batch ID; try importing again.")
    store.transition(run.id, State.NEW_DATA, State.LABELING, "manual demo import", task=task_name)
    return store.get(run.id)


def validate_labeling(config: dict, batch: Path) -> None:
    """Require an explicit valid annotation even for images without defects."""
    images = image_files(batch, config["dataset"]["image_extensions"])
    # Generated subdirectories can exist when reopening a failed preprocessing run.
    images = [image for image in images if image.parent == batch]
    if not images:
        raise ValueError("No source images remain in the labeling folder.")
    for image in images:
        annotation = image.with_suffix(".json")
        try:
            data = json.loads(annotation.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
                raise ValueError("expected a Labelme shapes list")
        except (OSError, ValueError) as error:
            raise ValueError(f"Save a valid Labelme annotation for {image.name}: {error}") from error


def run_training(config: dict, batch_id: str, emit=lambda **event: None, stop_file: str | None = None) -> None:
    from .orchestrator import Orchestrator

    store = DemoStore(config["paths"]["state_db"])
    run = store.get_by_batch_id(batch_id)
    config = demo_config(config, run.task)
    if stop_file:
        config["tasks"][run.task].setdefault("training", {})["stop_file"] = stop_file
    if run.state == State.LABELING:
        validate_labeling(config, Path(run.batch_path))
        marker = config["tasks"][run.task]["done_marker"]
        if Path(marker).name != marker:
            raise ValueError("The DONE marker must be a filename.")
        others = [task["done_marker"] for name, task in config["tasks"].items() if name != run.task]
        if any((Path(run.batch_path) / name).exists() for name in others):
            raise ValueError("Remove other task DONE markers before starting training.")
        (Path(run.batch_path) / marker).touch()
    elif run.state == State.FAILED:
        run = store.retry_failed(batch_id)
    elif run.state in TERMINAL:
        raise ValueError("This batch is finished. Import another batch to start a new demo run.")
    elif run.state in {State.DATASET_BUILDING, State.TRAINING, State.EVALUATING}:
        # The caller owns the demo worker lock. Only recover this explicitly selected run.
        target = {State.DATASET_BUILDING: State.LABELING_COMPLETE,
                  State.TRAINING: State.DATASET_READY, State.EVALUATING: State.TRAINING_COMPLETE}[run.state]
        store.transition(run.id, run.state, target, "resuming interrupted demo job")
    pipeline = Orchestrator(config, store)
    while True:
        run = store.get(run.id)
        emit(kind="state", batch_id=batch_id, state=run.state.value)
        if run.state in TERMINAL:
            if run.state == State.FAILED:
                raise RuntimeError(run.error or "Pipeline stage failed; inspect ERROR.txt.")
            return
        pipeline.step(run)
        if store.get(run.id).state == run.state:
            raise RuntimeError(f"Batch made no progress from {run.state}; check its DONE marker.")


def training_history(config: dict, run) -> list[dict[str, float]]:
    task_config = for_task(config, run.task or TASK, run.model_backend)
    folder = Path(task_config["paths"]["runs"]) / run.batch_id / f"train_{run.training_attempt:03d}"
    path = folder / "results.csv"
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            result = []
            for raw in csv.DictReader(handle):
                try:
                    row = {key.strip(): float(value) for key, value in raw.items()
                           if key and (key.strip() == "epoch" or "loss" in key)}
                    if "epoch" in row and all(math.isfinite(v) for v in row.values()):
                        result.append(row)
                except (TypeError, ValueError):
                    pass  # A writer may currently be appending the last row.
            return result
    except FileNotFoundError:
        return []


def run_inference(config: dict, source: str, model_path: str, output: str, emit=lambda **event: None, task_name: str = TASK, model_backend: str | None = None):
    task = for_task(config, task_name, model_backend)
    if task["model_backend"] == "florence":
        from .florence_backend import infer_florence
        return infer_florence(task, source, model_path, output, emit)
    from ultralytics import YOLO

    source, model_path, output = Path(source).resolve(), Path(model_path).resolve(), Path(output).resolve()
    if not model_path.is_file():
        raise ValueError(f"Model file does not exist: {model_path}")
    if source == output or source in output.parents:
        raise ValueError("Inference output must be outside the input folder.")
    images = image_files(source, config["dataset"]["image_extensions"])
    relative_stems = [p.relative_to(source).with_suffix("").as_posix().casefold() for p in images]
    if len(set(relative_stems)) != len(relative_stems):
        raise ValueError("Images sharing a filename stem would overwrite inference labels.")
    output.mkdir(parents=True, exist_ok=False)
    model = YOLO(str(model_path))
    records, errors = [], []
    for index, path in enumerate(images, 1):
        try:
            predictions = model.predict(str(path), device=task["training"].get("device"),
                                        conf=task.get("evaluation", {}).get("confidence", 0.25),
                                        save=False, verbose=False)
            result = predictions[0]
            relative = path.relative_to(source)
            annotated = output / "images" / relative
            label = output / "labels" / relative.with_suffix(".txt")
            annotated.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            result.save(filename=str(annotated))
            label.touch()
            result.save_txt(str(label), save_conf=True)
            records.append({"image": str(path), "annotated": str(annotated), "label": str(label)})
        except Exception as error:
            errors.append({"image": str(path), "error": str(error)})
        emit(kind="progress", current=index, total=len(images), message=f"Inference {index}/{len(images)}")
    report = {"model": str(model_path), "source": str(source), "task": task_name, "processed": len(records),
              "total": len(images), "images": records, "errors": errors}
    (output / "inference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not records:
        raise RuntimeError(f"All inference images failed. See {output / 'inference.json'}")
    return report
