"""Standard YOLO box/mask validation for the basic hole and panel tasks."""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from .dataset import _class_names, _entry_images, _label_path, _validate_label
from .evaluation_adapter import EvaluationResult


def evaluate_yolo_model(model_path, dataset_yaml, config, batch_id, evaluation_attempt=1):
    from ultralytics import YOLO

    options = config.get("evaluation", {})
    data_path = Path(options.get("dataset_yaml") or dataset_yaml).resolve()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = config["dataset"]["class_names"]
    if _class_names(data) != names:
        raise ValueError("evaluation dataset classes must match the training task")
    split = options.get("split", "val")
    if split not in {"val", "test"} or split not in data:
        raise ValueError("evaluation requires an existing val or test split")
    root = Path(data.get("path", data_path.parent))
    if not root.is_absolute():
        root = data_path.parent / root
    extensions = {ext.lower() for ext in config["dataset"]["image_extensions"]}
    images = list(_entry_images(data[split], root, data_path, extensions))
    if not images:
        raise ValueError("The evaluation split is empty. Label at least two images, or configure evaluation.dataset_yaml.")
    for image in images:
        if not image.is_file():
            raise FileNotFoundError(f"evaluation image not found: {image}")
        label = _label_path(image)
        if not label.is_file():
            raise ValueError(f"Missing evaluation label: {label}; use an empty file for reviewed negatives")
        _validate_label(label, len(names), config["task_type"])
    output_root = options.get("output_dir") or str(Path(config["paths"]["state_db"]).parent / "evaluations" / config["task_name"])
    output = Path(output_root) / batch_id / f"eval_{evaluation_attempt:03d}"
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    if model.task != config["task_type"]:
        raise ValueError(f"Model task {model.task!r} does not match {config['task_type']!r}")
    observed = {}

    def finished(validator):
        observed["images"] = int(validator.seen)

    model.add_callback("on_val_end", finished)
    metrics = model.val(
        data=str(data_path), split=split,
        device=options.get("device", config["training"].get("device")),
        imgsz=config["training"]["imgsz"], batch=config["training"]["batch"],
        conf=float(options.get("confidence", 0.001)),
        workers=int(options.get("workers", 0)),
        project=str(output), name="validation", exist_ok=True, plots=False, verbose=False,
    )
    count = observed.get("images", 0)
    if count != len(images):
        raise ValueError(f"Validation processed {count}/{len(images)} images; check for corrupt images or labels")
    segmentation = config["task_type"] == "segment"
    task_metrics = metrics.seg if segmentation else metrics.box
    values = {"precision": float(task_metrics.mp), "recall": float(task_metrics.mr),
              "map50": float(task_metrics.map50), "map50_95": float(task_metrics.map)}
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
        raise ValueError("validation returned invalid metrics")
    thresholds = {"min_precision": float(options.get("min_precision", 0.2)),
                  "min_recall": float(options.get("min_recall", 0.2)),
                  "min_images": int(options.get("min_images", 1))}
    passed = (values["precision"] >= thresholds["min_precision"] and
              values["recall"] >= thresholds["min_recall"] and count >= thresholds["min_images"])
    report = {"batch_id": batch_id, "task": config["task_name"], "evaluation_attempt": evaluation_attempt,
              "model_path": str(Path(model_path).resolve()), "training_dataset_yaml": str(Path(dataset_yaml).resolve()),
              "evaluation_data": str(data_path), "evaluation_split": split,
              "evaluation_source": "fixed_dataset" if options.get("dataset_yaml") else "batch_validation",
              "metric_type": "mask" if segmentation else "box", "images_evaluated": count,
              "thresholds": thresholds, **values, "passed": passed}
    report_path = output / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return EvaluationResult(
        passed, f"{report['metric_type']} precision={values['precision']:.4f}, recall={values['recall']:.4f}, "
                f"mAP50={values['map50']:.4f}, images={count}",
        str(report_path), {**values, "images_evaluated": count},
    )
