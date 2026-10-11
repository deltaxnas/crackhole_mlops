"""Optional Florence-2 training, evaluation, and inference services.

The JSONL targets, masked token loss, device handling, and generation mask fix
follow the supplied florence/train_florence2_od.py and infer_florence2_od.py.
Heavy dependencies are imported only when a Florence job is requested.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from .florence_data import parse_boxes, read_records

log = logging.getLogger(__name__)


def settings(config: dict) -> dict:
    options = {
        "model": "microsoft/Florence-2-base", "revision": None,
        "trust_remote_code": True, "local_files_only": False,
        "device": config.get("training", {}).get("device", "cpu"),
        "epochs": 10, "batch_size": 1, "gradient_accumulation_steps": 8,
        "learning_rate": 1e-5, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "freeze_vision_encoder": False, "max_new_tokens": 1024,
        "num_beams": 3, "iou_threshold": 0.5,
    }
    options.update(config.get("florence", {}))
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "max_new_tokens", "num_beams"):
        if isinstance(options[key], bool) or int(options[key]) != options[key] or options[key] < 1:
            raise ValueError(f"florence.{key} must be a positive integer")
    for key in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(float(options[key])) or float(options[key]) <= 0:
            raise ValueError(f"florence.{key} must be positive and finite")
    if not 0 < float(options["iou_threshold"]) <= 1:
        raise ValueError("florence.iou_threshold must be in (0, 1]")
    return options


class FlorenceRuntime:
    def __init__(self, model_path: str, options: dict):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        except ImportError as error:
            raise RuntimeError('Florence dependencies are missing. Install with: pip install -e ".[demo,florence]"') from error
        self.torch, self.options = torch, options
        value = str(options["device"])
        if value == "auto":
            value = "cuda:0" if torch.cuda.is_available() else "cpu"
        if value.isdigit():
            value = f"cuda:{value}"
        if value != "cpu" and value != "cuda" and not (value.startswith("cuda:") and value[5:].isdigit()):
            raise ValueError("Florence device must be cpu, auto, or one CUDA device (for example 0 or cuda:0)")
        self.device = torch.device(value)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; set florence.device to cpu")
            torch.cuda.set_device(self.device)
        self.dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if self.device.type == "cuda" else torch.float32
        kwargs = {"trust_remote_code": options["trust_remote_code"], "local_files_only": options["local_files_only"]}
        if options.get("revision") and not Path(model_path).is_dir():
            kwargs["revision"] = options["revision"]
        self.processor = AutoProcessor.from_pretrained(model_path, **kwargs)
        config = AutoConfig.from_pretrained(model_path, **kwargs)
        if getattr(config, "model_type", None) != "florence2":
            raise ValueError("The selected checkpoint is not a Florence-2 model")
        # Florence remote checkpoints saved with Transformers 4.49 can lose this.
        if hasattr(config, "vision_config"):
            config.vision_config.model_type = "davit"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=self.dtype, attn_implementation="eager", **kwargs,
        ).to(self.device)

    def batch(self, records: list[dict]) -> dict:
        images = []
        for record in records:
            with Image.open(record["image"]) as image:
                images.append(image.convert("RGB"))
        inputs = dict(self.processor(text=["<OD>"] * len(records), images=images, return_tensors="pt", padding=True))
        targets = self.processor.tokenizer([record["suffix"] for record in records], return_tensors="pt", padding=True, return_token_type_ids=False)
        targets["input_ids"][targets["attention_mask"] == 0] = -100
        inputs["labels"] = targets["input_ids"]
        return self.move(inputs)

    def move(self, inputs: dict) -> dict:
        return {key: value.to(self.device, dtype=self.dtype) if self.torch.is_floating_point(value)
                else value.to(self.device) for key, value in inputs.items()}

    def predict(self, image: Image.Image) -> dict:
        self.model.eval()
        inputs = self.move(dict(self.processor(text="<OD>", images=image, return_tensors="pt")))
        inputs.pop("attention_mask", None)  # Florence prepends image embeddings.
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=self.options["max_new_tokens"],
                                      num_beams=self.options["num_beams"], do_sample=False)
        text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        result = self.processor.post_process_generation(text, task="<OD>", image_size=image.size)["<OD>"]
        boxes, labels = result.get("bboxes", []), result.get("labels", [])
        if len(boxes) != len(labels) or any(len(box) != 4 or not all(math.isfinite(float(x)) for x in box) for box in boxes):
            raise ValueError("Florence returned invalid detection boxes")
        return {"bboxes": boxes, "labels": labels, "generated_text": text}

    def save(self, destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        backup = destination.with_name(f".{destination.name}.{uuid4().hex}.backup")
        try:
            # Transformers 4.49 auto-loads custom classes without registering
            # them for export. Include their Python files for offline reloads.
            self.model.register_for_auto_class("AutoModelForCausalLM")
            self.model.config.register_for_auto_class("AutoConfig")
            self.processor.register_for_auto_class("AutoProcessor")
            self.model.save_pretrained(temporary, safe_serialization=True)
            self.processor.save_pretrained(temporary)
            if destination.exists():
                destination.rename(backup)
            try:
                temporary.rename(destination)
            except Exception:
                if backup.exists():
                    backup.rename(destination)
                raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        if backup.exists():
            shutil.rmtree(backup)


def train_florence(config: dict, batch_id: str, dataset_path: str, training_attempt: int = 1) -> Path:
    options = settings(config)
    training = read_records(Path(dataset_path) / "train.jsonl")
    validation = read_records(Path(dataset_path) / "validation.jsonl")
    seed = config["dataset"]["seed"]
    runtime = FlorenceRuntime(options["model"], options)
    torch, model = runtime.torch, runtime.model
    # Keep optimizer parameters in FP32; GradScaler cannot unscale FP16 weights.
    model.float()
    torch.manual_seed(seed)
    if options["freeze_vision_encoder"]:
        tower = getattr(model, "vision_tower", None)
        if tower is None:
            tower = getattr(getattr(model, "model", None), "vision_tower", None)
        if tower is None:
            raise ValueError("Cannot find Florence vision encoder to freeze")
        for parameter in tower.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=options["learning_rate"], weight_decay=options["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.device.type == "cuda" and runtime.dtype == torch.float16)
    folder = Path(config["paths"]["runs"]) / batch_id / f"train_{training_attempt:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "training_config.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
    checkpoint = folder / "best"
    batch_size, accumulation = options["batch_size"], options["gradient_accumulation_steps"]
    stop_file = config.get("training", {}).get("stop_file")
    best_loss, stopped = math.inf, False
    with (folder / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train/loss", "val/loss"])
        writer.writeheader()
        handle.flush()
        for epoch in range(1, options["epochs"] + 1):
            shuffled = list(training)
            random.Random(seed + epoch).shuffle(shuffled)
            batches = [shuffled[i:i + batch_size] for i in range(0, len(shuffled), batch_size)]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for index, records in enumerate(batches):
                with torch.autocast("cuda", dtype=runtime.dtype) if runtime.device.type == "cuda" else nullcontext():
                    loss = model(**runtime.batch(records)).loss
                value = float(loss.detach().float())
                if not math.isfinite(value):
                    raise ValueError("Nonfinite Florence training loss; no checkpoint will be evaluated")
                group_start = (index // accumulation) * accumulation
                group_end = min(group_start + accumulation, len(batches))
                group_samples = sum(len(batch) for batch in batches[group_start:group_end])
                scaler.scale(loss * len(records) / group_samples).backward()
                if index + 1 == group_end:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), options["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                total_loss += value * len(records)
                log.info("[%s] Florence epoch=%d/%d step=%d/%d loss=%.6f", batch_id, epoch, options["epochs"], index + 1, len(batches), value)
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for offset in range(0, len(validation), batch_size):
                    records = validation[offset:offset + batch_size]
                    with torch.autocast("cuda", dtype=runtime.dtype) if runtime.device.type == "cuda" else nullcontext():
                        loss = model(**runtime.batch(records)).loss
                    validation_loss += float(loss.detach().float()) * len(records)
            validation_loss /= len(validation)
            if not math.isfinite(validation_loss):
                raise ValueError("Nonfinite Florence validation loss")
            if validation_loss < best_loss:
                runtime.save(checkpoint)
                best_loss = validation_loss
            writer.writerow({"epoch": epoch, "train/loss": total_loss / len(training), "val/loss": validation_loss})
            handle.flush()
            stopped = bool(stop_file and Path(stop_file).is_file())
            (folder / "training_summary.json").write_text(json.dumps({
                "model_backend": "florence", "requested_epochs": options["epochs"], "epochs_completed": epoch,
                "stop_requested": stopped, "checkpoint": str(checkpoint), "best_validation_loss": best_loss,
                "training_attempt": training_attempt,
            }, indent=2), encoding="utf-8")
            if stopped:
                log.info("[%s] Stop requested; continuing with best saved Florence checkpoint", batch_id)
                break
    del optimizer, model, runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checkpoint


def iou(first, second) -> float:
    intersection = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(0, min(first[3], second[3]) - max(first[1], second[1]))
    area = lambda box: max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    union = area(first) + area(second) - intersection
    return intersection / union if union else 0.0


def detection_counts(targets, predictions, threshold: float) -> tuple[int, int, int]:
    candidates = sorted(((iou(box, other_box), i, j)
                         for i, (label, box) in enumerate(targets)
                         for j, (other_label, other_box) in enumerate(predictions) if label == other_label), reverse=True)
    matched_targets, matched_predictions = set(), set()
    for overlap, i, j in candidates:
        if overlap < threshold:
            break
        if i not in matched_targets and j not in matched_predictions:
            matched_targets.add(i)
            matched_predictions.add(j)
    tp = len(matched_targets)
    return tp, len(predictions) - tp, len(targets) - tp


def evaluate_florence(model_path: str, dataset_yaml: str, config: dict, batch_id: str, evaluation_attempt: int = 1):
    from .evaluation_adapter import EvaluationResult

    options = settings(config)
    fixed = options.get("evaluation_jsonl")
    metadata = Path(fixed) if fixed else Path(dataset_yaml).parent / "validation.jsonl"
    records = read_records(metadata)
    runtime = FlorenceRuntime(model_path, options)
    tp = fp = fn = 0
    predictions = []
    for record in records:
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
        result = runtime.predict(image)
        targets = [(label, tuple(value / 999 * image.size[i % 2] for i, value in enumerate(box)))
                   for label, box in parse_boxes(record["suffix"], strict=True)]
        counts = detection_counts(targets, list(zip(result["labels"], result["bboxes"])), options["iou_threshold"])
        tp, fp, fn = tp + counts[0], fp + counts[1], fn + counts[2]
        predictions.append({"image": record["image"], **result, "tp": counts[0], "fp": counts[1], "fn": counts[2]})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    evaluation = config.get("evaluation", {})
    thresholds = {"min_precision": evaluation.get("min_precision", .2), "min_recall": evaluation.get("min_recall", .2), "min_images": evaluation.get("min_images", 1)}
    passed = precision >= thresholds["min_precision"] and recall >= thresholds["min_recall"] and len(records) >= thresholds["min_images"]
    root = Path(evaluation.get("output_dir") or Path(config["paths"]["state_db"]).parent / "evaluations" / config["task_name"])
    folder = root / "florence" / batch_id / f"eval_{evaluation_attempt:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "evaluation.json"
    metrics = {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "images_evaluated": len(records)}
    report = {**metrics, "passed": passed, "model_backend": "florence", "model": model_path,
              "task": config["task_name"], "metric_type": "box", "iou_threshold": options["iou_threshold"],
              "evaluation_source": "fixed_dataset" if fixed else "batch_validation",
              "evaluation_jsonl": str(metadata), "thresholds": thresholds, "predictions": predictions}
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return EvaluationResult(passed, f"precision={precision:.4f}, recall={recall:.4f}, tp={tp}, fp={fp}, fn={fn}", str(path), metrics)


def infer_florence(config: dict, source: str, model_path: str, output: str, emit=lambda **event: None):
    from .demo_backend import image_files

    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Inference output must be outside the input folder")
    if not Path(model_path).is_dir():
        raise ValueError("Choose a Florence checkpoint folder containing config.json, processor files, and model weights")
    images = image_files(source, config["dataset"]["image_extensions"])
    relative_stems = [path.relative_to(source).with_suffix("").as_posix().casefold() for path in images]
    if len(set(relative_stems)) != len(images):
        raise ValueError("Images sharing a filename stem would overwrite inference labels")
    runtime = FlorenceRuntime(model_path, settings(config))
    output.mkdir(parents=True, exist_ok=False)
    records, errors = [], []
    for index, path in enumerate(images, 1):
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            prediction = runtime.predict(image)
            relative = path.relative_to(source)
            annotated = output / "images" / relative
            label = output / "labels" / relative.with_suffix(".json")
            annotated.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            draw = ImageDraw.Draw(image)
            for box, name in zip(prediction["bboxes"], prediction["labels"]):
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1]), name, fill="red")
            image.save(annotated)
            label.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
            records.append({"image": str(path), "annotated": str(annotated), "label": str(label)})
        except Exception as error:
            errors.append({"image": str(path), "error": str(error)})
        emit(kind="progress", current=index, total=len(images), message=f"Florence inference {index}/{len(images)}")
    report = {"model": str(model_path), "model_backend": "florence", "source": str(source), "task": config["task_name"],
              "processed": len(records), "total": len(images), "images": records, "errors": errors}
    (output / "inference.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not records:
        raise RuntimeError(f"All inference images failed. See {output / 'inference.json'}")
    return report
