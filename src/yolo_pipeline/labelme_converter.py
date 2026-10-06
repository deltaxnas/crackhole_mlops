"""Basic single-class Labelme converter for panel segmentation and holes.

By default every rectangle/polygon is class 0. Set preprocessing.labels to a
list of Labelme label names to select only some shapes. Empty annotations are
valid negatives; malformed or unsupported annotations fail the entire batch.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from shapely.geometry import Polygon


def shape_points(shape: dict, width: int, height: int) -> list[tuple[float, float]]:
    kind = shape.get("shape_type", "polygon")
    if kind not in {"rectangle", "polygon"}:
        raise ValueError(f"unsupported Labelme shape {kind!r}; use polygons or rectangles")
    points = shape.get("points", [])
    if kind == "rectangle":
        if len(points) != 2:
            raise ValueError("a rectangle must have exactly two corners")
        (x1, y1), (x2, y2) = points
        points = [(min(x1, x2), min(y1, y2)), (max(x1, x2), min(y1, y2)),
                  (max(x1, x2), max(y1, y2)), (min(x1, x2), max(y1, y2))]
    if len(points) < 3:
        raise ValueError("a polygon must have at least three points")
    converted = []
    for point in points:
        if len(point) != 2:
            raise ValueError("each point needs x and y coordinates")
        x, y = map(float, point)
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
            raise ValueError("annotation coordinates must lie within the image")
        converted.append((x, y))
    polygon = Polygon(converted)
    if not polygon.is_valid or polygon.area <= 0:
        raise ValueError("annotation polygon is invalid or has zero area")
    return converted


def convert_labelme_batch(config: dict, batch_path: Path) -> Path:
    batch = Path(batch_path)
    if len(config["dataset"]["class_names"]) != 1:
        raise ValueError("the basic Labelme converter requires exactly one class name")
    task_type = config["task_type"]
    if task_type not in {"detect", "segment"}:
        raise ValueError(f"unsupported YOLO task type: {task_type}")
    extensions = {ext.lower() for ext in config["dataset"]["image_extensions"]}
    images = sorted(p for p in batch.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not images:
        raise ValueError("batch contains no images")
    if len({p.stem.casefold() for p in images}) != len(images):
        raise ValueError("image stems must be unique")
    accepted = config.get("preprocessing", {}).get("labels")
    if accepted is not None and (not isinstance(accepted, list) or not accepted):
        raise ValueError("preprocessing.labels must be a nonempty list or null (all shapes)")
    accepted = {str(label) for label in accepted} if accepted is not None else None
    output = batch / "converted_yolo"
    if output.is_symlink():
        raise ValueError("converted output must not be a symlink")
    with TemporaryDirectory(prefix=".labelme-convert-", dir=batch) as temporary:
        staging = Path(temporary) / "new"
        (staging / "images").mkdir(parents=True)
        (staging / "labels").mkdir()
        for image in images:
            try:
                if image.is_symlink() or image.with_suffix(".json").is_symlink():
                    raise ValueError("image/annotation symlinks are not supported")
                decoded = cv2.imread(str(image))
                if decoded is None:
                    raise ValueError("image could not be decoded")
                height, width = decoded.shape[:2]
                data = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
                    raise ValueError("expected a Labelme shapes list")
                if data.get("imageWidth", width) != width or data.get("imageHeight", height) != height:
                    raise ValueError("annotation image dimensions do not match the actual image")
                lines = []
                for shape in data["shapes"]:
                    if accepted is not None and str(shape["label"]) not in accepted:
                        continue
                    points = shape_points(shape, width, height)
                    if task_type == "segment":
                        values = [value for x, y in points for value in (x / width, y / height)]
                    else:
                        xs, ys = zip(*points)
                        values = [(min(xs) + max(xs)) / (2 * width), (min(ys) + max(ys)) / (2 * height),
                                  (max(xs) - min(xs)) / width, (max(ys) - min(ys)) / height]
                    lines.append("0 " + " ".join(f"{value:.8f}" for value in values) + "\n")
                shutil.copy2(image, staging / "images" / image.name)
                (staging / "labels" / f"{image.stem}.txt").write_text("".join(lines), encoding="utf-8")
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{image.name}: {error}") from error
        # Publish a fresh conversion only once all annotations pass, so retries
        # cannot train on leftover or partly converted examples.
        old = Path(temporary) / "old"
        if output.exists():
            output.rename(old)
        try:
            staging.rename(output)
        except OSError:
            if old.exists():
                old.rename(output)
            raise
    return output
