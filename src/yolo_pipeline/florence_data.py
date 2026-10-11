"""Labelme to Florence <OD> records, adapted from the colleague's converter.

Images are copied into immutable dataset versions. Duplicate images stay in one
split; grouping by capture is optional and disabled by default.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image

log = logging.getLogger(__name__)
CAPTURE = re.compile(r"D(\d{8}T\d{6})_P(\d+)_S\d+")
BOX = re.compile(r"([^<>]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>")


def parse_boxes(text: str, *, strict: bool = False) -> list[tuple[str, tuple[float, ...]]]:
    matches = list(BOX.finditer(text))
    if strict and "".join(match.group(0) for match in matches) != text:
        raise ValueError(f"Invalid Florence detection target: {text!r}")
    boxes = []
    for match in matches:
        label = match.group(1).strip()
        box = tuple(float(value) for value in match.groups()[1:])
        if not label or any(not 0 <= value <= 999 for value in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Invalid Florence box or class label")
        boxes.append((label, box))
    return boxes


def read_records(metadata: str | Path) -> list[dict]:
    metadata = Path(metadata).resolve()
    records = []
    seen = set()
    for index, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            image = (metadata.parent / item["image"]).resolve()
            if not image.is_relative_to(metadata.parent) or not image.is_file():
                raise ValueError("image must exist inside the JSONL dataset directory")
            if image in seen:
                raise ValueError("duplicate image record")
            if item["prefix"] != "<OD>" or not isinstance(item["suffix"], str):
                raise ValueError("expected <OD> prefix and string suffix")
            boxes = parse_boxes(item["suffix"], strict=True)
            if any(label != "crack" for label, _ in boxes):
                raise ValueError("Florence crack detection requires the class 'crack'")
            seen.add(image)
            records.append({**item, "image": str(image)})
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{metadata}:{index}: {error}") from error
    if not records:
        raise ValueError(f"No records in {metadata}")
    return records


def image_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_group(image: Path, digest: str) -> str:
    match = CAPTURE.search(image.stem)
    return f"P{match[2]}:{match[1]}" if match else f"image:{digest}"


def shape_box(shape: dict, width: int, height: int) -> tuple[float, float, float, float]:
    """Validate OD coordinates without imposing segmentation polygon topology."""
    kind = shape.get("shape_type", "polygon")
    points = shape.get("points", [])
    if kind not in {"rectangle", "polygon"}:
        raise ValueError(f"unsupported shape {kind!r}; use rectangles or polygons")
    if not isinstance(points, list) or (kind == "rectangle" and len(points) != 2) or (kind == "polygon" and len(points) < 3):
        raise ValueError("a rectangle needs two corners; a polygon needs at least three points")
    converted = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each point needs x and y coordinates")
        x, y = map(float, point)
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
            raise ValueError("annotation coordinates must lie within the image")
        converted.append((x, y))
    xs, ys = zip(*converted)
    return min(xs), min(ys), max(xs), max(ys)


def labelme_record(image: Path, selected_labels: list[str] | None) -> dict:
    annotation = image.with_suffix(".json")
    data = json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        raise ValueError(f"Expected a Labelme shapes list: {annotation}")
    with Image.open(image) as source:
        width, height = source.size
        source.load()
    for key, actual in (("imageWidth", width), ("imageHeight", height)):
        if data.get(key) is not None and data[key] != actual:
            raise ValueError(f"{annotation}: {key} does not match image")
    targets = []
    skipped = []
    for index, shape in enumerate(data["shapes"], 1):
        context = f"{annotation}: shape {index}"
        if not isinstance(shape, dict):
            raise ValueError(f"{context}: expected a Labelme shape object")
        if selected_labels is not None and str(shape.get("label")) not in selected_labels:
            continue
        try:
            x1, y1, x2, y2 = shape_box(shape, width, height)
            if x1 == x2 or y1 == y2:
                skipped.append(index)
                log.warning("%s: skipping zero-area bounding box (label=%r)", context, shape.get("label"))
                continue
            coordinates = [x1 / width, y1 / height, x2 / width, y2 / height]
            target = "crack" + "".join(f"<loc_{round(value * 999)}>" for value in coordinates)
            parse_boxes(target, strict=True)
            targets.append(target)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{context} (label={shape.get('label')!r}): {error}") from error
    if skipped and not targets:
        raise ValueError(
            f"{annotation}: all selected boxes have zero area (shapes {skipped}); "
            "fix these annotations, or explicitly save shapes: [] if this is a reviewed negative image"
        )
    suffix = "".join(sorted(targets))
    parse_boxes(suffix, strict=True)
    digest = image_digest(image)
    return {"image": str(image), "prefix": "<OD>", "suffix": suffix,
            "digest": digest, "group": capture_group(image, digest), "skipped_zero_area_shapes": skipped}


def build_florence_dataset(config: dict, batch_id: str, batch_path: str, base_dir: str | None = None) -> Path:
    options = config.get("florence", {})
    group_by_capture = options.get("group_by_capture", False)
    if not isinstance(group_by_capture, bool):
        raise ValueError("florence.group_by_capture must be true or false")
    labels = options.get("labels")
    if labels is not None and (not isinstance(labels, list) or not labels or not all(isinstance(x, str) for x in labels)):
        raise ValueError("florence.labels must be null or a nonempty list of label strings")
    extensions = {ext.lower() for ext in config["dataset"]["image_extensions"]}
    source = Path(batch_path)
    images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in extensions)
    if not images:
        raise ValueError("No source images for Florence conversion")
    root = Path(config["paths"]["datasets"]) / batch_id
    splits = {"train": [], "validation": []}
    known, inherited = {}, {}
    if base_dir:
        base = Path(base_dir).resolve()
        if base == root.resolve() or root.resolve() in base.parents:
            raise ValueError("Florence base dataset cannot be inside the output dataset")
        for split in splits:
            for record in read_records(base / f"{split}.jsonl"):
                path = Path(record["image"])
                digest = image_digest(path)
                if digest in known:
                    raise ValueError("Base dataset contains duplicate images")
                group = record.get("group", capture_group(path, digest))
                aliases = record.get("groups", [group])
                if not isinstance(aliases, list) or not aliases or not all(isinstance(value, str) for value in aliases):
                    raise ValueError("Invalid capture groups in Florence base dataset")
                record.update(digest=digest, group=group, groups=aliases)
                known[digest] = record
                inherited[digest] = split
    for image in images:
        record = labelme_record(image, labels)
        digest, group = record["digest"], record["group"]
        if digest in known:
            if record["suffix"] != known[digest]["suffix"]:
                raise ValueError(f"Conflicting annotations for duplicate image: {image.name}")
            # A renamed duplicate can connect two capture groups. Preserve every
            # alias so other images from those captures cannot leak across splits.
            known[digest]["groups"] = sorted(set(known[digest]["groups"]) | {group})
        else:
            record["groups"] = [group]
            known[digest] = record
    parents = {}

    def representative(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    for record in known.values():
        aliases = record["groups"]
        for alias in aliases[1:]:
            parents[representative(alias)] = representative(aliases[0])

    def split_group(record):
        return representative(record["groups"][0]) if group_by_capture else f"image:{record['digest']}"

    groups = {}
    for digest, split in inherited.items():
        group = split_group(known[digest])
        if group in groups and groups[group] != split:
            raise ValueError("Capture groups span existing splits; rebuild the seed dataset or set florence.group_by_capture: false")
        groups[group] = split
    pending = {}
    for record in known.values():
        group = split_group(record)
        record["group"] = group
        if group in groups:
            splits[groups[group]].append(record)
        else:
            pending.setdefault(group, []).append(record)
    fraction = float(config["dataset"]["val_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("dataset.val_fraction must be between 0 and 1")
    names = sorted(pending)
    random.Random(config["dataset"]["seed"]).shuffle(names)
    total = sum(map(len, splits.values())) + sum(map(len, pending.values()))
    target = max(1, round(total * fraction))
    for index, group in enumerate(names):
        # Keep at least one complete group for training.
        remaining = len(names) - index
        use_validation = len(splits["validation"]) < target and (bool(splits["train"]) or remaining > 1)
        splits["validation" if use_validation else "train"].extend(pending[group])
    if not all(splits.values()):
        unit = "capture groups (or set florence.group_by_capture: false)" if group_by_capture else "images"
        raise ValueError(f"Florence needs nonempty train and validation splits. Supply at least two distinct {unit} or a base_dataset_dir.")
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".florence-data-", dir=root.parent) as tmp:
        staged = Path(tmp) / "dataset"
        staged.mkdir()
        for split, records in splits.items():
            image_dir = staged / split / "images"
            image_dir.mkdir(parents=True)
            with (staged / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    image = Path(record["image"])
                    relative = Path(split) / "images" / f"{record['digest']}{image.suffix.lower()}"
                    shutil.copy2(image, staged / relative)
                    handle.write(json.dumps({**record, "image": relative.as_posix()}) + "\n")
        (staged / "classes.json").write_text('{"0": "crack"}\n', encoding="utf-8")
        (staged / "split_summary.json").write_text(json.dumps({
            "train": len(splits["train"]), "validation": len(splits["validation"]),
            "model_backend": "florence", "base_dataset_dir": base_dir,
            "labels": labels, "group_by_capture": group_by_capture,
            "split_group": "capture and position when present, otherwise image hash" if group_by_capture else "image hash",
        }, indent=2), encoding="utf-8")
        backup = root.with_name(f".{root.name}.{uuid4().hex}.backup")
        if root.exists():
            root.rename(backup)
        try:
            staged.rename(root)
        except Exception:
            if backup.exists():
                backup.rename(root)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    return root
