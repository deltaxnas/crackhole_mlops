from __future__ import annotations

import random
import shutil
import logging
from pathlib import Path
from typing import Iterable
import yaml

log = logging.getLogger(__name__)


def _validate_label(label: Path, class_count: int, task_type: str) -> None:
    for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if task_type == "detect" and len(parts) != 5:
            raise ValueError(f"{label.name}:{number}: expected 5 detection columns")
        if task_type == "segment" and (len(parts) < 7 or len(parts[1:]) % 2):
            raise ValueError(
                f"{label.name}:{number}: expected a class and at least 3 segmentation points"
            )
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
        if not 0 <= class_id < class_count or any(not 0 <= value <= 1 for value in coords):
            raise ValueError(f"{label.name}:{number}: invalid class or normalized coordinates")


def validate_batch(
    batch: Path, class_count: int, extensions: set[str], task_type: str = "detect"
) -> list[tuple[Path, Path]]:
    images = sorted(p for p in (batch / "images").iterdir() if p.suffix.lower() in extensions)
    if not images:
        raise ValueError("batch contains no images")
    pairs = []
    for image in images:
        label = batch / "labels" / f"{image.stem}.txt"
        if not label.exists():
            raise ValueError(f"missing label: {label.name}; use an empty file for a negative image")
        _validate_label(label, class_count, task_type)
        pairs.append((image, label))
    return pairs


def _class_names(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        try:
            return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
        except (TypeError, ValueError) as error:
            raise ValueError("base dataset class names must use numeric IDs") from error
    raise ValueError("base dataset YAML must contain class names")


def _entry_images(entry, dataset_root: Path, yaml_path: Path, extensions: set[str]) -> Iterable[Path]:
    entries = entry if isinstance(entry, list) else [entry]
    for value in entries:
        path = Path(value)
        path = path if path.is_absolute() else dataset_root / path
        if path.is_dir():
            yield from sorted(
                item for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in extensions
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                image = Path(line.strip())
                if not line.strip():
                    continue
                image = image if image.is_absolute() else path.parent / image
                if image.suffix.lower() in extensions:
                    yield image
        else:
            raise FileNotFoundError(
                f"base dataset entry from {yaml_path.name} does not exist: {path}"
            )


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"base dataset image is not under an images directory: {image}") from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _relative_to_images(image: Path) -> Path:
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    relative = Path(*parts[index + 1:])
    if relative.parts and relative.parts[0] in ("train", "val"):
        relative = Path(*relative.parts[1:])
    return relative


def _load_base_pairs(
    base_yaml: str | Path,
    expected_names: list[str],
    extensions: set[str],
    task_type: str,
) -> dict[str, list[tuple[Path, Path, Path]]]:
    yaml_path = Path(base_yaml).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"base dataset YAML not found: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if _class_names(data) != expected_names:
        raise ValueError(
            f"base dataset classes {_class_names(data)!r} do not match task classes {expected_names!r}"
        )
    configured_root = Path(data.get("path", yaml_path.parent))
    dataset_root = (
        configured_root if configured_root.is_absolute()
        else (yaml_path.parent / configured_root).resolve()
    )
    result: dict[str, list[tuple[Path, Path, Path]]] = {"train": [], "val": []}
    for split in result:
        if split not in data:
            raise ValueError(f"base dataset YAML is missing {split!r}")
        for image in _entry_images(data[split], dataset_root, yaml_path, extensions):
            if not image.is_file():
                raise FileNotFoundError(f"base dataset image not found: {image}")
            label = _label_path(image)
            if label.exists():
                _validate_label(label, len(expected_names), task_type)
            result[split].append((image, label, _relative_to_images(image)))
    return result


def _copy_pair(image: Path, label: Path, image_target: Path, label_target: Path) -> None:
    if image_target.exists() or label_target.exists():
        raise ValueError(f"dataset filename collision: {image_target}")
    image_target.parent.mkdir(parents=True, exist_ok=True)
    label_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_target)
    if label.exists():
        shutil.copy2(label, label_target)
    else:
        label_target.touch()  # Missing YOLO labels represent a negative image.


def build_dataset(
    config: dict,
    batch_id: str,
    batch_path: str,
    base_dataset_yaml: str | Path | None = None,
) -> Path:
    names = config["dataset"]["class_names"]
    extensions = {x.lower() for x in config["dataset"]["image_extensions"]}
    pairs = validate_batch(
        Path(batch_path), len(names), extensions, config.get("task_type", "detect")
    )
    random.Random(config["dataset"]["seed"]).shuffle(pairs)
    val_count = max(1, round(len(pairs) * config["dataset"]["val_fraction"])) if len(pairs) > 1 else 0
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}
    task_type = config.get("task_type", "detect")
    base = (
        _load_base_pairs(base_dataset_yaml, names, extensions, task_type)
        if base_dataset_yaml else {"train": [], "val": []}
    )
    log.info(
        "[%s] Labels validated | base=%d | incoming=%d | train=%d | val=%d",
        batch_id,
        sum(len(items) for items in base.values()),
        len(pairs),
        len(base["train"]) + len(splits["train"]),
        len(base["val"]) + len(splits["val"]),
    )
    root = Path(config["paths"]["datasets"]) / batch_id
    if any(
        image == root or root in image.parents
        for split_items in base.values()
        for image, _, _ in split_items
    ):
        raise ValueError(
            f"base dataset points inside output dataset {root}; use a separate seed dataset"
        )
    if root.exists():
        shutil.rmtree(root)
    for split, items in splits.items():
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for image, label, relative in base[split]:
            _copy_pair(
                image, label,
                root / "images" / split / relative,
                root / "labels" / split / relative.with_suffix(".txt"),
            )
        for image, label in items:
            relative = Path(image.name)
            if (root / "images" / split / relative).exists():
                relative = Path("batches") / batch_id / image.name
            _copy_pair(
                image, label,
                root / "images" / split / relative,
                root / "labels" / split / relative.with_suffix(".txt"),
            )
    data = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "names": names}
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    log.info("[%s] Dataset and data.yaml created | path=%s", batch_id, root)
    return root
