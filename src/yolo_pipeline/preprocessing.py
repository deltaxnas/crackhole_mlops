from __future__ import annotations

from pathlib import Path
from typing import Callable


def preprocess_crack_detection(config: dict, batch_path: Path) -> Path:
    """Convert crack Labelme annotations to an images/ + labels/ YOLO batch."""
    from .crack_det_pre import create_yolo_dataset

    print("Preprocessing crack detection dataset")
    create_yolo_dataset(batch_path, config)
    return batch_path


def preprocess_hole_detection(config: dict, batch_path: Path) -> Path:
    """Convert hole Labelme annotations to an images/ + labels/ YOLO batch."""
    from .labelme_converter import convert_labelme_batch

    return convert_labelme_batch(config, batch_path)


def preprocess_panel_segmentation(config: dict, batch_path: Path) -> Path:
    """Convert panel polygons to an images/ + labels/ YOLO segmentation batch."""
    from .labelme_converter import convert_labelme_batch

    return convert_labelme_batch(config, batch_path)


PREPROCESSORS: dict[str, Callable[[dict, Path], Path]] = {
    "crack_detection": preprocess_crack_detection,
    "hole_detection": preprocess_hole_detection,
    "panel_segmentation": preprocess_panel_segmentation,
}


def preprocess_labelme_batch(config: dict, task_name: str, batch_path: str) -> Path:
    """Dispatch to the preprocessing hook for the selected task."""
    try:
        preprocessor = PREPROCESSORS[task_name]
    except KeyError as error:
        raise ValueError(f"no preprocessor registered for task: {task_name}") from error
    return preprocessor(config, Path(batch_path))
