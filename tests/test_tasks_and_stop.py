import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from yolo_pipeline.config import for_task
from yolo_pipeline.demo_backend import DemoStore, demo_config, import_batch, leaderboard
from yolo_pipeline.labelme_converter import convert_labelme_batch
from yolo_pipeline.models import State
from yolo_pipeline.preprocessing import preprocess_labelme_batch
from yolo_pipeline.training import train_yolo
from yolo_pipeline.yolo_evaluation import evaluate_yolo_model


@pytest.fixture
def task_config(tmp_path):
    return {
        "task_name": "panel_segmentation", "task_type": "segment",
        "paths": {"state_db": str(tmp_path / "state.db"), "runs": str(tmp_path / "runs"),
                  "labeling": str(tmp_path / "labeling"), "datasets": str(tmp_path / "datasets")},
        "dataset": {"class_names": ["panel"], "image_extensions": [".png"], "val_fraction": .5, "seed": 42},
        "training": {"model": "model.pt", "device": "cpu", "epochs": 5, "imgsz": 64, "batch": 2},
        "evaluation": {"min_precision": .2, "min_recall": .2},
    }


def source(tmp_path, shapes):
    folder = tmp_path / "source"
    folder.mkdir(exist_ok=True)
    assert cv2.imwrite(str(folder / "image.png"), np.full((50, 100, 3), 128, dtype=np.uint8))
    (folder / "image.json").write_text(json.dumps({"imageWidth": 100, "imageHeight": 50, "shapes": shapes}))
    return folder


def polygon(label="any label"):
    return {"label": label, "shape_type": "polygon", "points": [[10, 5], [90, 5], [90, 45], [10, 45]]}


@pytest.mark.parametrize("shape", [polygon(), {"label": "rectangle", "shape_type": "rectangle", "points": [[90, 45], [10, 5]]}])
def test_single_class_segmentation_normalizes_outline(task_config, tmp_path, shape):
    folder = source(tmp_path, [shape])
    output = preprocess_labelme_batch(task_config, "panel_segmentation", str(folder))
    values = (output / "labels/image.txt").read_text().split()
    assert values[0] == "0"
    assert list(map(float, values[1:])) == pytest.approx([.1, .1, .9, .1, .9, .9, .1, .9])
    assert (output / "images/image.png").read_bytes() == (folder / "image.png").read_bytes()


def test_hole_detection_converts_polygon_to_bbox(task_config, tmp_path):
    task_config.update(task_type="detect", task_name="hole_detection")
    output = preprocess_labelme_batch(task_config, "hole_detection", str(source(tmp_path, [polygon()])))
    values = (output / "labels/image.txt").read_text().split()
    assert list(map(float, values)) == pytest.approx([0, .5, .5, .8, .8])


def test_empty_annotation_is_retained_as_negative(task_config, tmp_path):
    output = convert_labelme_batch(task_config, source(tmp_path, []))
    assert (output / "labels/image.txt").read_text() == ""
    assert (output / "images/image.png").exists()


def test_optional_label_filter_and_single_class_enforced(task_config, tmp_path):
    folder = source(tmp_path, [polygon("panel"), polygon("crack")])
    task_config["preprocessing"] = {"labels": ["panel"]}
    output = convert_labelme_batch(task_config, folder)
    assert len((output / "labels/image.txt").read_text().splitlines()) == 1
    task_config["dataset"]["class_names"] = ["one", "two"]
    with pytest.raises(ValueError, match="exactly one"):
        convert_labelme_batch(task_config, folder)


@pytest.mark.parametrize("shape", [
    {"shape_type": "circle", "points": [[10, 10], [20, 20]]},
    {"shape_type": "polygon", "points": [[-1, 0], [30, 0], [30, 30]]},
    {"shape_type": "polygon", "points": [[0, 0], [10, 10], [20, 20]]},
    {"shape_type": "polygon", "points": [[0, 0], [float("nan"), 20], [20, 20]]},
])
def test_invalid_shapes_fail_without_partial_conversion(task_config, tmp_path, shape):
    folder = source(tmp_path, [shape])
    with pytest.raises(ValueError):
        convert_labelme_batch(task_config, folder)
    assert not (folder / "converted_yolo").exists()


def test_bad_reprocess_preserves_previous_conversion(task_config, tmp_path):
    folder = source(tmp_path, [polygon()])
    output = convert_labelme_batch(task_config, folder)
    previous = (output / "labels/image.txt").read_bytes()
    (folder / "image.json").write_text("malformed")
    with pytest.raises(ValueError):
        convert_labelme_batch(task_config, folder)
    assert (output / "labels/image.txt").read_bytes() == previous


def test_new_tasks_keep_independent_baselines(task_config, tmp_path):
    config = {**task_config, "tasks": {
        "panel_segmentation": {"type": "segment", "done_marker": "DONE_panel_seg"},
        "hole_detection": {"type": "detect", "done_marker": "DONE_hole_det"},
    }}
    folder = source(tmp_path, [polygon()])
    panel = import_batch(config, str(folder), task_name="panel_segmentation")
    hole = import_batch(config, str(folder), task_name="hole_detection")
    store = DemoStore(config["paths"]["state_db"])
    report = tmp_path / "panel-report.json"
    report.write_text('{"precision": .8, "recall": .8}'.replace('.8', '0.8'))
    store.transition(panel.id, State.LABELING, State.APPROVED, evaluation_path=str(report))
    assert leaderboard(store, "panel_segmentation")[0]["run"].id == panel.id
    assert leaderboard(store, "hole_detection") == []
    assert store.previous_approved("hole_detection", hole.id) is None
    assert demo_config(config, "panel_segmentation")["tasks"]["panel_segmentation"]["evaluation"]["min_recall"] == .2
    config["tasks"]["panel_segmentation"]["preprocessing"] = {"labels": ["500"]}
    assert for_task(config, "panel_segmentation")["preprocessing"]["labels"] == ["500"]


@pytest.mark.parametrize("checkpoint", ["best.pt", "last.pt", None])
def test_stop_saves_completed_epoch_and_uses_checkpoint(task_config, tmp_path, monkeypatch, checkpoint):
    from yolo_pipeline import training

    marker = tmp_path / "STOP"
    task_config["training"]["stop_file"] = str(marker)
    task_config["training"]["extra"] = {"save": False}
    models = []

    class FakeYOLO:
        def __init__(self, name):
            self.callbacks = {}
            models.append(self)

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **kwargs):
            assert kwargs["save"] is True
            folder = Path(kwargs["project"]) / kwargs["name"]
            (folder / "weights").mkdir(parents=True)
            self.trainer = SimpleNamespace(epoch=0, stop=False, save_dir=folder)
            for epoch in range(kwargs["epochs"]):
                self.trainer.epoch = epoch
                if checkpoint:
                    (folder / "weights" / checkpoint).write_bytes(f"saved epoch {epoch + 1}".encode())
                if epoch == 1:
                    marker.touch()
                self.callbacks["on_fit_epoch_end"](self.trainer)
                if self.trainer.stop:
                    break
            return SimpleNamespace(save_dir=folder)

    monkeypatch.setattr(training, "YOLO", FakeYOLO)
    if checkpoint is None:
        with pytest.raises(FileNotFoundError):
            train_yolo(task_config, "batch", str(tmp_path))
    else:
        result = train_yolo(task_config, "batch", str(tmp_path))
        assert result.name == checkpoint
        assert result.read_bytes() == b"saved epoch 2"
        summary = json.loads((result.parent.parent / "training_summary.json").read_text())
        assert summary["stop_requested"]
        assert summary["epochs_completed"] == 2
        assert summary["requested_epochs"] == 5
    assert models[0].trainer.stop


@pytest.mark.parametrize("task_type", ["segment", "detect"])
def test_evaluation_selects_mask_or_box_metrics(task_config, tmp_path, monkeypatch, task_type):
    import ultralytics

    task_config["task_type"] = task_type
    dataset = tmp_path / "dataset"
    (dataset / "images/val").mkdir(parents=True)
    (dataset / "labels/val").mkdir(parents=True)
    (dataset / "images/val/test.png").write_bytes(b"image")
    label = "0 .1 .1 .9 .1 .9 .9\n" if task_type == "segment" else "0 .5 .5 .8 .8\n"
    (dataset / "labels/val/test.txt").write_text(label)
    yaml_path = dataset / "data.yaml"
    yaml_path.write_text(f"path: {dataset}\nval: images/val\nnames: [panel]\n")

    class FakeYOLO:
        task = task_type

        def __init__(self, path):
            pass

        def add_callback(self, event, callback):
            self.finished = callback

        def val(self, **kwargs):
            self.finished(SimpleNamespace(seen=1))
            return SimpleNamespace(seg=SimpleNamespace(mp=.8, mr=.7, map50=.6, map=.4),
                                   box=SimpleNamespace(mp=.9, mr=.85, map50=.8, map=.6))

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    result = evaluate_yolo_model("unused.pt", str(yaml_path), task_config, "batch")
    report = json.loads(Path(result.report_path).read_text())
    assert result.passed
    assert result.metrics["precision"] == (.8 if task_type == "segment" else .9)
    assert report["metric_type"] == ("mask" if task_type == "segment" else "box")
    assert report["evaluation_source"] == "batch_validation"
