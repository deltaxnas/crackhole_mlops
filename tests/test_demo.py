"""Demo integration tests use temporary data and substitute training/inference."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from yolo_pipeline.demo_backend import (
    DemoStore, TASK, demo_config, import_batch, leaderboard, run_inference,
    run_training, training_history, validate_labeling, worker_lock,
)
from yolo_pipeline.evaluation_adapter import EvaluationResult
from yolo_pipeline.models import State


@pytest.fixture
def config(tmp_path):
    return {
        "paths": {**{name: str(tmp_path / name) for name in ("incoming", "labeling", "datasets", "runs")},
                  "state_db": str(tmp_path / "state.db")},
        "dataset": {"image_extensions": [".jpg", ".png"], "seed": 42, "val_fraction": 0.2},
        "training": {"device": "cpu", "epochs": 3, "imgsz": 32, "batch": 1},
        "tasks": {TASK: {
            "done_marker": "DONE_crack_det", "type": "detect",
            "dataset": {"class_names": ["no_crack", "crack"]},
            "training": {"model": "unused.pt"},
            "evaluation": {"min_precision": 0.8, "min_recall": 0.8,
                           "comparison": {"enabled": False}},
            "deployment": {"destination": str(tmp_path / "deployment")},
        }},
    }


def source_batch(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "panel.jpg").write_bytes(b"fake image")
    (source / "panel.json").write_text('{"shapes": []}')
    return source


def test_import_is_manual_preserves_annotations_and_never_polls(config, tmp_path):
    source = source_batch(tmp_path)
    first = import_batch(config, str(source))
    second = import_batch(config, str(source))
    assert first.batch_id != second.batch_id
    assert first.state == State.LABELING
    assert first.task == TASK
    assert not Path(config["paths"]["incoming"]).exists()
    assert (Path(first.batch_path) / "panel.json").read_text() == '{"shapes": []}'
    assert (source / "panel.jpg").read_bytes() == b"fake image"
    assert not (Path(first.batch_path) / "DONE_crack_det").exists()


def test_import_rejects_annotation_collisions(config, tmp_path):
    source = source_batch(tmp_path)
    (source / "panel.png").write_bytes(b"second image")
    with pytest.raises(ValueError, match="stems"):
        import_batch(config, str(source))


def test_labeling_requires_saved_annotations_for_every_image(config, tmp_path):
    source = source_batch(tmp_path)
    (source / "unlabeled.jpg").write_bytes(b"image")
    with pytest.raises(ValueError, match="unlabeled.jpg"):
        validate_labeling(config, source)


def add_result(store, tmp_path, batch_id, p, r, state=State.APPROVED):
    report = tmp_path / f"{batch_id}.json"
    report.write_text(json.dumps({"precision": p, "recall": r, "passed": state == State.APPROVED}))
    model = tmp_path / f"{batch_id}.pt"
    model.write_bytes(b"weights")
    run = store.create(batch_id, str(tmp_path / batch_id))
    store.transition(run.id, State.NEW_DATA, state, task=TASK,
                     model_path=str(model), evaluation_path=str(report))
    return store.get(run.id)


def test_leaderboard_uses_best_approved_f1_not_discovery_order(config, tmp_path):
    store = DemoStore(config["paths"]["state_db"])
    best = add_result(store, tmp_path, "best", .8, .8)
    add_result(store, tmp_path, "newer", .9, .5)
    add_result(store, tmp_path, "rejected", .99, .99, State.REJECTED)
    rows = leaderboard(store)
    assert rows[0]["run"].batch_id == "rejected"
    assert store.previous_approved(TASK, 999).id == best.id
    assert store.previous_approved(TASK, best.id).batch_id == "newer"


def test_missing_approved_report_cannot_bypass_comparison(config, tmp_path):
    store = DemoStore(config["paths"]["state_db"])
    run = add_result(store, tmp_path, "approved", .8, .8)
    Path(run.evaluation_path).unlink()
    assert leaderboard(store)[0]["error"]
    with pytest.raises(ValueError, match="Cannot compare"):
        store.previous_approved(TASK, 999)


def test_demo_thresholds_do_not_mutate_original_configuration(config):
    adjusted = demo_config(config)
    assert adjusted["tasks"][TASK]["evaluation"]["min_precision"] == .2
    assert adjusted["tasks"][TASK]["evaluation"]["min_recall"] == .2
    assert config["tasks"][TASK]["evaluation"]["min_precision"] == .8
    assert not config["tasks"][TASK]["evaluation"]["comparison"]["enabled"]


@pytest.mark.parametrize("precision,recall,baseline,expected", [
    (.2, .2, None, State.APPROVED),
    (.19, .9, None, State.REJECTED),
    (.9, .19, None, State.REJECTED),
    (.75, .9, (.8, .8), State.REJECTED),
    (.85, .85, (.8, .8), State.APPROVED),
])
def test_done_to_evaluation_uses_demo_floor_and_baseline(config, tmp_path, monkeypatch, precision, recall, baseline, expected):
    from yolo_pipeline import orchestrator

    store = DemoStore(config["paths"]["state_db"])
    if baseline:
        add_result(store, tmp_path, "baseline", *baseline)
    run = import_batch(config, str(source_batch(tmp_path)))
    unrelated = store.create("other", str(tmp_path / "other"))
    store.transition(unrelated.id, State.NEW_DATA, State.TRAINING, task=TASK)
    dataset = tmp_path / "converted"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: [no_crack, crack]\n")
    model = tmp_path / "best.pt"
    model.write_bytes(b"trained weights")
    monkeypatch.setattr(orchestrator, "preprocess_labelme_batch", lambda *a: dataset)
    monkeypatch.setattr(orchestrator, "build_dataset", lambda *a, **kw: dataset)
    monkeypatch.setattr(orchestrator, "train_yolo", lambda *a: model)

    def evaluate(model_path, yaml_path, task_config, batch_id, attempt):
        options = task_config["evaluation"]
        passed = precision >= options["min_precision"] and recall >= options["min_recall"]
        report = tmp_path / "candidate.json"
        report.write_text(json.dumps({"precision": precision, "recall": recall, "passed": passed,
                                      "thresholds": {"min_precision": options["min_precision"],
                                                     "min_recall": options["min_recall"]}}))
        return EvaluationResult(passed, "test evaluation", str(report), {"precision": precision, "recall": recall})

    monkeypatch.setattr(orchestrator, "evaluate", evaluate)
    events = []
    run_training(config, run.batch_id, lambda **event: events.append(event))
    result = store.get(run.id)
    assert result.state == expected
    assert (Path(run.batch_path) / "DONE_crack_det").is_file()
    assert bool(result.deployment_path) == (expected == State.APPROVED)
    assert store.get(unrelated.id).state == State.TRAINING
    assert events[-1]["state"] == expected.value
    assert json.loads(Path(result.evaluation_path).read_text())["passed"] == (expected == State.APPROVED)


def test_csv_progress_handles_partial_writer_row(config, tmp_path):
    store = DemoStore(config["paths"]["state_db"])
    run = add_result(store, tmp_path, "progress", .5, .5)
    folder = Path(config["paths"]["runs"]) / run.batch_id / "train_001"
    folder.mkdir(parents=True)
    (folder / "results.csv").write_text(" epoch, train/box_loss, val/box_loss\n1, 2.0, 3.0\n2, 1.5, 2.0\n3,")
    rows = training_history(config, run)
    assert len(rows) == 2
    assert rows[-1]["train/box_loss"] == 1.5


def test_inference_preserves_subdirectories_and_records_errors(config, tmp_path, monkeypatch):
    source = source_batch(tmp_path)
    nested = source / "nested"
    nested.mkdir()
    (nested / "panel.jpg").write_bytes(b"nested")
    (source / "bad.jpg").write_bytes(b"bad")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model")
    output = tmp_path / "results"

    class FakeModel:
        def __init__(self, path):
            assert path == str(model)

        def predict(self, path, **kwargs):
            if Path(path).name == "bad.jpg":
                raise ValueError("unreadable image")
            return [SimpleNamespace(
                save=lambda filename: Path(filename).write_bytes(b"annotated"),
                save_txt=lambda filename, save_conf: Path(filename).write_text("0 .5 .5 .2 .2 .9\n"),
            )]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeModel))
    report = run_inference(config, str(source), str(model), str(output))
    assert report["processed"] == 2
    assert len(report["errors"]) == 1
    assert (output / "images" / "nested" / "panel.jpg").is_file()
    assert (output / "labels" / "panel.txt").is_file()
    assert (output / "inference.json").is_file()
    assert (source / "panel.jpg").read_bytes() == b"fake image"


def test_demo_workers_are_exclusive(config):
    with worker_lock(config["paths"]["state_db"]):
        with pytest.raises(RuntimeError, match="Another demo job"):
            with worker_lock(config["paths"]["state_db"]):
                pass
    with worker_lock(config["paths"]["state_db"]):
        pass
