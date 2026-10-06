import json
import shutil
import sys
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
import yaml

from yolo_pipeline import cli
from yolo_pipeline.models import State
from yolo_pipeline.store import StateStore
from yolo_pipeline.watcher import DatasetWatcher


@pytest.fixture
def setup_pipeline(tmp_path):
    config = {
        "paths": {key: str(tmp_path / key) for key in ("incoming", "labeling", "datasets", "runs")},
        "dataset": {"image_extensions": [".jpg", ".png"]},
        "watcher": {"settle_seconds": 0},
        "orchestrator": {"poll_seconds": 5},
        "tasks": {},
    }
    config["paths"]["state_db"] = str(tmp_path / "state.db")
    Path(config["paths"]["incoming"]).mkdir()
    return config, StateStore(config["paths"]["state_db"])


def invoke_cli(monkeypatch, tmp_path, config, command, *args):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["yolo-pipeline", command, *args, "--config", str(path)])
    cli.main()


def test_status_leaves_active_state_database_and_directories_unchanged(setup_pipeline, tmp_path, monkeypatch, capsys):
    config, store = setup_pipeline
    run = store.create("active", str(tmp_path / "active"))
    store.transition(run.id, State.NEW_DATA, State.TRAINING)
    # Status must also work without performing the new source-key migration.
    with store.connection() as db:
        db.execute("DROP INDEX pipeline_runs_source_key")
        db.execute("ALTER TABLE pipeline_runs DROP COLUMN source_key")
    before = Path(store.path).read_bytes()
    config["logging"] = {"file": str(tmp_path / "logs" / "service.log")}
    invoke_cli(monkeypatch, tmp_path, config, "status")
    assert "TRAINING" in capsys.readouterr().out
    assert store.get(run.id).state == State.TRAINING
    assert Path(store.path).read_bytes() == before
    assert not (tmp_path / "logs").exists()
    assert not Path(config["paths"]["labeling"]).exists()


def test_status_without_database_creates_nothing(setup_pipeline, tmp_path, monkeypatch, capsys):
    config, _ = setup_pipeline
    config["paths"]["state_db"] = str(tmp_path / "missing" / "state.db")
    invoke_cli(monkeypatch, tmp_path, config, "status")
    assert "No pipeline runs" in capsys.readouterr().out
    assert not (tmp_path / "missing").exists()


def test_existing_database_is_migrated_and_source_identity_is_unique(setup_pipeline):
    _, store = setup_pipeline
    old = store.create("old", "old-path")
    with store.connection() as db:
        db.execute("DROP INDEX pipeline_runs_source_key")
        db.execute("ALTER TABLE pipeline_runs DROP COLUMN source_key")
    migrated = StateStore(store.path)
    assert migrated.get(old.id).batch_id == "old"
    assert migrated.create("new", "new-path", source_key="snapshot") is not None
    assert migrated.create("different-id", "another-path", source_key="snapshot") is None
    assert migrated.has_source("snapshot")


@pytest.mark.parametrize("command", ["retry", "reprocess", "retrain", "reevaluate"])
def test_manual_commands_do_not_recover_other_active_work(setup_pipeline, tmp_path, monkeypatch, command):
    config, store = setup_pipeline
    active = store.create("active", str(tmp_path / "active"))
    store.transition(active.id, State.NEW_DATA, State.TRAINING)
    target = store.create("target", str(tmp_path / "target"))
    store.transition(target.id, State.NEW_DATA, State.TRAINING,
                     task="crack_detection", dataset_path="dataset", model_path="model.pt")
    store.transition(target.id, State.TRAINING, State.FAILED)
    monkeypatch.setattr(cli, "configure_logging", lambda _: None)
    invoke_cli(monkeypatch, tmp_path, config, command, "target")
    assert store.get(active.id).state == State.TRAINING
    assert store.get(target.id).state != State.FAILED


def make_source(config):
    source = Path(config["paths"]["incoming"]) / "reusable-name"
    source.mkdir()
    (source / "image.jpg").write_bytes(b"first image")
    return source


def test_unique_ids_and_persisted_ingestion_on_restart(setup_pipeline):
    config, store = setup_pipeline
    source = make_source(config)
    assert DatasetWatcher(config, store).scan_once() == 1
    first = store.list_all()[0]
    timestamp, identifier = first.batch_id.split("_")
    assert timestamp.endswith("Z")
    assert UUID(identifier).version == 4
    assert DatasetWatcher(config, StateStore(store.path)).scan_once() == 0
    shutil.rmtree(source)
    source.mkdir()
    (source / "image.jpg").write_bytes(b"second image")
    assert DatasetWatcher(config, store).scan_once() == 1
    second = store.list_all()[1]
    assert first.batch_id != second.batch_id
    assert (Path(first.batch_path) / "image.jpg").read_bytes() == b"first image"
    assert (Path(second.batch_path) / "image.jpg").read_bytes() == b"second image"
    assert DatasetWatcher(config, store).scan_once() == 0


def test_changed_contents_in_same_directory_create_new_batch(setup_pipeline):
    config, store = setup_pipeline
    source = make_source(config)
    watcher = DatasetWatcher(config, store)
    assert watcher.scan_once() == 1
    (source / "another.jpg").write_bytes(b"new image")
    assert watcher.scan_once() == 1
    assert watcher.scan_once() == 0
    assert len(store.list_all()) == 2


def test_legacy_batch_is_adopted_without_reimport_and_name_can_be_reused(setup_pipeline):
    config, store = setup_pipeline
    source = make_source(config)
    legacy = Path(config["paths"]["labeling"]) / source.name
    legacy.mkdir(parents=True)
    shutil.copy2(source / "image.jpg", legacy / "image.jpg")
    old = store.create(source.name, str(legacy))
    assert DatasetWatcher(config, store).scan_once() == 0
    assert DatasetWatcher(config, StateStore(store.path)).scan_once() == 0
    (source / "image.jpg").write_bytes(b"new content")
    assert DatasetWatcher(config, store).scan_once() == 1
    assert store.get(old.id).batch_id == source.name
    assert len(store.list_all()) == 2


def test_waits_until_last_observed_change_has_settled(setup_pipeline, monkeypatch):
    config, store = setup_pipeline
    source = make_source(config)
    config["watcher"]["settle_seconds"] = 10
    clock = [0]
    monkeypatch.setattr("yolo_pipeline.watcher.time.monotonic", lambda: clock[0])
    watcher = DatasetWatcher(config, store)
    assert watcher.scan_once() == 0
    clock[0] = 5
    (source / "image.jpg").write_bytes(b"still transferring")
    assert watcher.scan_once() == 0
    clock[0] = 10
    assert watcher.scan_once() == 0
    clock[0] = 15
    assert watcher.scan_once() == 1


def test_source_changing_during_copy_is_not_published(setup_pipeline, monkeypatch):
    config, store = setup_pipeline
    source = make_source(config)
    original = shutil.copy2

    def copy_and_change(src, dst):
        original(src, dst)
        (source / "image.jpg").write_bytes(b"changed during transfer")

    monkeypatch.setattr("yolo_pipeline.watcher.shutil.copy2", copy_and_change)
    watcher = DatasetWatcher(config, store)
    assert watcher.scan_once() == 0
    assert not store.list_all()
    assert not list(Path(config["paths"]["labeling"]).iterdir())
    monkeypatch.setattr("yolo_pipeline.watcher.shutil.copy2", original)
    assert watcher.scan_once() == 1


def test_failed_copy_can_be_retried_without_orphan_batch(setup_pipeline, monkeypatch):
    config, store = setup_pipeline
    make_source(config)
    original = shutil.copy2

    def fail_copy(src, dst):
        Path(dst).write_bytes(b"partial")
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr("yolo_pipeline.watcher.shutil.copy2", fail_copy)
    watcher = DatasetWatcher(config, store)
    with pytest.raises(OSError):
        watcher.scan_once()
    assert not store.list_all()
    assert not list(Path(config["paths"]["labeling"]).iterdir())
    monkeypatch.setattr("yolo_pipeline.watcher.shutil.copy2", original)
    assert watcher.scan_once() == 1


def test_duplicate_image_names_fail_before_creating_a_labeling_batch(setup_pipeline):
    config, store = setup_pipeline
    source = make_source(config)
    nested = source / "nested"
    nested.mkdir()
    (nested / "image.jpg").write_bytes(b"duplicate")
    with pytest.raises(ValueError, match="duplicate image names"):
        DatasetWatcher(config, store).scan_once()
    assert not store.list_all()
    assert not Path(config["paths"]["labeling"]).exists()


@pytest.mark.parametrize("include_negative", [True, False])
@pytest.mark.parametrize("label", [None, "1", "701"])
def test_preprocessing_negative_switch_preserves_class_labels(tmp_path, monkeypatch, include_negative, label):
    from yolo_pipeline import crack_det_pre as pre

    monkeypatch.setattr(pre, "USE_HOLE_MODEL", False)
    monkeypatch.setattr(pre, "INCLUDE_NEGATIVE_IMAGES", include_negative)
    annotation = tmp_path / "image.json"
    shapes = [] if label is None else [{"label": label, "points": [[2, 2], [10, 10]]}]
    annotation.write_text(json.dumps({"shapes": shapes}))
    output = tmp_path / "image.txt"
    image = np.ones((20, 20, 3), dtype=np.uint8)
    accepted = pre.process_annotation(image, image, annotation, output, (0, 0, 20, 20), None)
    assert accepted == (include_negative or label == "701")
    if accepted:
        if label is None:
            assert output.read_text() == ""
        else:
            assert output.read_text().split()[0] == ("1" if label == "701" else "0")
    else:
        assert not output.exists()


def test_reprocessing_after_disabling_negatives_removes_previous_example(tmp_path, monkeypatch):
    from yolo_pipeline import crack_det_pre as pre

    monkeypatch.setattr(pre, "USE_HOLE_MODEL", False)
    monkeypatch.setattr(pre, "INCLUDE_NEGATIVE_IMAGES", True)
    monkeypatch.setattr(pre, "mask_image", lambda image, *a, **kw: (image, (0, 0, 20, 20), True))
    image = np.ones((20, 20, 3), dtype=np.uint8)
    assert pre.cv2.imwrite(str(tmp_path / "C01_P1_S2.png"), image)
    (tmp_path / "C01_P1_S2.json").write_text('{"shapes": []}')
    assert pre.create_yolo_dataset(tmp_path, {})["saved"] == 1
    assert (tmp_path / "labels" / "C01_P1_S2.txt").read_text() == ""
    monkeypatch.setattr(pre, "INCLUDE_NEGATIVE_IMAGES", False)
    assert pre.create_yolo_dataset(tmp_path, {})["saved"] == 0
    assert not list((tmp_path / "images").iterdir())
    assert not list((tmp_path / "labels").iterdir())


@pytest.mark.parametrize("contents", [
    "not json",
    '{"shapes": [{"label": "invalid", "points": [[2, 2], [10, 10]]}]}',
    '{"shapes": [{"label": "701", "points": [[2, 2]]}]}',
    '{"shapes": [{"label": "701"}]}',
])
def test_invalid_annotations_do_not_become_negative_examples(tmp_path, monkeypatch, contents):
    from yolo_pipeline import crack_det_pre as pre

    monkeypatch.setattr(pre, "USE_HOLE_MODEL", False)
    monkeypatch.setattr(pre, "INCLUDE_NEGATIVE_IMAGES", True)
    annotation = tmp_path / "image.json"
    annotation.write_text(contents)
    output = tmp_path / "image.txt"
    image = np.ones((20, 20, 3), dtype=np.uint8)
    assert not pre.process_annotation(image, image, annotation, output, (0, 0, 20, 20), None)
    assert not output.exists()
