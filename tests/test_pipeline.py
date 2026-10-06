from pathlib import Path
import sys
import types

from yolo_pipeline.dataset import build_dataset, validate_batch
from yolo_pipeline.config import for_task, load_config
from yolo_pipeline.deployment import stage_approved_model
from yolo_pipeline.evaluation_adapter import EvaluationResult, compare_with_previous
from yolo_pipeline.models import State
from yolo_pipeline.orchestrator import Orchestrator
from yolo_pipeline.store import StateStore
from yolo_pipeline.watcher import DatasetWatcher


def test_state_compare_and_swap(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    run = store.create("batch1", str(tmp_path / "batch1"))
    assert store.transition(run.id, State.NEW_DATA, State.LABELING)
    assert not store.transition(run.id, State.NEW_DATA, State.LABELING)
    assert store.get(run.id).state == State.LABELING


def test_task_is_persisted(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    run = store.create("batch1", str(tmp_path / "batch1"))
    assert store.transition(
        run.id, State.NEW_DATA, State.LABELING_COMPLETE, task="crack_detection"
    )
    assert store.get(run.id).task == "crack_detection"


def test_interrupted_stage_is_recovered(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    run = store.create("batch1", str(tmp_path / "batch1"))
    store.transition(run.id, State.NEW_DATA, State.DATASET_BUILDING)
    assert store.recover_interrupted() == 1
    assert store.get(run.id).state == State.LABELING_COMPLETE


def test_failed_training_can_be_retried_from_dataset_ready(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    run = store.create("batch1", str(tmp_path / "batch1"))
    store.transition(run.id, State.NEW_DATA, State.DATASET_READY)
    store.transition(run.id, State.DATASET_READY, State.TRAINING)
    store.transition(run.id, State.TRAINING, State.FAILED, error="out of memory")

    retried = store.retry_failed("batch1")

    assert retried.state == State.DATASET_READY
    assert retried.error is None


def test_failed_preprocessing_can_be_retried(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    run = store.create("batch1", str(tmp_path / "batch1"))
    store.transition(run.id, State.NEW_DATA, State.LABELING_COMPLETE)
    store.transition(run.id, State.LABELING_COMPLETE, State.DATASET_BUILDING)
    store.transition(run.id, State.DATASET_BUILDING, State.FAILED, error="bad annotation")

    assert store.retry_failed("batch1").state == State.LABELING_COMPLETE


def test_latest_prior_dataset_is_task_specific(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    first = store.create("first", str(tmp_path / "first"))
    store.transition(
        first.id, State.NEW_DATA, State.DATASET_READY,
        task="crack_detection", dataset_path="/datasets/crack/first",
    )
    other = store.create("other", str(tmp_path / "other"))
    store.transition(
        other.id, State.NEW_DATA, State.DATASET_READY,
        task="hole_detection", dataset_path="/datasets/hole/other",
    )
    current = store.create("current", str(tmp_path / "current"))
    store.transition(
        current.id, State.NEW_DATA, State.LABELING_COMPLETE, task="crack_detection"
    )

    assert store.latest_prior_dataset("crack_detection", current.id) == "/datasets/crack/first"


def test_previous_approved_is_task_specific(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    approved = store.create("approved", str(tmp_path / "approved"))
    store.transition(
        approved.id, State.NEW_DATA, State.APPROVED,
        task="crack_detection", evaluation_path="/reports/approved.json",
    )
    rejected = store.create("rejected", str(tmp_path / "rejected"))
    store.transition(
        rejected.id, State.NEW_DATA, State.REJECTED,
        task="crack_detection", evaluation_path="/reports/rejected.json",
    )
    current = store.create("current", str(tmp_path / "current"))
    store.transition(
        current.id, State.NEW_DATA, State.TRAINING_COMPLETE, task="crack_detection"
    )

    baseline = store.previous_approved("crack_detection", current.id)
    assert baseline.batch_id == "approved"


def test_reevaluate_preserves_model_and_increments_only_evaluation_attempt(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    run = store.create("batch1", str(tmp_path / "batch1"))
    store.transition(
        run.id,
        State.NEW_DATA,
        State.REJECTED,
        task="crack_detection",
        dataset_path="/datasets/batch1",
        model_path="/runs/batch1/best.pt",
        evaluation_path="/reports/eval_001/evaluation.json",
    )

    rerun = store.requeue_stage("batch1", "evaluation")

    assert rerun.state == State.TRAINING_COMPLETE
    assert rerun.model_path == "/runs/batch1/best.pt"
    assert rerun.evaluation_path is None
    assert rerun.training_attempt == 1
    assert rerun.evaluation_attempt == 2


def test_retrain_preserves_dataset_and_starts_new_attempt(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    run = store.create("batch1", str(tmp_path / "batch1"))
    store.transition(
        run.id,
        State.NEW_DATA,
        State.APPROVED,
        task="crack_detection",
        dataset_path="/datasets/batch1",
        model_path="/runs/batch1/best.pt",
        evaluation_path="/reports/evaluation.json",
    )

    rerun = store.requeue_stage("batch1", "training")

    assert rerun.state == State.DATASET_READY
    assert rerun.dataset_path == "/datasets/batch1"
    assert rerun.model_path is None
    assert rerun.training_attempt == 2
    assert rerun.evaluation_attempt == 2


def test_reprocessing_older_cumulative_dataset_is_refused(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    first = store.create("first", str(tmp_path / "first"))
    store.transition(
        first.id, State.NEW_DATA, State.REJECTED,
        task="crack_detection", dataset_path="/datasets/first",
    )
    second = store.create("second", str(tmp_path / "second"))
    store.transition(
        second.id, State.NEW_DATA, State.APPROVED,
        task="crack_detection", dataset_path="/datasets/second",
    )

    try:
        store.requeue_stage("first", "preprocessing")
    except ValueError as error:
        assert "depends on it" in str(error)
    else:
        raise AssertionError("reprocessing a dataset ancestor should be refused")


def test_dataset_and_yaml(tmp_path: Path):
    batch = tmp_path / "labeling" / "batch1"
    (batch / "images").mkdir(parents=True)
    (batch / "labels").mkdir()
    for index in range(5):
        (batch / "images" / f"{index}.jpg").write_bytes(b"fake")
        (batch / "labels" / f"{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    config = {"paths": {"datasets": str(tmp_path / "datasets")}, "dataset": {
        "class_names": ["defect"], "image_extensions": [".jpg"], "seed": 1, "val_fraction": .2}}
    result = build_dataset(config, "batch1", str(batch))
    assert (result / "data.yaml").exists()
    assert len(list((result / "images" / "train").iterdir())) == 4
    assert len(list((result / "images" / "val").iterdir())) == 1


def test_dataset_extends_base_yaml_without_modifying_base(tmp_path: Path):
    seed = tmp_path / "seed"
    for split in ("train", "val"):
        (seed / "images" / split).mkdir(parents=True)
        (seed / "labels" / split).mkdir(parents=True)
        (seed / "images" / split / f"base_{split}.jpg").write_bytes(b"base")
        (seed / "labels" / split / f"base_{split}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n"
        )
    seed_yaml = seed / "data.yaml"
    seed_yaml.write_text(
        f"path: {seed}\ntrain: images/train\nval: images/val\nnames: [defect]\n"
    )

    incoming = tmp_path / "converted" / "batch1"
    (incoming / "images").mkdir(parents=True)
    (incoming / "labels").mkdir()
    (incoming / "images" / "new.jpg").write_bytes(b"new")
    (incoming / "labels" / "new.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    config = {
        "paths": {"datasets": str(tmp_path / "versions")},
        "dataset": {
            "class_names": ["defect"], "image_extensions": [".jpg"],
            "seed": 1, "val_fraction": .2,
        },
        "task_type": "detect",
    }

    result = build_dataset(
        config, "batch1", str(incoming), base_dataset_yaml=seed_yaml
    )

    assert len(list((result / "images").rglob("*.jpg"))) == 3
    assert (seed / "images" / "train" / "base_train.jpg").read_bytes() == b"base"


def test_next_dataset_version_extends_previous_version(tmp_path: Path):
    config = {
        "paths": {"datasets": str(tmp_path / "versions")},
        "dataset": {
            "class_names": ["defect"], "image_extensions": [".jpg"],
            "seed": 1, "val_fraction": .2,
        },
        "task_type": "detect",
    }
    first = tmp_path / "converted" / "first"
    second = tmp_path / "converted" / "second"
    for batch, name in ((first, "one"), (second, "two")):
        (batch / "images").mkdir(parents=True)
        (batch / "labels").mkdir()
        (batch / "images" / f"{name}.jpg").write_bytes(b"image")
        (batch / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    version_one = build_dataset(config, "batch1", str(first))
    version_two = build_dataset(
        config, "batch2", str(second), base_dataset_yaml=version_one / "data.yaml"
    )

    assert len(list((version_one / "images").rglob("*.jpg"))) == 1
    assert len(list((version_two / "images").rglob("*.jpg"))) == 2


def test_segmentation_labels_accept_polygon_points(tmp_path: Path):
    batch = tmp_path / "converted"
    (batch / "images").mkdir(parents=True)
    (batch / "labels").mkdir()
    (batch / "images" / "panel.jpg").write_bytes(b"fake")
    (batch / "labels" / "panel.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
    )
    pairs = validate_batch(batch, 1, {".jpg"}, "segment")
    assert len(pairs) == 1


def test_relative_log_path_is_resolved(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
paths:
  incoming: incoming
  labeling: labeling
  datasets: datasets
  runs: runs
  state_db: pipeline.db
logging:
  file: logs/pipeline.log
""")
    config = load_config(config_file)
    assert config["logging"]["file"] == str((tmp_path / "logs/pipeline.log").resolve())


def test_watcher_stages_a_flat_labelme_workspace(tmp_path: Path):
    incoming = tmp_path / "incoming"
    source = incoming / "batch1"
    source.mkdir(parents=True)
    (source / "one.jpg").write_bytes(b"fake")
    config = {
        "paths": {"incoming": str(incoming), "labeling": str(tmp_path / "labeling")},
        "dataset": {"image_extensions": [".jpg"]},
        "watcher": {"settle_seconds": 0},
        "tasks": {
            "crack_detection": {"done_marker": "DONE_crack_det"},
            "hole_detection": {"done_marker": "DONE_hole_det"},
        },
    }
    store = StateStore(tmp_path / "state.db")

    assert DatasetWatcher(config, store).scan_once() == 1
    run = store.list_all()[0]
    batch = Path(run.batch_path)
    assert run.batch_id != "batch1"
    assert batch.name == run.batch_id
    assert (batch / "one.jpg").exists()
    assert not (batch / "images").exists()
    assert not (batch / "labels").exists()
    info = (batch / "BATCH_INFO.txt").read_text()
    assert "source_name=batch1" in info
    assert "Labelme JSON" in info
    assert "DONE_crack_det" in info


def test_task_specific_done_marker_sets_task(tmp_path: Path):
    batch = tmp_path / "labeling" / "batch1"
    batch.mkdir(parents=True)
    (batch / "DONE_crack_det").touch()
    store = StateStore(tmp_path / "state.db")
    run = store.create("batch1", str(batch))
    store.transition(run.id, State.NEW_DATA, State.LABELING)

    config = {"tasks": {
        "crack_detection": {"done_marker": "DONE_crack_det"},
        "hole_detection": {"done_marker": "DONE_hole_det"},
    }}
    Orchestrator(config, store).run_once()
    assert store.get(run.id).state == State.LABELING_COMPLETE
    assert store.get(run.id).task == "crack_detection"


def test_failure_writes_instructions_and_retry_marker_resets_run(tmp_path: Path):
    batch = tmp_path / "labeling" / "batch1"
    batch.mkdir(parents=True)
    store = StateStore(tmp_path / "state.db")
    run = store.create("batch1", str(batch))
    store.transition(run.id, State.NEW_DATA, State.TRAINING)
    store.transition(run.id, State.TRAINING, State.FAILED, error="out of memory")
    failed = store.get(run.id)
    pipeline = Orchestrator({}, store)
    pipeline._write_error_instructions(failed, RuntimeError("out of memory"))

    instructions = (batch / "ERROR.txt").read_text()
    assert "Fix the problem" in instructions
    assert "yolo-pipeline retry batch1" in instructions
    (batch / "RETRY").touch()
    pipeline._accept_retry_markers()

    assert store.get(run.id).state == State.DATASET_READY
    assert not (batch / "RETRY").exists()
    assert not (batch / "ERROR.txt").exists()


def test_task_config_overrides_shared_defaults():
    config = {
        "paths": {"datasets": "all-data", "runs": "all-runs"},
        "dataset": {"seed": 42},
        "training": {"epochs": 100},
        "tasks": {"panel_segmentation": {
            "type": "segment",
            "paths": {"datasets": "panel-data", "runs": "panel-runs"},
            "dataset": {"class_names": ["panel"]},
            "training": {"model": "yolo11n-seg.pt"},
        }},
    }
    task = for_task(config, "panel_segmentation")
    assert task["task_type"] == "segment"
    assert task["paths"]["datasets"] == "panel-data"
    assert task["dataset"] == {"seed": 42, "class_names": ["panel"]}
    assert task["training"] == {"epochs": 100, "model": "yolo11n-seg.pt"}


def test_evaluation_report_path_is_persisted(tmp_path: Path):
    store = StateStore(tmp_path / "state.db")
    run = store.create("batch1", str(tmp_path / "batch1"))
    store.transition(run.id, State.NEW_DATA, State.EVALUATING)
    store.transition(
        run.id,
        State.EVALUATING,
        State.APPROVED,
        evaluation_path=str(tmp_path / "evaluation.json"),
        deployment_path=str(tmp_path / "deployment" / "approved.pt"),
    )
    assert store.get(run.id).evaluation_path == str(tmp_path / "evaluation.json")
    assert store.get(run.id).deployment_path == str(
        tmp_path / "deployment" / "approved.pt"
    )


def test_previous_result_regression_can_reject_candidate(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"precision": 0.90, "recall": 0.85}')
    current = tmp_path / "current.json"
    current.write_text('{"precision": 0.88, "recall": 0.86, "passed": true}')
    result = EvaluationResult(
        passed=True,
        summary="absolute thresholds passed",
        report_path=str(current),
        metrics={"precision": 0.88, "recall": 0.86},
    )
    config = {"evaluation": {"comparison": {
        "enabled": True,
        "metrics": ["precision", "recall"],
        "max_regression": {"precision": 0.01, "recall": 0.01},
    }}}

    compared = compare_with_previous(result, config, "baseline_batch", str(baseline))

    assert not compared.passed
    assert "precision_delta=-0.0200" in compared.summary


def test_approved_model_is_staged_with_informative_name(tmp_path: Path):
    model = tmp_path / "runs" / "best.pt"
    model.parent.mkdir()
    model.write_bytes(b"weights")
    config = {"deployment": {
        "destination": str(tmp_path / "deployment"),
        "filename_metrics": ["precision", "recall"],
    }}

    staged = stage_approved_model(
        config,
        "crack_detection",
        "batch 01",
        2,
        3,
        str(model),
        {"precision": 0.91234, "recall": 0.85678},
    )

    assert staged.read_bytes() == b"weights"
    assert "crack_detection__batch-batch-01" in staged.name
    assert "train-002__eval-003" in staged.name
    assert "precision-0.9123__recall-0.8568" in staged.name
    assert "sha256-" in staged.name


def test_crack_evaluation_adapter_returns_pipeline_result(monkeypatch):
    fake_module = types.ModuleType("yolo_pipeline.crack_det_eval")
    fake_module.evaluate_crack_model = lambda model, config, batch, dataset_yaml, attempt: {
        "passed": True,
        "precision": 0.9,
        "recall": 0.8,
        "tp": 9,
        "fp": 1,
        "fn": 2,
        "images_evaluated": 12,
        "report_path": "/tmp/evaluation.json",
    }
    monkeypatch.setitem(sys.modules, "yolo_pipeline.crack_det_eval", fake_module)
    from yolo_pipeline.evaluation_adapter import evaluate

    result = evaluate(
        "best.pt", "data.yaml", {"task_name": "crack_detection"}, "batch1"
    )
    assert result.passed
    assert result.report_path == "/tmp/evaluation.json"
    assert "precision=0.9000" in result.summary
