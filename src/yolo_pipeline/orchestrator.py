from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import for_task
from .dataset import build_dataset
from .deployment import stage_approved_model
from .evaluation_adapter import compare_with_previous, evaluate
from .models import PipelineRun, State
from .preprocessing import preprocess_labelme_batch
from .store import StateStore
from .training import train_yolo

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, config: dict, store: StateStore):
        self.config, self.store = config, store

    def step(self, run: PipelineRun) -> None:
        try:
            if run.state == State.NEW_DATA:
                self.store.transition(run.id, run.state, State.LABELING, "batch ready for local labeler")
            elif run.state == State.LABELING:
                task_name = self._completed_task(run)
                if task_name:
                    log.info("[%s] Task marker detected | task=%s", run.batch_id, task_name)
                    self.store.transition(
                        run.id, run.state, State.LABELING_COMPLETE,
                        "task-specific DONE marker found", task=task_name,
                    )
            elif run.state == State.LABELING_COMPLETE:
                task_config = for_task(self.config, self._require_task(run))
                if self.store.transition(run.id, run.state, State.DATASET_BUILDING, "starting preprocessing"):
                    converted_batch = preprocess_labelme_batch(
                        task_config, run.task, run.batch_path
                    )
                    previous_dataset = self.store.latest_prior_dataset(run.task, run.id)
                    base_yaml = (
                        str(Path(previous_dataset) / "data.yaml")
                        if previous_dataset
                        else task_config["dataset"].get("base_yaml")
                    )
                    if base_yaml:
                        log.info(
                            "[%s] Extending base dataset | task=%s | base=%s",
                            run.batch_id, run.task, base_yaml,
                        )
                    dataset = build_dataset(
                        task_config,
                        run.batch_id,
                        str(converted_batch),
                        base_dataset_yaml=base_yaml,
                    )
                    self.store.transition(
                        run.id, State.DATASET_BUILDING, State.DATASET_READY,
                        "preprocessing and dataset build complete",
                        base_dataset_path=str(base_yaml) if base_yaml else None,
                        dataset_path=str(dataset),
                    )
            elif run.state == State.DATASET_READY:
                task_config = for_task(self.config, self._require_task(run))
                if self.store.transition(run.id, run.state, State.TRAINING, "starting Ultralytics training"):
                    model = train_yolo(
                        task_config,
                        run.batch_id,
                        run.dataset_path,
                        run.training_attempt,
                    )
                    self.store.transition(
                        run.id, State.TRAINING, State.TRAINING_COMPLETE,
                        "trained checkpoint ready", model_path=str(model),
                    )
            elif run.state == State.TRAINING_COMPLETE:
                task_config = for_task(self.config, self._require_task(run))
                if self.store.transition(run.id, run.state, State.EVALUATING, "starting evaluation"):
                    log.info("[%s] Evaluating model | model=%s", run.batch_id, run.model_path)
                    result = evaluate(
                        run.model_path,
                        str(Path(run.dataset_path) / "data.yaml"),
                        task_config,
                        run.batch_id,
                        run.evaluation_attempt,
                    )
                    previous = self.store.previous_approved(run.task, run.id)
                    result = compare_with_previous(
                        result,
                        task_config,
                        previous.batch_id if previous else None,
                        previous.evaluation_path if previous else None,
                    )
                    log.info(
                        "[%s] Evaluation completed | passed=%s | summary=%s",
                        run.batch_id, result.passed, result.summary or "(none)",
                    )
                    final = State.APPROVED if result.passed else State.REJECTED
                    self.store.transition(
                        run.id,
                        State.EVALUATING,
                        State.EVALUATING,
                        "evaluation report and comparison ready",
                        evaluation_path=result.report_path,
                    )
                    deployment_path = None
                    if result.passed:
                        deployment_path = stage_approved_model(
                            task_config,
                            run.task,
                            run.batch_id,
                            run.training_attempt,
                            run.evaluation_attempt,
                            run.model_path,
                            result.metrics,
                        )
                        log.info(
                            "[%s] Approved model staged for deployment | path=%s",
                            run.batch_id, deployment_path,
                        )
                    self.store.transition(
                        run.id,
                        State.EVALUATING,
                        final,
                        result.summary,
                        evaluation_path=result.report_path,
                        deployment_path=(
                            str(deployment_path) if deployment_path else None
                        ),
                    )
        except Exception as error:
            latest = self.store.get(run.id)
            fields = {"error": str(error)}
            report_path = getattr(error, "report_path", None)
            if report_path:
                fields["evaluation_path"] = report_path
            self.store.transition(run.id, latest.state, State.FAILED, str(error), **fields)
            self._write_error_instructions(latest, error)
            log.exception("run %s failed", run.id)

    @staticmethod
    def _write_error_instructions(run: PipelineRun, error: Exception) -> None:
        batch_path = Path(run.batch_path)
        if not batch_path.is_dir():
            return
        try:
            (batch_path / "ERROR.txt").write_text(
                f"Batch: {run.batch_id}\n"
                f"Task: {run.task or 'not selected'}\n"
                f"Failed stage: {run.state}\n"
                f"Error: {error}\n\n"
                "Fix the problem, then either:\n"
                f"  yolo-pipeline retry {run.batch_id} --config config.yaml\n"
                "or create an empty RETRY file in this directory.\n",
                encoding="utf-8",
            )
        except OSError:
            log.exception("[%s] Could not write ERROR.txt", run.batch_id)

    def _accept_retry_markers(self) -> None:
        for run in self.store.list_failed():
            marker = Path(run.batch_path) / "RETRY"
            if not marker.is_file():
                continue
            try:
                retried = self.store.retry_failed(run.batch_id)
                marker.unlink()
                error_file = Path(run.batch_path) / "ERROR.txt"
                if error_file.exists():
                    error_file.unlink()
                log.info(
                    "[%s] RETRY marker accepted | retry_state=%s",
                    retried.batch_id, retried.state,
                )
            except Exception:
                log.exception("[%s] Could not accept RETRY marker", run.batch_id)

    def _completed_task(self, run: PipelineRun) -> str | None:
        matches = [
            name for name, task in self.config.get("tasks", {}).items()
            if (Path(run.batch_path) / task["done_marker"]).is_file()
        ]
        if len(matches) > 1:
            raise ValueError(f"multiple task DONE markers found: {', '.join(matches)}")
        return matches[0] if matches else None

    @staticmethod
    def _require_task(run: PipelineRun) -> str:
        if not run.task:
            raise ValueError(f"run {run.id} has no task")
        return run.task

    def run_once(self) -> None:
        self._accept_retry_markers()
        for run in self.store.list_active():
            self.step(run)

    def run_forever(self) -> None:
        while True:
            self.run_once()
            time.sleep(self.config["orchestrator"]["poll_seconds"])
