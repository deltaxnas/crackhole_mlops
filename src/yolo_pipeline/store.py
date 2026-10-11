from __future__ import annotations

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .models import PipelineRun, State

log = logging.getLogger(__name__)


class StateStore:
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = str(path)
        self.read_only = read_only
        if not read_only:
            self.init()

    @contextmanager
    def connection(self):
        if self.read_only:
            connection = sqlite3.connect(
                Path(self.path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30
            )
        else:
            connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    def init(self) -> None:
        with self.connection() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL UNIQUE,
                    task TEXT,
                    state TEXT NOT NULL,
                    batch_path TEXT NOT NULL,
                    base_dataset_path TEXT,
                    dataset_path TEXT,
                    model_path TEXT,
                    evaluation_path TEXT,
                    deployment_path TEXT,
                    training_attempt INTEGER NOT NULL DEFAULT 1,
                    evaluation_attempt INTEGER NOT NULL DEFAULT 1,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            # Migrate databases created by versions that predate multi-task runs.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(pipeline_runs)")}
            if "model_backend" not in columns:
                db.execute("ALTER TABLE pipeline_runs ADD COLUMN model_backend TEXT NOT NULL DEFAULT 'yolo'")
            if "source_key" not in columns:
                db.execute("ALTER TABLE pipeline_runs ADD COLUMN source_key TEXT")
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS pipeline_runs_source_key "
                "ON pipeline_runs(source_key)"
            )
            if "task" not in columns:
                db.execute("ALTER TABLE pipeline_runs ADD COLUMN task TEXT")
            if "evaluation_path" not in columns:
                db.execute("ALTER TABLE pipeline_runs ADD COLUMN evaluation_path TEXT")
            if "base_dataset_path" not in columns:
                db.execute("ALTER TABLE pipeline_runs ADD COLUMN base_dataset_path TEXT")
            if "training_attempt" not in columns:
                db.execute(
                    "ALTER TABLE pipeline_runs ADD COLUMN training_attempt INTEGER NOT NULL DEFAULT 1"
                )
            if "evaluation_attempt" not in columns:
                db.execute(
                    "ALTER TABLE pipeline_runs ADD COLUMN evaluation_attempt INTEGER NOT NULL DEFAULT 1"
                )
            if "deployment_path" not in columns:
                db.execute("ALTER TABLE pipeline_runs ADD COLUMN deployment_path TEXT")
            db.execute("""
                CREATE TABLE IF NOT EXISTS state_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    old_state TEXT,
                    new_state TEXT NOT NULL,
                    message TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(id)
                )
            """)

    def has_source(self, source_key: str) -> bool:
        with self.connection() as db:
            return db.execute(
                "SELECT 1 FROM pipeline_runs WHERE source_key=?", (source_key,)
            ).fetchone() is not None

    def remember_legacy_source(self, run_id: int, source_key: str) -> bool:
        """Associate a pre-upgrade run with its unchanged incoming snapshot."""
        with self.connection() as db:
            result = db.execute(
                "UPDATE pipeline_runs SET source_key=? WHERE id=? AND source_key IS NULL",
                (source_key, run_id),
            )
            return result.rowcount == 1

    def create(
        self, batch_id: str, batch_path: str, *, source_key: str | None = None,
        model_backend: str = "yolo",
    ) -> PipelineRun | None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as db:
            try:
                cursor = db.execute(
                    "INSERT INTO pipeline_runs(batch_id,state,batch_path,created_at,updated_at,source_key,model_backend) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (batch_id, State.NEW_DATA, batch_path, now, now, source_key, model_backend),
                )
            except sqlite3.IntegrityError:
                return None
            run_id = cursor.lastrowid
            db.execute(
                "INSERT INTO state_events(run_id,new_state,message,created_at) VALUES(?,?,?,?)",
                (run_id, State.NEW_DATA, "batch discovered", now),
            )
        run = self.get(run_id)
        log.info("[%s] Pipeline run created | run_id=%s | state=%s", batch_id, run_id, State.NEW_DATA)
        return run

    def get(self, run_id: int) -> PipelineRun:
        with self.connection() as db:
            row = db.execute("SELECT * FROM pipeline_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._run(row)

    def get_by_batch_id(self, batch_id: str) -> PipelineRun:
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM pipeline_runs WHERE batch_id=?", (batch_id,)
            ).fetchone()
        if row is None:
            raise KeyError(batch_id)
        return self._run(row)

    def list_all(self) -> list[PipelineRun]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM pipeline_runs ORDER BY id").fetchall()
        return [self._run(row) for row in rows]

    def list_active(self) -> list[PipelineRun]:
        terminal = (State.APPROVED, State.REJECTED, State.FAILED)
        marks = ",".join("?" for _ in terminal)
        with self.connection() as db:
            rows = db.execute(
                f"SELECT * FROM pipeline_runs WHERE state NOT IN ({marks}) ORDER BY id", terminal
            ).fetchall()
        return [self._run(row) for row in rows]

    def list_failed(self) -> list[PipelineRun]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM pipeline_runs WHERE state=? ORDER BY id", (State.FAILED,)
            ).fetchall()
        return [self._run(row) for row in rows]

    def latest_prior_dataset(self, task: str, before_run_id: int, model_backend: str = "yolo") -> str | None:
        """Return the preceding immutable dataset version for cumulative training."""
        with self.connection() as db:
            row = db.execute(
                """
                SELECT dataset_path FROM pipeline_runs
                WHERE task=? AND id<? AND dataset_path IS NOT NULL AND model_backend=?
                ORDER BY id DESC LIMIT 1
                """,
                (task, before_run_id, model_backend),
            ).fetchone()
        return row["dataset_path"] if row else None

    def previous_approved(self, task: str, before_run_id: int, model_backend: str = "yolo") -> PipelineRun | None:
        with self.connection() as db:
            row = db.execute(
                """
                SELECT * FROM pipeline_runs
                WHERE task=? AND id<? AND state=? AND model_backend=?
                ORDER BY id DESC LIMIT 1
                """,
                (task, before_run_id, State.APPROVED, model_backend),
            ).fetchone()
        return self._run(row) if row else None

    def requeue_stage(self, batch_id: str, stage: str) -> PipelineRun:
        """Rewind a terminal run to a validated stage boundary for a new attempt."""
        run = self.get_by_batch_id(batch_id)
        terminal = (State.APPROVED, State.REJECTED, State.FAILED)
        if run.state not in terminal:
            raise ValueError(
                f"batch {batch_id!r} is active in {run.state}; only terminal runs can be rerun"
            )
        if not run.task:
            raise ValueError(f"batch {batch_id!r} has no selected task")

        targets = {
            "preprocessing": State.LABELING_COMPLETE,
            "training": State.DATASET_READY,
            "evaluation": State.TRAINING_COMPLETE,
        }
        try:
            target = targets[stage]
        except KeyError as error:
            raise ValueError(f"unknown rerun stage: {stage}") from error
        if stage == "training" and not run.dataset_path:
            raise ValueError(f"batch {batch_id!r} has no dataset to retrain")
        if stage == "evaluation" and not run.model_path:
            raise ValueError(f"batch {batch_id!r} has no model to reevaluate")

        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as db:
            if stage == "preprocessing":
                descendant = db.execute(
                    """
                    SELECT batch_id FROM pipeline_runs
                    WHERE task=? AND id>? AND model_backend=?
                    ORDER BY id LIMIT 1
                    """,
                    (run.task, run.id, run.model_backend),
                ).fetchone()
                if descendant:
                    raise ValueError(
                        f"cannot reprocess {batch_id!r}: later dataset batch "
                        f"{descendant['batch_id']!r} depends on it"
                    )

            assignments = ["state=?", "updated_at=?", "error=NULL"]
            values: list[object] = [target, now]
            if stage == "preprocessing":
                assignments.extend([
                    "base_dataset_path=NULL", "dataset_path=NULL", "model_path=NULL",
                    "evaluation_path=NULL", "deployment_path=NULL",
                    "training_attempt=training_attempt+1",
                    "evaluation_attempt=evaluation_attempt+1",
                ])
            elif stage == "training":
                assignments.extend([
                    "model_path=NULL", "evaluation_path=NULL", "deployment_path=NULL",
                    "training_attempt=training_attempt+1",
                    "evaluation_attempt=evaluation_attempt+1",
                ])
            else:
                assignments.extend([
                    "evaluation_path=NULL", "deployment_path=NULL",
                    "evaluation_attempt=evaluation_attempt+1",
                ])
            marks = ",".join("?" for _ in terminal)
            result = db.execute(
                f"UPDATE pipeline_runs SET {','.join(assignments)} "
                f"WHERE id=? AND state IN ({marks})",
                [*values, run.id, *terminal],
            )
            if result.rowcount != 1:
                raise RuntimeError(f"batch {batch_id!r} changed state while requeuing")
            db.execute(
                """
                INSERT INTO state_events(run_id,old_state,new_state,message,created_at)
                VALUES(?,?,?,?,?)
                """,
                (run.id, run.state, target, f"manual rerun from {stage}", now),
            )
        rerun = self.get(run.id)
        log.info(
            "[%s] Manual rerun queued | stage=%s | %s -> %s",
            batch_id, stage, run.state, target,
        )
        return rerun

    def recover_interrupted(self) -> int:
        """Return interrupted synchronous stages to a safely repeatable boundary."""
        rollback = {
            State.DATASET_BUILDING: State.LABELING_COMPLETE,
            State.TRAINING: State.DATASET_READY,
            State.EVALUATING: State.TRAINING_COMPLETE,
        }
        recovered = 0
        for run in self.list_active():
            target = rollback.get(run.state)
            if target:
                log.warning("[%s] Recovering interrupted stage | %s -> %s", run.batch_id, run.state, target)
                if self.transition(run.id, run.state, target, "recovered after service restart"):
                    recovered += 1
        return recovered

    def retry_failed(self, batch_id: str) -> PipelineRun:
        """Restore a failed run to the safe boundary before its failed stage."""
        run = self.get_by_batch_id(batch_id)
        if run.state != State.FAILED:
            raise ValueError(f"batch {batch_id!r} is {run.state}, not FAILED")

        with self.connection() as db:
            event = db.execute(
                """
                SELECT old_state FROM state_events
                WHERE run_id=? AND new_state=?
                ORDER BY id DESC LIMIT 1
                """,
                (run.id, State.FAILED),
            ).fetchone()
        if event is None or event["old_state"] is None:
            raise ValueError(f"batch {batch_id!r} has no recorded failed stage")

        failed_stage = State(event["old_state"])
        retry_state = {
            State.DATASET_BUILDING: State.LABELING_COMPLETE,
            State.TRAINING: State.DATASET_READY,
            State.EVALUATING: State.TRAINING_COMPLETE,
        }.get(failed_stage, failed_stage)
        if retry_state in (State.APPROVED, State.REJECTED, State.FAILED):
            raise ValueError(f"cannot retry batch {batch_id!r} from {failed_stage}")
        if not self.transition(
            run.id,
            State.FAILED,
            retry_state,
            f"manual retry after failure in {failed_stage}",
            error=None,
        ):
            raise RuntimeError(f"batch {batch_id!r} changed state while retrying")
        return self.get(run.id)

    def transition(self, run_id: int, expected: State, new: State, message: str = "", **fields) -> bool:
        allowed = {
            "task", "base_dataset_path", "dataset_path", "model_path",
            "evaluation_path", "deployment_path", "error", "model_backend",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown fields: {unknown}")
        now = datetime.now(timezone.utc).isoformat()
        assignments = ["state=?", "updated_at=?"] + [f"{key}=?" for key in fields]
        values = [new, now, *fields.values(), run_id, expected]
        with self.connection() as db:
            result = db.execute(
                f"UPDATE pipeline_runs SET {','.join(assignments)} WHERE id=? AND state=?", values
            )
            if result.rowcount != 1:
                return False
            db.execute(
                "INSERT INTO state_events(run_id,old_state,new_state,message,created_at) VALUES(?,?,?,?,?)",
                (run_id, expected, new, message, now),
            )
            batch_id = db.execute(
                "SELECT batch_id FROM pipeline_runs WHERE id=?", (run_id,)
            ).fetchone()["batch_id"]
        suffix = f" | {message}" if message else ""
        log.info("[%s] %s -> %s%s", batch_id, expected, new, suffix)
        return True

    @staticmethod
    def _run(row: sqlite3.Row) -> PipelineRun:
        return PipelineRun(
            id=row["id"], batch_id=row["batch_id"], task=row["task"], state=State(row["state"]),
            batch_path=row["batch_path"], base_dataset_path=row["base_dataset_path"],
            dataset_path=row["dataset_path"],
            model_path=row["model_path"], evaluation_path=row["evaluation_path"],
            deployment_path=row["deployment_path"], error=row["error"],
            training_attempt=row["training_attempt"],
            evaluation_attempt=row["evaluation_attempt"],
            model_backend=row["model_backend"] if "model_backend" in row.keys() else "yolo",
        )
