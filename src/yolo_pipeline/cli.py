from __future__ import annotations

import argparse
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import ensure_directories, load_config
from .store import StateStore

log = logging.getLogger(__name__)


def configure_logging(config: dict) -> None:
    settings = config.get("logging", {})
    level_name = str(settings.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    handlers: list[logging.Handler] = [console]
    log_file = settings.get("file")
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path,
            maxBytes=int(settings.get("max_bytes", 10 * 1024 * 1024)),
            backupCount=int(settings.get("backup_count", 5)),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    logging.basicConfig(level=level, handlers=handlers, force=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    stage_commands = {"retry", "reprocess", "retrain", "reevaluate"}
    parser.add_argument(
        "command",
        choices=["run", "once", "watch", "status", *sorted(stage_commands)],
    )
    parser.add_argument("batch_id", nargs="?", help="batch used by retry/rerun commands")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    if args.command in stage_commands and not args.batch_id:
        parser.error(f"{args.command} requires a batch_id")
    if args.command not in stage_commands and args.batch_id:
        parser.error("batch_id is only valid with retry/rerun commands")
    config = load_config(args.config)
    if args.command == "status":
        database = Path(config["paths"]["state_db"])
        runs = StateStore(database, read_only=True).list_all() if database.is_file() else []
        if not runs:
            print("No pipeline runs.")
        for item in runs:
            print(item)
        return

    configure_logging(config)
    ensure_directories(config)
    store = StateStore(config["paths"]["state_db"])
    recovered = store.recover_interrupted() if args.command in {"run", "once"} else 0
    log.info(
        "Service started | command=%s | poll_seconds=%s | recovered_runs=%d",
        args.command, config["orchestrator"]["poll_seconds"], recovered,
    )
    try:
        if args.command == "retry":
            try:
                retried = store.retry_failed(args.batch_id)
            except (KeyError, ValueError, RuntimeError) as error:
                parser.exit(1, f"retry failed: {error}\n")
            for name in ("ERROR.txt", "RETRY"):
                marker = Path(retried.batch_path) / name
                try:
                    if marker.exists():
                        marker.unlink()
                except OSError:
                    log.warning("Could not remove stale marker: %s", marker)
            print(
                f"Batch {retried.batch_id} reset to {retried.state}. "
                "The running service will retry it on its next poll."
            )
        elif args.command in {"reprocess", "retrain", "reevaluate"}:
            stage = {
                "reprocess": "preprocessing",
                "retrain": "training",
                "reevaluate": "evaluation",
            }[args.command]
            try:
                rerun = store.requeue_stage(args.batch_id, stage)
            except (KeyError, ValueError, RuntimeError) as error:
                parser.exit(1, f"{args.command} failed: {error}\n")
            for name in ("ERROR.txt", "RETRY"):
                marker = Path(rerun.batch_path) / name
                try:
                    if marker.exists():
                        marker.unlink()
                except OSError:
                    log.warning("Could not remove stale marker: %s", marker)
            print(
                f"Batch {rerun.batch_id} queued from {stage} at {rerun.state}. "
                "The service will pick it up on its next poll. Restart an already-running "
                "service first if config.yaml changed."
            )
        elif args.command == "once":
            from .orchestrator import Orchestrator
            from .watcher import DatasetWatcher

            watcher = DatasetWatcher(config, store)
            orchestrator = Orchestrator(config, store)
            watcher.scan_once(); orchestrator.run_once()
        elif args.command == "watch":
            from .watcher import DatasetWatcher

            watcher = DatasetWatcher(config, store)
            while True:
                try:
                    watcher.scan_once()
                except Exception:
                    log.exception("Watcher poll failed; service will continue")
                time.sleep(config["orchestrator"]["poll_seconds"])
        else:
            from .orchestrator import Orchestrator
            from .watcher import DatasetWatcher

            watcher = DatasetWatcher(config, store)
            orchestrator = Orchestrator(config, store)
            while True:
                try:
                    watcher.scan_once()
                    orchestrator.run_once()
                except Exception:
                    log.exception("Pipeline poll failed; service will continue")
                time.sleep(config["orchestrator"]["poll_seconds"])
    except KeyboardInterrupt:
        log.info("Shutdown requested with Ctrl+C; service stopped cleanly")


if __name__ == "__main__":
    main()
