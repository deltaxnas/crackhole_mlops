"""Subprocess entry point used only by the optional desktop demo."""
from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path

from .demo_backend import TASK, import_batch, run_inference, run_training, worker_lock


def emit(**event):
    print("DEMO_EVENT " + json.dumps(event), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
        config = job["config"]
        with worker_lock(config["paths"]["state_db"]):
            if job["action"] == "import":
                run = import_batch(config, job["source"], emit, task_name=job.get("task", TASK), model_backend=job.get("model_backend"))
                emit(kind="imported", batch_id=run.batch_id)
            elif job["action"] == "train":
                run_training(config, job["batch_id"], emit, stop_file=job.get("stop_file"))
            elif job["action"] == "infer":
                report = run_inference(config, job["source"], job["model"], job["output"], emit, task_name=job.get("task", TASK), model_backend=job.get("model_backend"))
                emit(kind="inferred", output=job["output"], processed=report["processed"], errors=len(report["errors"]))
            else:
                raise ValueError(f"Unknown demo action: {job['action']}")
        emit(kind="finished")
    except Exception as error:
        traceback.print_exc()
        emit(kind="error", message=str(error))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
