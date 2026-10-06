from __future__ import annotations

from pathlib import Path
import json
import logging
from ultralytics import YOLO

log = logging.getLogger(__name__)


class StopAfterEpoch:
    """Cooperative stop checked after validation and checkpoint saving."""

    def __init__(self, marker: str):
        self.marker = Path(marker)
        self.requested = False

    def __call__(self, trainer):
        if self.marker.is_file():
            trainer.stop = True
            if not self.requested:
                log.info("Stop requested: epoch %d saved; continuing with model evaluation", trainer.epoch + 1)
            self.requested = True


def train_yolo(
    config: dict, batch_id: str, dataset_path: str, training_attempt: int = 1
) -> Path:
    options = config["training"]
    log.info(
        "[%s] Starting YOLO training | task=%s | attempt=%d | model=%s | epochs=%s | device=%s",
        batch_id, config.get("task_name"), training_attempt,
        options["model"], options["epochs"], options["device"],
    )
    model = YOLO(options["model"])
    stop = StopAfterEpoch(options["stop_file"]) if options.get("stop_file") else None
    extra = dict(options.get("extra", {}))
    if stop:
        device = options["device"]
        if (isinstance(device, (list, tuple)) and len(device) > 1) or "," in str(device):
            raise ValueError("Stop & evaluate currently supports a single training device; select one GPU or CPU")
        # Ultralytics invokes this hook after saving best.pt/last.pt. Never stop
        # by raising an exception or terminating a process: those skip finalization.
        model.add_callback("on_fit_epoch_end", stop)
        extra["save"] = True
    result = model.train(
        data=str(Path(dataset_path) / "data.yaml"),
        epochs=options["epochs"], imgsz=options["imgsz"], batch=options["batch"],
        device=options["device"], project=config["paths"]["runs"],
        name=f"{batch_id}/train_{training_attempt:03d}",
        exist_ok=True, **extra,
    )
    folder = Path(model.trainer.save_dir)
    best = folder / "weights" / "best.pt"
    if not best.is_file() and stop and stop.requested:
        best = folder / "weights" / "last.pt"
    if not best.exists():
        raise FileNotFoundError(f"training completed but {best} is missing")
    (folder / "training_summary.json").write_text(json.dumps({
        "requested_epochs": options["epochs"],
        "epochs_completed": model.trainer.epoch + 1,
        "stop_requested": bool(stop and stop.requested),
        "checkpoint": str(best),
        "training_attempt": training_attempt,
    }, indent=2), encoding="utf-8")
    log.info("[%s] YOLO training completed | best_model=%s", batch_id, best)
    return best
