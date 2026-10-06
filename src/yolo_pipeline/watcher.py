from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from .store import StateStore

log = logging.getLogger(__name__)


def _snapshot(source: Path) -> tuple:
    """Track individual files, including replacements with the same size."""
    directory = source.stat()
    entries = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            stat = path.stat()
            entries.append((
                path.relative_to(source).as_posix(), stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns,
            ))
    return directory.st_dev, directory.st_ino, tuple(entries)


class DatasetWatcher:
    """Stage each settled incoming snapshot once, with an independent batch ID."""

    def __init__(self, config: dict, store: StateStore):
        self.config, self.store = config, store
        self.seen: dict[Path, tuple[tuple, float]] = {}

    def _adopt_legacy_batch(self, source: Path, images: list[Path], source_key: str) -> bool:
        """Avoid reimporting unchanged folders already ingested before UUID IDs."""
        try:
            run = self.store.get_by_batch_id(source.name)
        except KeyError:
            return False
        batch = Path(run.batch_path)
        if not batch.is_dir():
            return False
        extensions = {x.lower() for x in self.config["dataset"]["image_extensions"]}
        staged = [p for p in batch.iterdir() if p.is_file() and p.suffix.lower() in extensions]

        def metadata(paths):
            return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths)

        if metadata(images) != metadata(staged):
            return False
        return self.store.remember_legacy_source(run.id, source_key)

    def scan_once(self) -> int:
        incoming = Path(self.config["paths"]["incoming"])
        labeling = Path(self.config["paths"]["labeling"])
        extensions = {x.lower() for x in self.config["dataset"]["image_extensions"]}
        created = 0
        sources = sorted(path for path in incoming.iterdir() if path.is_dir())
        self.seen = {path: value for path, value in self.seen.items() if path in sources}
        for source in sources:
            signature = _snapshot(source)
            previous = self.seen.get(source)
            if previous is None or previous[0] != signature:
                self.seen[source] = (signature, time.monotonic())
            if time.monotonic() - self.seen[source][1] < self.config["watcher"]["settle_seconds"]:
                continue

            images = [source / entry[0] for entry in signature[2]
                      if Path(entry[0]).suffix.lower() in extensions]
            if not images:
                continue
            source_key = hashlib.sha256(json.dumps(
                [str(source.resolve()), signature, sorted(extensions)]
            ).encode("utf-8")).hexdigest()
            if self.store.has_source(source_key):
                continue
            if len({image.name for image in images}) != len(images):
                raise ValueError(f"incoming folder {source.name!r} contains duplicate image names")
            if self._adopt_legacy_batch(source, images, source_key):
                continue

            now = datetime.now(timezone.utc)
            batch_id = f"{now:%Y%m%dT%H%M%S%fZ}_{uuid4().hex}"
            destination = labeling / batch_id
            labeling.mkdir(parents=True, exist_ok=True)
            # Publish only a complete copy. Failed copies leave no visible batch
            # directory that could block the next ingestion attempt.
            with TemporaryDirectory(prefix=".ingest-", dir=labeling) as temporary:
                staging = Path(temporary) / "batch"
                staging.mkdir()
                for image in images:
                    shutil.copy2(image, staging / image.name)
                after_copy = _snapshot(source)
                if after_copy != signature:
                    self.seen[source] = (after_copy, time.monotonic())
                    log.info("Incoming folder changed during copying; waiting again | source=%s", source)
                    continue
                markers = ", ".join(
                    task["done_marker"] for task in self.config.get("tasks", {}).values()
                )
                (staging / "BATCH_INFO.txt").write_text(
                    f"batch_id={batch_id}\nsource_name={source.name}\nsource_path={source.resolve()}\n"
                    f"created={now.isoformat()}\n"
                    "Open this directory in Labelme. Save each Labelme JSON annotation beside "
                    f"its image, then create one task marker when labeling is complete: {markers}.\n",
                    encoding="utf-8",
                )
                staging.rename(destination)
                try:
                    run = self.store.create(batch_id, str(destination), source_key=source_key)
                except Exception:
                    shutil.rmtree(destination)
                    raise
                if run is None:
                    shutil.rmtree(destination)
                    continue
            log.info(
                "[%s] New settled batch detected | source=%s | images=%d | labeling_path=%s",
                batch_id, source.name, len(images), destination,
            )
            created += 1
        return created
