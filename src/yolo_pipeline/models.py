from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class State(StrEnum):
    NEW_DATA = "NEW_DATA"
    LABELING = "LABELING"
    LABELING_COMPLETE = "LABELING_COMPLETE"
    DATASET_BUILDING = "DATASET_BUILDING"
    DATASET_READY = "DATASET_READY"
    TRAINING = "TRAINING"
    TRAINING_COMPLETE = "TRAINING_COMPLETE"
    EVALUATING = "EVALUATING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class PipelineRun:
    id: int
    batch_id: str
    task: str | None
    state: State
    batch_path: str
    base_dataset_path: str | None
    dataset_path: str | None
    model_path: str | None
    evaluation_path: str | None
    deployment_path: str | None
    training_attempt: int
    evaluation_attempt: int
    error: str | None
