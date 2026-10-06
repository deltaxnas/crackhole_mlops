# Local YOLO automation

A deliberately small pipeline for:

`new images -> Labelme labeling -> task preprocessing -> YOLO dataset/YAML -> task model -> your evaluation`

SQLite stores durable state. Files store images, labels, datasets and weights. A task-specific `DONE_*` marker is the explicit handoff from the human labeler.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml
```

Edit `config.yaml`, especially each task's class names, model, device, evaluation data, and approval thresholds.

Run the service:

```bash
yolo-pipeline run --config config.yaml
```

## Optional desktop demo

The demo is a separate desktop entry point. It does not start the incoming-folder watcher, and the existing `yolo-pipeline` commands keep their original behavior.

```bash
pip install -e ".[demo]"
yolo-demo --config config.yaml
```

Without reinstalling the editable package, you can also launch it with:

```bash
python -m yolo_pipeline.demo --config config.yaml
```

Run it in the same Python environment as training, in a desktop session with Tk support. The current Conda training environment already has Tk and matplotlib. Labelme may be in a separate environment:

```bash
python -m yolo_pipeline.demo --config config.yaml --labelme /path/to/labelme/environment/bin/labelme
```

The GUI searches PATH, the current environment, and a sibling Conda environment named `labelme`. If it cannot find Labelme, Open Labelme lets you locate its executable. Missing/crashed Labelme is reported in the GUI; its output is saved under `workspace/demo/labelme/`.

1. **Choose task, then Select batch folder**: select Crack detection, Panel segmentation, or Hole detection in the top-right task selector, then choose a completed image folder. The GUI copies it directly to a new unique labeling batch and opens Labelme. Existing matching Labelme JSON annotations are also copied; the source remains untouched. Image stems must be unique because annotations share the image stem. No watcher or settling timer is involved; finish transferring the folder first.
2. **Label**: save a Labelme JSON beside every image, including an empty `shapes` list for reviewed negative images. Save and close the Labelme window before proceeding. The crack preprocessor still requires the expected panel/camera filenames and local panel/hole-model assets; the GUI does not remove those existing requirements.
3. **Mark done & train**: validate that annotations were saved, create the selected task's `DONE_*` marker, then run preprocessing, cumulative dataset creation, training, evaluation, and approved-model staging. A subprocess keeps the window responsive. Loss curves refresh from Ultralytics `results.csv` after each epoch; evaluation appears as a separate stage, followed by its report and metrics. **Stop training & evaluate** finishes the current epoch and checkpoint save, then automatically continues to evaluation using `best.pt` (or `last.pt` if no best checkpoint was saved). A request during the first epoch waits for that epoch to finish. Approval checks still apply; stopping never auto-approves a model.
4. **Leaderboard**: completed models for the selected task appear ranked by F1, with precision, recall, and approval status. Each task has its own baseline. Segmentation uses **mask** precision/recall; holes use bounding-box metrics. Reports for both also include mAP50 and mAP50–95. Select a row to view its report or use its model for inference. The board uses the configured SQLite database, so existing CLI results also appear.
5. **Inference**: select an input folder and a model from the board, or browse for a trusted local `.pt` file. Outputs go to a unique `workspace/demo/inference/<id>/` directory containing annotated images, YOLO labels with confidence, and `inference.json`. Nested input paths are preserved. Per-image failures are reported. This is inference on original images, without the annotation-dependent panel cropping used during crack training/evaluation; it does not change leaderboard scores.

Demo approval uses a floor of **0.2 precision and 0.2 recall**. If there is no approved baseline (including an empty leaderboard), both floor values must be met. Otherwise both metrics must also meet or exceed the highest-F1 approved model's respective values, with zero allowed regression. Rejected models remain visible but do not become the baseline. Missing or malformed approved reports block comparison instead of silently falling back to 0.2. These demo rules are applied in memory and do not modify `config.yaml` or the CLI's thresholds.

All three tasks are available. Crack detection retains its custom preprocessing and fixed Labelme benchmark. Panel segmentation and hole detection use the basic converter described below. Inference accepts compatible local YOLO weights, including segmentation masks. The worker saves each job's input configuration and complete log under `workspace/demo/jobs/`. Restarting the GUI preserves batch and leaderboard records; select an unfinished/failed batch and choose **Resume / retry**. Closing the GUI during a job terminates the worker after confirmation; this is different from **Stop training & evaluate**, which keeps the worker running through evaluation. Interrupted training resumes by rerunning the stage, not by continuing a partial epoch.

Use the GUI **instead of** a running `yolo-pipeline run`/`once`/`watch` process against the same database. Demo jobs have a mutual-exclusion lock, but the existing CLI does not participate in that lock. The configured benchmark must also remain fixed when comparing model reports. No production model rollout is performed.

## Workflow

1. Create a batch directory under `workspace/incoming`, for example `incoming/batch_0001/`, and copy images into it.
2. After the directory is unchanged for `settle_seconds`, the watcher copies images into a temporary directory, checks that the source is still unchanged, and publishes `labeling/<batch_id>/`. The orchestrator then enters `LABELING`. It deliberately does not create a YOLO directory structure.
3. Open the generated `labeling/<batch_id>/` directory in Labelme. Save each Labelme `.json` annotation beside its image.
4. When labeling is finished, create exactly one task marker: `DONE_crack_det`, `DONE_hole_det`, or `DONE_panel_seg`.
5. The orchestrator derives and persists the task from that marker, calls its preprocessing hook, creates the task's YOLO dataset and `data.yaml`, trains its configured model, and evaluates it.

New batch IDs use a UTC timestamp and a UUID, independently of the incoming folder name, for example `20260911T091500123456Z_96a63a7524db4a2280d66a94b02e8a63`. The original folder name and path are recorded in `BATCH_INFO.txt`. Use the generated ID shown by `status`, the logs, or the labeling directory for retry/rerun commands. In the examples below, `batch_0001` stands for that ID. Existing runs retain their old IDs and artifact paths.

SQLite records each imported source snapshot using the source path, directory identity, file metadata, and accepted extensions. Unchanged incoming folders are not imported again after a restart. A reused folder name with replaced or changed contents receives a new batch ID. Changing an already imported folder imports its **entire current contents** as a new batch, so clear out the previous images before reusing a folder to avoid repeating them in cumulative datasets. Unchanged legacy folders are matched to their existing staged image metadata during migration.

Polling frequency is not a transfer-completion signal. Each observed file change resets the settling timer, so a transfer spanning many polls normally waits. However, a transfer paused longer than `settle_seconds` can appear finished and be staged incompletely. For a reliable handoff, finish copying into a directory **outside `incoming` on the same filesystem**, then atomically rename/move that completed directory into `incoming`. Merely increasing the settling interval cannot guarantee that a transfer has finished. The post-copy source check detects observed changes during copying; it cannot predict later writes.

## Cumulative base datasets

Each task can start from an immutable YOLO dataset YAML:

```yaml
tasks:
  crack_detection:
    dataset:
      class_names: [no_crack, crack]
      base_yaml: ./base_datasets/crack_detection/data.yaml
```

The seed YAML must have `train`, `val`, and `names`, use the conventional `images/` and `labels/` layout, and have exactly the same ordered class names as the task. Both detection and segmentation labels are validated while copying.

The first batch creates a new dataset version containing `seed + batch 1`. Each later batch automatically uses the preceding dataset version for that same task, producing `seed + batch 1 + batch 2 + ...`. Existing train/validation assignments are preserved; only the new batch is split. Dataset versions are immutable with respect to one another: rebuilding a run replaces only that run's output directory, never its seed or preceding version.

Every successfully built labeled batch participates in later versions, even if its trained model is subsequently rejected by evaluation. That matches a cumulative labeling workflow: model rejection does not imply bad labels. If labels themselves are suspect, do not create the task `DONE_*` marker until they pass review.

This design favors traceability and safe retries, but full copies consume increasing disk space. Hard links, content-addressed storage, or dataset manifests would be a sensible later optimization for large datasets.

On Linux/macOS:

```bash
touch workspace/labeling/batch_0001/DONE_crack_det
```

On PowerShell:

```powershell
New-Item workspace/labeling/batch_0001/DONE_crack_det -ItemType File
```

Inspect all runs, including failures:

```bash
yolo-pipeline status --config config.yaml
```

`status` opens an existing database read-only. It does not create directories, migrate the database, write logs, or recover active stages. Retry/rerun commands and the standalone watcher also do not recover unrelated work; recovery runs only when starting `run` or `once`. Keep a single orchestrator running per database, including when using `once`.

## Retrying a failed step

After correcting the data, configuration, preprocessing code, training problem, or evaluation code, reset a failed batch with:

```bash
yolo-pipeline retry batch_0001 --config config.yaml
```

If the main service is running, it picks the batch up on its next poll. Otherwise, run one tick yourself:

```bash
yolo-pipeline once --config config.yaml
```

Every caught batch error also creates `ERROR.txt` in its labeling directory with the failed stage, error message, and recovery instructions. As an alternative to the command, fix the problem and create an empty `RETRY` marker:

```bash
touch workspace/labeling/batch_0001/RETRY
```

The running service consumes that marker, clears `ERROR.txt`, restores the safe stage, and retries. It does not use a blocking keyboard prompt, so one failed batch cannot pause monitoring or other batches.

Batch-stage and unexpected polling errors do not terminate the continuous service. Startup failures such as malformed configuration or an unavailable state database still stop immediately, because retrying without valid global state would be unsafe.

Retry uses the recorded failure event to return to the appropriate safe boundary:

- Failed preprocessing/dataset building returns to `LABELING_COMPLETE`.
- Failed training returns to `DATASET_READY` and reruns training from the start.
- Failed evaluation returns to `TRAINING_COMPLETE` and reevaluates the existing model.
- A failure raised before entering one of those stages returns to the state where it failed.

The previous error remains in `state_events` history, while the current `pipeline_runs.error` field is cleared. Only `FAILED` batches can be manually retried; approved and rejected results are not treated as execution errors.

## Reprocessing, retraining, and reevaluation

Terminal runs (`APPROVED`, `REJECTED`, or `FAILED`) can be deliberately rewound:

```bash
yolo-pipeline reprocess batch_0001 --config config.yaml
yolo-pipeline retrain batch_0001 --config config.yaml
yolo-pipeline reevaluate batch_0001 --config config.yaml
```

| Command | Preserved | Rebuilt | Safe boundary |
| --- | --- | --- | --- |
| `reevaluate` | Dataset and trained model | Evaluation report and decision | `TRAINING_COMPLETE` |
| `retrain` | Labeled data and generated dataset | Model, evaluation, and decision | `DATASET_READY` |
| `reprocess` | Labelme source data | Dataset, model, evaluation, and decision | `LABELING_COMPLETE` |

Manual retraining and reevaluation increment persisted attempt counters. Artifacts are not overwritten:

```text
workspace/runs/crack_detection/batch_0001/train_001/weights/best.pt
workspace/runs/crack_detection/batch_0001/train_002/weights/best.pt

workspace/evaluations/crack_detection/batch_0001/eval_001/evaluation.json
workspace/evaluations/crack_detection/batch_0001/eval_002/evaluation.json
```

Only the latest artifact paths remain on the main run row; state events retain the audit trail of each rewind. A same-attempt `retry` remains different from a deliberate rerun: retry repeats a failed attempt, while these commands create a new attempt.

Reprocessing an older cumulative dataset version is refused when a later dataset version for that task already depends on it. This prevents silently changing an ancestor while descendants retain old copied data. Retraining or reevaluating an older batch is allowed because it does not alter dataset lineage or later artifacts.

The continuous service loads configuration at startup. When rerunning because `config.yaml` changed, stop/restart that service so the new configuration is used. Alternatively, queue the command while it is stopped and execute `yolo-pipeline once --config config.yaml`. Avoid editing configuration between training and evaluation unless that difference is intentional.

## Logging

Important changes are printed to the terminal and written to the configured rotating log file. Polling cycles that find nothing are intentionally silent.

```yaml
logging:
  level: INFO
  file: ./workspace/pipeline.log
  max_bytes: 10485760
  backup_count: 5
```

Normal output includes batch discovery, every state transition, label validation and split counts, training start/completion, evaluation results, recovery after restart, failures, and graceful Ctrl+C shutdown. Set `level: DEBUG` only when adding detailed diagnostic logs; the current normal workflow uses `INFO`, `WARNING`, and `ERROR`.

Run one watcher/orchestrator tick for debugging or Task Scheduler/cron:

```bash
yolo-pipeline once --config config.yaml
```

## States

`NEW_DATA -> LABELING -> LABELING_COMPLETE -> DATASET_BUILDING -> DATASET_READY -> TRAINING -> TRAINING_COMPLETE -> EVALUATING -> APPROVED/REJECTED`

Any exception produces `FAILED` with the error in SQLite. State changes use compare-and-swap updates, preventing the same stage from starting twice in one database. On service startup, interrupted build, train, or evaluation states return to their preceding safe boundary and run again.

## What `pipeline.db` stores

The SQLite file contains two tables:

- `pipeline_runs`: numeric run ID, unique batch ID, task, current state, labeling batch path, selected base-dataset YAML, generated dataset path, latest trained model path, latest evaluation report path, deployment-staging path, training/evaluation attempt numbers, last error, and created/updated timestamps.
- `state_events`: an append-only transition history containing the run ID, old/new states, transition message, and timestamp.

It does **not** contain images, Labelme JSON files, generated YOLO files, model weights, logs, configuration, or in-progress Ultralytics optimizer checkpoints. Those remain at the filesystem paths recorded by the database.

The default `.gitignore` excludes the whole `workspace/` directory, including `pipeline.db` and all data. A normal Git push and later pull therefore does not restore pipeline state. To resume on another checkout, back up/restore the database **together with** the referenced `labeling/`, `datasets/`, and `runs/` files at compatible paths (and the same configuration). Stop the service before copying SQLite, or use SQLite's online backup facility. Committing a live database is not recommended.

With a consistent database and files, startup recovery resumes at a stage boundary: interrupted preprocessing/dataset building returns to `LABELING_COMPLETE`, training to `DATASET_READY`, and evaluation to `TRAINING_COMPLETE`. It reruns that whole stage; it does not resume midway through an epoch.

## Tasks and preprocessing

The `tasks` section of the configuration is the registry for the unified framework. Each task supplies:

- A unique Labelme completion marker.
- Its YOLO task type (`detect` or `segment`).
- Its dataset root and training-run/model-output root.
- Its ordered class names, which determine class IDs in labels and `data.yaml`.
- Its starting detection or segmentation model.

Shared split and training settings remain under `dataset` and `training`; a task can override them. Each generated YAML is stored inside that task's batch dataset directory. Each trained model path and task are persisted in SQLite.

Task preprocessing hooks live in `src/yolo_pipeline/preprocessing.py`. Each returns a directory containing converted `images/` and `labels/`. Crack detection uses the custom converter; panel segmentation and hole detection use `src/yolo_pipeline/labelme_converter.py`.

The basic converter handles one class (ID `0`) and keeps images at their original size:

- **Panel segmentation:** Labelme polygons become normalized YOLO polygons. Rectangles become four-corner polygons.
- **Hole detection:** polygons and rectangles become normalized YOLO bounding boxes.
- Every image requires a matching JSON annotation. An empty `shapes` list produces an empty negative label file.
- By default, every shape label maps to class `0`. For mixed annotations, set the task's `preprocessing.labels` to a list such as `[panel]`, `["500"]`, or `[hole]`; only matching shapes are converted.
- Unsupported shape types (circles, lines, points), invalid polygons, out-of-bounds coordinates, and unreadable/missing annotations fail the batch. Nothing is silently converted into a negative example.
- Conversion is built in a temporary directory and replaces `converted_yolo/` only after the whole batch succeeds, so failed retries cannot use a partially rebuilt conversion.

These tasks require at least two labeled images to produce a nonempty train/validation split without a seed dataset. Segmentation defaults to `yolo11n-seg.pt`; hole detection defaults to `yolo11n.pt`. Ultralytics may download initial weights if they are absent locally. Unlike crack preprocessing, these basic tasks do not require the panel/camera filename convention or external panel/hole-model assets.

Crack preprocessing has a script-level option in `src/yolo_pipeline/crack_det_pre.py`:

```python
INCLUDE_NEGATIVE_IMAGES = True
```

With this enabled (the default), reviewed images without cracks are kept. Existing `no_crack` class-0 boxes, including those produced by the optional hole model, remain labeled; images with no boxes get an empty YOLO label file. Set it to `False` for the previous behavior that keeps only images with cracks. This option is intentionally not part of `config.yaml`. A valid panel crop and a matching Labelme JSON annotation are still required. Restart the service after editing the script; use `reprocess` on an eligible existing batch to apply the change to its generated data. Previously converted examples for each processed annotation are removed before conversion so disabling the option also removes that annotation's old negative example.

## Training

Training is intentionally just the Ultralytics call in `training.py`:

```python
YOLO(model_name).train(data=data_yaml, epochs=100, ...)
```

All normal Ultralytics options can be added under `training.extra` in the config.

The demo's cooperative stop uses an `on_fit_epoch_end` callback, after validation and checkpoint saving, and forces checkpoint saving on. It currently supports one GPU or CPU per job. It does not kill training or resume from an unrelated model. The run's `training_summary.json` records requested/completed epochs, whether a stop was requested, and the checkpoint used. If no usable checkpoint exists, the run fails instead of proceeding with missing weights.

## Evaluation contract

The task-aware adapter receives:

```python
evaluate(
    model_path: str,
    dataset_yaml: str,
    config: dict,
    batch_id: str,
) -> EvaluationResult
```

Crack detection is connected to the pipeline version of `crack_det_eval.py`. Unlike the original manual entry point, it evaluates exactly the newly trained `best.pt` against the configured fixed Labelme benchmark. It writes:

```text
workspace/evaluations/crack_detection/<batch_id>/eval_<attempt>/evaluation.json
```

The report contains the model and evaluation-data paths, confidence and IoU settings, TP/FP/FN, precision, recall, processed-image counts, per-image errors, approval thresholds, and the final decision. Its path is also stored in `pipeline.db`.

The crack task's `evaluation` configuration controls the held-out data directory, class/Labelme label mapping, inference confidence and device, optional panel masking/crop-letterbox behavior, annotated-image output, minimum precision/recall/image count, and error policy. The included `0.80` precision and recall approval thresholds are initial values—set them to the acceptance criteria for your production problem.

A valid evaluation below a metric threshold produces `REJECTED`. Missing data, an unreadable model, or—by default—any image-processing error raises an exception and produces `FAILED`, which can be fixed and retried.

Hole detection and panel segmentation use native Ultralytics validation through `yolo_evaluation.py`. The default input is the generated dataset's `val` split. Configure a task's `evaluation.dataset_yaml` to use a fixed external YOLO benchmark, optionally with `evaluation.split: test`; relative paths resolve against `config.yaml`. Segmentation reports mask metrics, and hole detection reports box metrics. Both record precision, recall, mAP50, mAP50–95, the validation source and image count, thresholds, and the decision in `evaluation.json`. An empty validation split or skipped/corrupt images fails evaluation. By default their minimum precision and recall are both `0.2`.

**For meaningful leaderboard comparisons, use the same fixed evaluation dataset for each model of a task.** Default batch-validation scores may use different images as cumulative datasets change; these are useful demo results but do not establish that one model is better on a common benchmark. The GUI identifies those results as evaluated on the dataset's validation split.

## Comparing and staging approved models

Approval has two optional gates:

1. The candidate must satisfy its absolute evaluation thresholds, such as minimum precision and recall.
2. When comparison is enabled, it must not regress beyond the configured tolerance relative to the latest earlier `APPROVED` model for the same task.

```yaml
evaluation:
  min_precision: 0.50
  min_recall: 0.50
  comparison:
    enabled: true
    metrics: [precision, recall]
    max_regression:
      precision: 0.01
      recall: 0.01
```

Here, either metric may drop by at most `0.01`. The current and baseline values, deltas, limits, baseline batch, and baseline report are added to the candidate's `evaluation.json`. The first candidate has no baseline and uses only absolute thresholds. If an earlier approved run is recorded but its report is missing or lacks a configured metric, comparison fails explicitly instead of being bypassed.

After both gates pass, the weight is atomically copied to the task's deployment-staging directory:

```yaml
deployment:
  destination: ./workspace/deployment/crack_detection
  filename_metrics: [precision, recall]
```

Example filename:

```text
crack_detection__batch-batch_0001__train-002__eval-003__precision-0.9123__recall-0.8568__20260907T120000Z__sha256-a1b2c3d4e5f6.pt
```

The filename carries the task, batch, training/evaluation attempts, selected metrics, UTC approval time, and the first 12 characters of the SHA-256 weight checksum. The copied path is stored in `pipeline.db`. A copy failure makes the run `FAILED`; the database never marks a model approved unless deployment staging succeeds.

This directory is a handoff boundary, not an automatic production rollout. A deployment agent can watch it or consume its files without giving the training pipeline authority to restart production services.

## Current scope

- One process should own one SQLite database.
- Each incoming subdirectory is one batch.
- Crack detection has custom preprocessing/evaluation; hole detection and panel segmentation have basic single-class Labelme conversion and native YOLO validation.
- New batch IDs are generated from UTC timestamps and UUIDs across all tasks; incoming folder names may be reused.
- Dataset versions accumulate prior successfully built batches independently per task.
- Approved models are copied to local deployment staging; production rollout, registries, CVAT, Ray, MLflow, cloud services, and checkpoint-by-checkpoint evaluation remain out of scope.
