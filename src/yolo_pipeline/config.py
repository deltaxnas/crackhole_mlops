from __future__ import annotations

from pathlib import Path
import yaml


def model_backend(value: str) -> str:
    value = str(value).lower()
    if value == "vlm":
        value = "florence"
    if value not in {"yolo", "florence"}:
        raise ValueError(f"Unknown model_backend: {value}; choose yolo or florence")
    return value


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = config_path.parent
    for section in [config, *config.get("tasks", {}).values()]:
        if "model_backend" in section:
            section["model_backend"] = model_backend(section["model_backend"])
        options = section.get("florence", {})
        for key in ("base_dataset_dir", "evaluation_jsonl"):
            if options.get(key):
                options[key] = str((base / options[key]).resolve())
        if options.get("model") and (str(options["model"]).startswith((".", "/", "~")) or (base / options["model"]).exists()):
            options["model"] = str((base / Path(options["model"]).expanduser()).resolve())
    for key in ("incoming", "labeling", "datasets", "runs", "state_db"):
        value = Path(config["paths"][key])
        config["paths"][key] = str(value if value.is_absolute() else (base / value).resolve())
    for task in config.get("tasks", {}).values():
        for key in ("datasets", "runs"):
            value_text = task.get("paths", {}).get(key)
            if value_text:
                value = Path(value_text)
                task["paths"][key] = str(value if value.is_absolute() else (base / value).resolve())
        for key in ("data_dir", "output_dir", "panel_gt_dir", "dataset_yaml"):
            value_text = task.get("evaluation", {}).get(key)
            if value_text:
                value = Path(value_text)
                task["evaluation"][key] = str(
                    value if value.is_absolute() else (base / value).resolve()
                )
        base_yaml_text = task.get("dataset", {}).get("base_yaml")
        if base_yaml_text:
            value = Path(base_yaml_text)
            task["dataset"]["base_yaml"] = str(
                value if value.is_absolute() else (base / value).resolve()
            )
        deployment_text = task.get("deployment", {}).get("destination")
        if deployment_text:
            value = Path(deployment_text)
            task["deployment"]["destination"] = str(
                value if value.is_absolute() else (base / value).resolve()
            )
    log_file = config.get("logging", {}).get("file")
    if log_file:
        value = Path(log_file)
        config["logging"]["file"] = str(value if value.is_absolute() else (base / value).resolve())
    return config


def ensure_directories(config: dict) -> None:
    for key in ("incoming", "labeling", "datasets", "runs"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["state_db"]).parent.mkdir(parents=True, exist_ok=True)
    for task in config.get("tasks", {}).values():
        for path in task.get("paths", {}).values():
            Path(path).mkdir(parents=True, exist_ok=True)


def for_task(config: dict, task_name: str, backend: str | None = None) -> dict:
    """Return shared configuration with one task's overrides applied."""
    try:
        task = config["tasks"][task_name]
    except KeyError as error:
        raise ValueError(f"unknown task: {task_name}") from error
    resolved = dict(config)
    resolved["task_name"] = task_name
    resolved["task_type"] = task["type"]
    resolved["paths"] = {**config["paths"], **task.get("paths", {})}
    resolved["dataset"] = {**config["dataset"], **task.get("dataset", {})}
    resolved["training"] = {**config["training"], **task.get("training", {})}
    resolved["evaluation"] = {
        **config.get("evaluation", {}), **task.get("evaluation", {})
    }
    resolved["preprocessing"] = {
        **config.get("preprocessing", {}), **task.get("preprocessing", {})
    }
    resolved["deployment"] = {
        **config.get("deployment", {}), **task.get("deployment", {})
    }
    resolved["model_backend"] = model_backend(backend or task.get("model_backend", config.get("model_backend", "yolo")))
    resolved["florence"] = {**config.get("florence", {}), **task.get("florence", {})}
    if resolved["model_backend"] == "florence":
        if task_name != "crack_detection":
            raise ValueError("Florence currently supports crack_detection only; use YOLO for other tasks")
        for key in ("datasets", "runs"):
            resolved["paths"][key] = str(Path(resolved["paths"][key]) / "florence")
        if resolved["deployment"].get("destination"):
            resolved["deployment"]["destination"] = str(Path(resolved["deployment"]["destination"]) / "florence")
    return resolved
