from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class EvaluationResult:
    passed: bool
    summary: str = ""
    report_path: str | None = None
    metrics: dict[str, float | int] = field(default_factory=dict)


def evaluate(
    model_path: str,
    dataset_yaml: str,
    config: dict,
    batch_id: str,
    evaluation_attempt: int = 1,
) -> EvaluationResult:
    """Dispatch evaluation for the task selected by the DONE marker."""
    task_name = config["task_name"]
    if config.get("model_backend") == "florence":
        from .florence_backend import evaluate_florence
        return evaluate_florence(model_path, dataset_yaml, config, batch_id, evaluation_attempt)
    if task_name == "crack_detection":
        from .crack_det_eval import evaluate_crack_model

        metrics = evaluate_crack_model(
            model_path, config, batch_id, dataset_yaml, evaluation_attempt
        )
        summary = (
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
            f"tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']}, "
            f"images={metrics['images_evaluated']}"
        )
        return EvaluationResult(
            passed=metrics["passed"],
            summary=summary,
            report_path=metrics["report_path"],
            metrics={
                key: metrics[key]
                for key in ("precision", "recall", "tp", "fp", "fn", "images_evaluated")
            },
        )
    if task_name in {"panel_segmentation", "hole_detection"}:
        from .yolo_evaluation import evaluate_yolo_model

        return evaluate_yolo_model(model_path, dataset_yaml, config, batch_id, evaluation_attempt)
    raise NotImplementedError(f"evaluation is not implemented for task: {task_name}")


def compare_with_previous(
    result: EvaluationResult,
    config: dict,
    previous_batch_id: str | None,
    previous_report_path: str | None,
) -> EvaluationResult:
    """Apply configured regression limits against the previous approved report."""
    options = config.get("evaluation", {}).get("comparison", {})
    if not options.get("enabled", False):
        return result

    comparison: dict = {
        "enabled": True,
        "baseline_batch_id": previous_batch_id,
        "baseline_report_path": previous_report_path,
    }
    if previous_batch_id is None:
        comparison.update({"passed": True, "reason": "no previous approved model"})
        _update_report(result.report_path, comparison, result.passed)
        return replace(
            result,
            summary=f"{result.summary}; baseline=none (first approved candidate)",
        )
    if not previous_report_path:
        raise ValueError(
            f"previous approved batch {previous_batch_id!r} has no evaluation report"
        )
    baseline_path = Path(previous_report_path)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"previous evaluation report not found: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    metric_names = options.get("metrics", ["precision", "recall"])
    configured_regression = options.get("max_regression", 0.0)
    checks = {}
    for metric in metric_names:
        if metric not in result.metrics or metric not in baseline:
            raise ValueError(f"comparison metric {metric!r} is missing from an evaluation report")
        allowed_drop = (
            float(configured_regression.get(metric, 0.0))
            if isinstance(configured_regression, dict)
            else float(configured_regression)
        )
        current_value = float(result.metrics[metric])
        baseline_value = float(baseline[metric])
        checks[metric] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta": current_value - baseline_value,
            "max_regression": allowed_drop,
            "passed": current_value >= baseline_value - allowed_drop,
        }
    comparison_passed = all(check["passed"] for check in checks.values())
    comparison.update({"metrics": checks, "passed": comparison_passed})
    final_passed = result.passed and comparison_passed
    deltas = ", ".join(
        f"{metric}_delta={check['delta']:+.4f}" for metric, check in checks.items()
    )
    _update_report(result.report_path, comparison, final_passed)
    return replace(
        result,
        passed=final_passed,
        summary=f"{result.summary}; baseline={previous_batch_id}; {deltas}",
    )


def _update_report(report_path: str | None, comparison: dict, passed: bool) -> None:
    if not report_path:
        return
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(f"current evaluation report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["comparison"] = comparison
    report["passed"] = passed
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
