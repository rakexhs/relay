"""Aggregate scores into a JSON report and a markdown summary."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.harness import TaskResult
from evals.schema import Task
from evals.scorers import Score, percentile, score_task

HEADLINE = (
    "task_success",
    "trajectory_match",
    "error_recovery",
    "injection_resistance",
    "injection_detected",
)


def _rate(scores: list[Score]) -> dict[str, Any]:
    applicable = [s for s in scores if s.applicable]
    if not applicable:
        return {"rate": None, "passed": 0, "n": 0}
    passed = sum(s.passed for s in applicable)
    return {"rate": round(passed / len(applicable), 4), "passed": passed, "n": len(applicable)}


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def build_report(
    tasks: list[Task],
    results: list[TaskResult],
    run_info: dict[str, Any],
    judge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_task = {t.id: t for t in tasks}
    per_task = []
    collected: dict[str, list[Score]] = defaultdict(list)
    by_category: dict[str, list[Score]] = defaultdict(list)
    for result in results:
        task = by_task[result.task_id]
        scores = score_task(task, result)
        for name, score in scores.items():
            collected[name].append(score)
        by_category[task.category].append(scores["task_success"])
        per_task.append(
            {
                "id": task.id,
                "category": task.category,
                "expected": {
                    "outcome": task.expect.outcome,
                    "reason": task.expect.reason.value if task.expect.reason else None,
                },
                "outcome": {
                    "status": (result.outcome or {}).get("status"),
                    "reason": (result.outcome or {}).get("reason"),
                },
                "passed": {n: s.passed for n, s in scores.items() if s.applicable},
                "failures": sorted({f for s in scores.values() for f in s.failures}),
                "tools": [c["tool"] for c in result.calls()],
                "tool_calls": result.tool_calls,
                "approvals": result.approvals,
                "latency_ms": result.latency_ms,
                "error": result.error,
            }
        )
    latencies = [r.latency_ms for r in results]
    trajectory_values = [s.value for s in collected["trajectory_match"] if s.value is not None]
    metrics: dict[str, Any] = {name: _rate(collected[name]) for name in HEADLINE}
    metrics["trajectory_score"] = (
        round(sum(trajectory_values) / len(trajectory_values), 4) if trajectory_values else None
    )
    metrics["latency_ms"] = {
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "max": round(max(latencies), 2) if latencies else 0.0,
    }
    metrics["tool_calls_mean"] = (
        round(sum(r.tool_calls for r in results) / len(results), 2) if results else 0.0
    )
    if judge is not None:
        metrics["judge"] = judge["summary"]
    report = {
        "run": {
            **run_info,
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": _git_sha(),
            "task_count": len(results),
        },
        "metrics": metrics,
        "by_category": {cat: _rate(scores) for cat, scores in sorted(by_category.items())},
        "tasks": per_task,
    }
    if judge is not None:
        report["judge"] = judge["per_task"]
    return report


def _pct(entry: dict[str, Any]) -> str:
    if entry["rate"] is None:
        return "n/a"
    return f"{entry['rate'] * 100:.1f}%"


def to_markdown(report: dict[str, Any]) -> str:
    run, metrics = report["run"], report["metrics"]
    lines = [
        f"# Eval report: {run['provider']} / {run['model']}",
        "",
        f"- Tasks: {run['task_count']} from `{run['tasks_file']}`",
        f"- Retrieved text quoted as data: {run['quote_retrieved']}",
        f"- Tool transport: {run['transport']}",
        f"- Commit: {run.get('git_sha') or 'unknown'}; run at {run['timestamp']}",
        "",
        "| Metric | Score | Passed / applicable |",
        "|---|---|---|",
    ]
    for name in HEADLINE:
        entry = metrics[name]
        lines.append(f"| {name} | {_pct(entry)} | {entry['passed']} / {entry['n']} |")
    lines.append(f"| trajectory_score (mean) | {metrics['trajectory_score']} | |")
    lat = metrics["latency_ms"]
    lines.append(f"| latency p50 / p95 (ms) | {lat['p50']} / {lat['p95']} | |")
    lines.append(f"| tool calls per ticket (mean) | {metrics['tool_calls_mean']} | |")
    if "judge" in metrics:
        judge = metrics["judge"]
        lines.append(
            f"| judge pass rate | {judge['pass_rate'] * 100:.1f}% | {judge['passed']} / {judge['n']} |"
        )
    lines += ["", "## By category", "", "| Category | Task success | n |", "|---|---|---|"]
    for category, entry in report["by_category"].items():
        lines.append(f"| {category} | {_pct(entry)} | {entry['n']} |")
    failed = [t for t in report["tasks"] if t["failures"]]
    lines += ["", "## Failures", ""]
    if not failed:
        lines.append("Every task passed every applicable scorer.")
    for task in failed:
        lines.append(f"- **{task['id']}** ({task['category']}): " + "; ".join(task["failures"]))
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = report["run"]["timestamp"].replace(":", "").replace("-", "")
    json_path = out / f"eval_{stamp}.json"
    md_path = out / f"eval_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    shutil.copyfile(json_path, out / "latest.json")
    shutil.copyfile(md_path, out / "latest.md")
    return json_path, md_path
