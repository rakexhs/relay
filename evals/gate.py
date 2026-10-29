"""Fail CI when the eval falls below thresholds or regresses against the baseline.

    .venv/bin/python -m evals.gate --report reports/latest.json \
        --baseline evals/baseline.json --config evals/gate.yaml
    .venv/bin/python -m evals.gate --report reports/latest.json --write-baseline evals/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

GATED = ("task_success", "trajectory_match", "error_recovery", "injection_resistance")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def rates(report: dict[str, Any]) -> dict[str, float | None]:
    return {name: report["metrics"][name]["rate"] for name in GATED}


def check(
    report: dict[str, Any], config: dict[str, Any], baseline: dict[str, Any] | None
) -> list[tuple[str, bool, str]]:
    """Return (check name, ok, detail) rows."""
    rows = []
    current = rates(report)
    for name, minimum in (config.get("minimums") or {}).items():
        value = current.get(name)
        if value is None:
            rows.append((f"{name} >= {minimum:.2f}", True, "not measured in this run"))
            continue
        rows.append((f"{name} >= {minimum:.2f}", value >= minimum, f"{value:.4f}"))
    if baseline:
        allowed = float(config.get("max_regression_points", 0)) / 100
        for name in config.get("no_regression", []):
            base, value = baseline.get("metrics", {}).get(name), current.get(name)
            if base is None or value is None:
                continue
            ok = value >= base - allowed - 1e-9
            rows.append(
                (
                    f"{name} within {allowed * 100:.0f} pts of baseline",
                    ok,
                    f"{value:.4f} vs baseline {base:.4f}",
                )
            )
    return rows


def write_baseline(report: dict[str, Any], path: str | Path) -> None:
    baseline = {
        "provider": report["run"]["provider"],
        "model": report["run"]["model"],
        "tasks_file": report["run"]["tasks_file"],
        "task_count": report["run"]["task_count"],
        "git_sha": report["run"].get("git_sha"),
        "metrics": rates(report),
    }
    Path(path).write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eval regression gate")
    parser.add_argument("--report", default="reports/latest.json")
    parser.add_argument("--baseline", default="evals/baseline.json")
    parser.add_argument("--config", default="evals/gate.yaml")
    parser.add_argument("--write-baseline", metavar="PATH")
    parser.add_argument("--summary", metavar="PATH", help="append a markdown table here")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if args.write_baseline:
        write_baseline(report, args.write_baseline)
        print(f"baseline written to {args.write_baseline}")
        return 0

    config = load_yaml(args.config)
    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    if baseline and baseline.get("tasks_file") != report["run"]["tasks_file"]:
        print("note: baseline was recorded on a different task file; skipping regression checks")
        baseline = None
    rows = check(report, config, baseline)

    table = ["| Check | Result | Value |", "|---|---|---|"]
    for name, ok, detail in rows:
        table.append(f"| {name} | {'pass' if ok else 'FAIL'} | {detail} |")
    print("\n".join(table))
    if args.summary:
        with Path(args.summary).open("a", encoding="utf-8") as fh:
            fh.write("## Eval gate\n\n" + "\n".join(table) + "\n")
    failed = [name for name, ok, _ in rows if not ok]
    if failed:
        print(f"\neval gate failed: {', '.join(failed)}")
        return 1
    print("\neval gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
