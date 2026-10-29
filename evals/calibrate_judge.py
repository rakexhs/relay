"""Measure how well the LLM judge agrees with human labels.

    .venv/bin/python -m evals.calibrate_judge --labels evals/data/judge_labels.jsonl --provider anthropic

Each label line is a judge case plus a human verdict:

    {"id": "...", "ticket": {...}, "action": {...}, "evidence": [...],
     "reference_reply": null, "human": {"verdict": "pass" | "fail", "notes": "..."}}

Reports raw agreement, Cohen's kappa, the confusion matrix (with "fail" as the positive class,
since catching bad replies is the judge's job) and every disagreement, so the rubric can be
tightened where the judge and humans part ways.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.judge import Judge, JudgeItem
from relay.config import Settings
from relay.providers import get_provider


def load_labels(path: str | Path) -> list[tuple[JudgeItem, str, str]]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            data = json.loads(line)
            human = data.pop("human")
            data.pop("labeler", None)
            rows.append((JudgeItem.model_validate(data), human["verdict"], human.get("notes", "")))
    return rows


def cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    labels = {label for pair in pairs for label in pair}
    expected = sum(
        (sum(a == label for a, _ in pairs) / n) * (sum(b == label for _, b in pairs) / n)
        for label in labels
    )
    if expected == 1:
        return 1.0
    return round((observed - expected) / (1 - expected), 4)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in results if r["judge"] != "error"]
    pairs = [(r["human"], r["judge"]) for r in graded]
    tp = sum(h == "fail" and j == "fail" for h, j in pairs)
    fp = sum(h == "pass" and j == "fail" for h, j in pairs)
    fn = sum(h == "fail" and j == "pass" for h, j in pairs)
    tn = sum(h == "pass" and j == "pass" for h, j in pairs)
    return {
        "n": len(results),
        "graded": len(graded),
        "agreement": round((tp + tn) / len(graded), 4) if graded else None,
        "cohens_kappa": cohens_kappa(pairs),
        "confusion": {"fail_caught": tp, "false_alarm": fp, "fail_missed": fn, "pass_agreed": tn},
        "fail_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "fail_precision": round(tp / (tp + fp), 4) if tp + fp else None,
    }


async def calibrate(judge: Judge, labels: list[tuple[JudgeItem, str, str]]) -> dict[str, Any]:
    results = []
    for item, human, notes in labels:
        verdict = await judge.grade(item)
        results.append(
            {
                "id": item.id,
                "human": human,
                "judge": verdict.verdict,
                "scores": verdict.scores,
                "rationale": verdict.rationale,
                "human_notes": notes,
            }
        )
    return {"summary": summarize(results), "cases": results}


def to_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    c = s["confusion"]
    lines = [
        f"# Judge calibration: {report['judge']['provider']} / {report['judge']['model']}",
        "",
        f"- Labels: {s['n']} from `{report['labels']}` ({s['graded']} graded)",
        f"- Agreement: {s['agreement']}  Cohen's kappa: {s['cohens_kappa']}",
        f"- Fail recall: {s['fail_recall']}  Fail precision: {s['fail_precision']}",
        "",
        "| | judge fail | judge pass |",
        "|---|---|---|",
        f"| human fail | {c['fail_caught']} | {c['fail_missed']} |",
        f"| human pass | {c['false_alarm']} | {c['pass_agreed']} |",
        "",
        "## Disagreements",
        "",
    ]
    disagreements = [r for r in report["cases"] if r["judge"] != r["human"]]
    if not disagreements:
        lines.append("None.")
    for r in disagreements:
        lines.append(
            f"- **{r['id']}**: human {r['human']}, judge {r['judge']} {r['scores']}. "
            f"Human note: {r['human_notes']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge vs. human agreement")
    parser.add_argument("--labels", default="evals/data/judge_labels.jsonl")
    parser.add_argument("--provider", choices=["stub", "openai", "anthropic"], default="stub")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--min-agreement", type=float, default=None)
    args = parser.parse_args(argv)

    settings = Settings(_env_file=None, provider=args.provider, model=args.model)
    judge = Judge(get_provider(settings))
    report = asyncio.run(calibrate(judge, load_labels(args.labels)))
    report["judge"] = judge.provider.describe()
    report["labels"] = args.labels
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"judge_calibration_{stamp}.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = to_markdown(report)
    (out / f"judge_calibration_{stamp}.md").write_text(markdown)
    print(markdown)
    agreement = report["summary"]["agreement"]
    if args.min_agreement is not None and (agreement or 0) < args.min_agreement:
        print(f"agreement {agreement} is below --min-agreement {args.min_agreement}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
