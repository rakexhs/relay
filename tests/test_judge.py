import asyncio
from pathlib import Path

from evals.calibrate_judge import calibrate, cohens_kappa, load_labels
from evals.harness import run_task
from evals.judge import Judge, item_from_result, judge_results, parse_verdict
from evals.schema import load_tasks
from relay.config import Settings
from relay.providers import StubProvider

ROOT = Path(__file__).resolve().parent.parent


def test_parse_verdict_enforces_the_rubric_when_verdict_is_missing():
    text = 'Sure: {"scores": {"grounded": 5, "resolves": 4, "safe": 3, "tone": 5}}'
    verdict = parse_verdict("x", text)
    assert verdict.verdict == "fail"  # safe must be >= 4
    assert parse_verdict("x", "no json here").verdict == "error"


def test_cohens_kappa():
    assert cohens_kappa([("pass", "pass"), ("fail", "fail")]) == 1.0
    assert cohens_kappa([("pass", "fail"), ("fail", "pass")]) == -1.0
    assert cohens_kappa([]) is None


def test_calibration_reports_agreement_and_disagreements():
    labels = load_labels(ROOT / "evals" / "data" / "judge_labels.jsonl")
    report = asyncio.run(calibrate(Judge(StubProvider()), labels))
    summary = report["summary"]
    assert summary["n"] == len(labels)
    confusion = summary["confusion"]
    assert sum(confusion.values()) == summary["graded"]
    assert 0 <= summary["agreement"] <= 1


def test_judge_grades_eval_results():
    tasks = [
        t
        for t in load_tasks(ROOT / "evals" / "data" / "tasks.sample.jsonl")
        if t.id in {"download-invoices", "refund-duplicate-charge"}
    ]
    settings = Settings(_env_file=None)
    results = [asyncio.run(run_task(t, StubProvider(), settings)) for t in tasks]
    item = item_from_result(tasks[0], results[0])
    assert item.action["type"] in {"reply", "escalation"}
    graded = asyncio.run(judge_results(Judge(StubProvider()), tasks, results))
    assert graded["summary"]["n"] == 2
    assert {v["verdict"] for v in graded["per_task"]} <= {"pass", "fail"}
