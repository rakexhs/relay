import json
from pathlib import Path

from evals.schema import Task
from scripts.build_tasks import build, scrub, thread_to_task

ROOT = Path(__file__).resolve().parent.parent
THREADS = ROOT / "evals" / "data" / "forum_threads.sample.jsonl"


def _threads():
    return [json.loads(line) for line in THREADS.read_text().splitlines() if line.strip()]


def test_scrub_removes_contact_details_and_secrets():
    text = scrub("mail me at a.b@example.org, see https://x.example/y, key key_abcdef123456")
    assert "@" not in text and "https" not in text and "key_abcdef" not in text


def test_threads_become_valid_tasks_with_expected_outcomes():
    tasks = {t["source"]["thread_id"]: t for t in build(_threads(), inject_rate=0, seed=1)}
    assert "th-1006" not in tasks  # too short to be a ticket
    assert tasks["th-1001"]["expect"]["outcome"] == "resolved"
    assert tasks["th-1003"]["expect"]["reason"] == "needs_refund"
    assert tasks["th-1004"]["expect"]["reason"] == "low_confidence"
    assert "[email]" in tasks["th-1001"]["ticket"]["body"]
    for task in tasks.values():
        Task.model_validate(task)


def test_injection_variants_carry_a_canary_and_forbidden_action():
    tasks = build(_threads(), inject_rate=1.0, seed=3)
    injected = [t for t in tasks if t["category"] == "injection"]
    assert len(injected) == 3  # one per resolvable thread
    for task in injected:
        Task.model_validate(task)
        planted = task["fixtures"]["docs"][-1]["body"]
        assert task["injection"]["canary"].lower() in planted.lower()
        assert task["forbidden"]


def test_duplicate_titles_are_dropped():
    thread = _threads()[0]
    assert thread_to_task(thread) is not None
    assert len(build([thread, dict(thread, thread_id="copy")], 0, 1)) == 1
