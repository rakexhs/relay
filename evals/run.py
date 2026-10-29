"""Run the trajectory eval.

.venv/bin/python -m evals.run --tasks evals/data/tasks.sample.jsonl --provider stub
.venv/bin/python -m evals.run --provider anthropic --judge
.venv/bin/python -m evals.run --no-quote          # ablation: retrieved text not quoted
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from evals.harness import run_suite
from evals.report import build_report, write_report
from evals.schema import load_tasks
from relay.config import Settings
from relay.providers import get_provider


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tasks", default="evals/data/tasks.sample.jsonl")
    parser.add_argument("--provider", choices=["stub", "openai", "anthropic"], default="stub")
    parser.add_argument("--model", default="")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--transport", choices=["local", "mcp"], default="local")
    parser.add_argument("--no-quote", action="store_true", help="do not quote retrieved text")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--category", action="append", help="only run these categories")
    parser.add_argument(
        "--judge", action="store_true", help="also grade replies with the LLM judge"
    )
    parser.add_argument("--judge-provider", default=None)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_tasks(args.tasks)
    if args.category:
        tasks = [t for t in tasks if t.category in set(args.category)]
    overrides: dict[str, Any] = {"provider": args.provider, "quote_retrieved": not args.no_quote}
    if args.model:
        overrides["model"] = args.model
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    settings = Settings(_env_file=None, **overrides)
    provider = get_provider(settings)
    results = await run_suite(tasks, provider, settings, transport=args.transport)

    judge = None
    if args.judge:
        from evals.judge import Judge, judge_results

        judge_settings = Settings(_env_file=None, provider=args.judge_provider or args.provider)
        judge = await judge_results(Judge(get_provider(judge_settings)), tasks, results)

    run_info = {
        "provider": provider.name,
        "model": provider.model,
        "tasks_file": args.tasks,
        "quote_retrieved": settings.quote_retrieved,
        "transport": args.transport,
        "max_steps": settings.max_steps,
    }
    return build_report(tasks, results, run_info, judge)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(main_async(args))
    json_path, md_path = write_report(report, args.out)
    metrics = report["metrics"]
    print(f"tasks: {report['run']['task_count']}  provider: {report['run']['provider']}")
    for name in ("task_success", "trajectory_match", "error_recovery", "injection_resistance"):
        entry = metrics[name]
        rate = "n/a" if entry["rate"] is None else f"{entry['rate'] * 100:.1f}%"
        print(f"  {name:<22} {rate:>7}  ({entry['passed']}/{entry['n']})")
    lat = metrics["latency_ms"]
    print(f"  {'latency p50/p95 ms':<22} {lat['p50']} / {lat['p95']}")
    print(f"report: {json_path}\nsummary: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
