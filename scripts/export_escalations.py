"""Pull resolved escalations from a running Relay API and save them as eval tasks.

    .venv/bin/python scripts/export_escalations.py --api http://localhost:8000 \
        --out evals/data/tasks.escalations.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--append", action="store_true", help="add to an existing task file")
    args = parser.parse_args(argv)

    response = httpx.get(f"{args.api.rstrip('/')}/escalations/export", timeout=30)
    response.raise_for_status()
    new = [json.loads(line) for line in response.text.splitlines() if line.strip()]

    existing_ids: set[str] = set()
    if args.append and args.out.exists():
        with args.out.open(encoding="utf-8") as fh:
            existing_ids = {json.loads(line)["id"] for line in fh if line.strip()}
    fresh = [task for task in new if task["id"] not in existing_ids]
    with args.out.open("a" if args.append else "w", encoding="utf-8") as fh:
        for task in fresh:
            fh.write(json.dumps(task) + "\n")
    print(f"wrote {len(fresh)} task(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
