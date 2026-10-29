"""A tool client wrapper that logs every call to a file and can hang or crash on purpose."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path


class SimulatedCrash(RuntimeError):
    pass


class RecordingToolClient:
    def __init__(self, inner, log_path: Path, *, hang_at: int | None = None, crash_at=None):
        self.inner = inner
        self.log_path = Path(log_path)
        self.hang_at = hang_at
        self.crash_at = crash_at
        self.count = 0

    async def list_tools(self):
        return await self.inner.list_tools()

    def requires_approval(self, name, args):
        return self.inner.requires_approval(name, args)

    def _log(self, event: str, name: str, args: dict) -> None:
        line = {"event": event, "n": self.count, "tool": name, "args": args}
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
            fh.flush()

    async def call(self, name, args, meta):
        self.count += 1
        self._log("start", name, args)
        if self.count == self.hang_at:
            await asyncio.sleep(3600)  # the test kills the process while we sit here
        if self.count == self.crash_at:
            self.crash_at = None
            raise SimulatedCrash(f"crashed during call {self.count} ({name})")
        result = await self.inner.call(name, args, meta)
        self._log("done", name, args)
        return result


def read_log(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
