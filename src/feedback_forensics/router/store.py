"""The call store: raw answers on disk, in a layout that can be uploaded as part
of a dataset and read without this code (see `STORE_README`)."""

import hashlib
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

from feedback_forensics.router.datatypes import Call, Message, Params, Result, Usage
from feedback_forensics.router.wire import split_pieces

ORIGIN_ROUTER, ORIGIN_MIGRATED = "router", "migrated"
DEFAULT_TASK = "calls"
MAX_PIECES_FILE_BYTES = 64 * 1024 * 1024

STORE_README = """\
# Call store

Raw answers of LLM calls, written by the model-call router of Feedback Forensics
(`feedback_forensics.router`). Everything is JSON Lines (UTF-8, one JSON object
per line) and can be read without that code.

```
README.md                    this file
pieces/part-000.jsonl        each distinct piece of prompt text, once
<task>/<model>.jsonl         one line per call, e.g. generation/<model>.jsonl
<task>/<model>/<group>.jsonl ... or, for calls added with a group, e.g.
                             annotation/<annotator>/<annotated model>.jsonl
```

`<task>` and `<group>` are chosen by the caller (`Router.add(calls, task=...,
group=...)`), `<model>` is the model of the call. In file names, every run of
characters other than letters, digits, `.`, `_` and `-` is replaced by `_`.
A directory `.router/` may also be present: working state of the router (pending
batches, the plan of an unfinished run, a log of call states). It holds no
answers and can be left out of uploads.

## Pieces (`pieces/part-*.jsonl`)

`{"hash": ..., "text": ...}`: `hash` is the SHA-256 (hex) of `text` (UTF-8).
Prompts are stored as pieces so that text shared by many calls is stored once.

## Calls

Fields of a line:

- `key`: identifies the call. SHA-256 (hex) of the UTF-8 JSON, with sorted keys,
  no spaces after `,` and `:`, and non-ASCII characters not escaped, of
  `{"messages": [[role, text], ...], "model": model, "params": params}`.
  It does not depend on how the prompt is cut into pieces.
- `task`, `group`: as given by the caller (`group` may be null).
- `model`: e.g. `openrouter/<provider>/<model>`.
- `params`: `max_tokens`, `reasoning_effort`, `temperature` (null: provider default).
- `messages`: the prompt, `[{"role": ..., "pieces": [hash, ...]}, ...]`. The text
  of a message is the texts of its pieces joined (without separator). Pieces
  end where a message ends and where the router placed a cache breakpoint.
- `tag`: the caller's data for the call, as given.
- `text`: the answer. `finish_reason`: as reported by the provider.
- `usage`: `input_tokens` (including `cached_tokens`, read from the provider's
  prompt cache, and `cache_write_tokens`, written to it), `output_tokens`
  (including `reasoning_tokens`), `cost_usd` (as reported by the provider);
  null where not reported.
- `route`: how the answer was obtained, `live` or `batch` (null: not known).
- `batch_id`: id of the batch (null for live calls).
- `provider`: the provider that served the call, if reported.
- `created`: when the answer was received (ISO 8601).
- `origin`: `router`, or `migrated` (converted from files written before the
  router existed).

If a key appears on more than one line, the last line is the answer in use:
earlier ones were answers that the caller's check rejected and that were asked
again. Answers are stored as received, never parsed. A file whose last line is
incomplete (an interrupted write) is valid up to that line.
"""


def piece_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_name(name: str) -> str:
    """Name of a file or directory for a task, model or group."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).lstrip(".")
    if not name:
        raise ValueError("Empty name for a store file.")
    return name


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _drop_incomplete_last_line(path: Path) -> None:
    """Remove a last line without line end (left by an interrupted write)."""
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        # Find the end of the last complete line
        position, last_line_end = size, 0
        while position > 0:
            step = min(65536, position)
            position -= step
            f.seek(position)
            found = f.read(step).rfind(b"\n")
            if found >= 0:
                last_line_end = position + found + 1
                break
        logger.warning(f"Removing an incomplete last line from {path}.")
        f.truncate(last_line_end)


def _append(path: Path, lines: list[str]) -> None:
    """Append complete lines and make sure they are on disk."""
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in lines))
        f.flush()
        os.fsync(f.fileno())


class Store:
    """Answers by call key.

    One process writes to a store at a time (the router holds a lock while it
    runs). Every write appends complete lines and is flushed to disk before the
    call counts as answered; the pieces of a call are written before its line.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        readme = self.path / "README.md"
        if not readme.exists():
            readme.write_text(STORE_README, encoding="utf-8")
        self._pieces_dir = self.path / "pieces"
        self._pieces_dir.mkdir(exist_ok=True)
        self._piece_hashes: set[str] = set()
        self._pieces_file = self._pieces_dir / "part-000.jsonl"
        for file in sorted(self._pieces_dir.glob("part-*.jsonl")):
            _drop_incomplete_last_line(file)
            self._pieces_file = file
            with open(file, encoding="utf-8") as f:
                for line in f:
                    # Lines start with '{"hash": "<64 characters>"'
                    self._piece_hashes.add(line[10:74])
        self._records: dict[str, dict] = {}
        for file in self.call_files():
            _drop_incomplete_last_line(file)
            with open(file, encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        self._records[record["key"]] = record
                    except (ValueError, KeyError, TypeError):
                        logger.warning(f"Skipping malformed line in {file}.")

    def call_files(self) -> list[Path]:
        """The files with calls (everything but pieces and working state)."""
        return sorted(
            file
            for file in self.path.rglob("*.jsonl")
            if file.relative_to(self.path).parts[0] not in ("pieces", ".router")
        )

    def file_for(self, call: Call, task: str = DEFAULT_TASK, group=None) -> Path:
        """`<task>/<model>.jsonl`, or `<task>/<model>/<group>.jsonl` with a group."""
        directory = self.path / safe_name(task)
        if group is None:
            return directory / (safe_name(call.model) + ".jsonl")
        return directory / safe_name(call.model) / (safe_name(group) + ".jsonl")

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def keys(self) -> Iterator[str]:
        return iter(self._records)

    def record(self, key: str) -> dict | None:
        """The stored line of a call (None if it has no answer)."""
        return self._records.get(key)

    def result(self, key: str, tag: Any = None, route: str = "store") -> Result | None:
        """The stored answer of a call (None if it has none)."""
        record = self._records.get(key)
        if record is None:
            return None
        return Result(
            key=key,
            tag=tag,
            text=record.get("text"),
            finish_reason=record.get("finish_reason"),
            usage=Usage(**(record.get("usage") or {})),
            route=route,
        )

    def put(
        self,
        call: Call,
        result: Result,
        task: str = DEFAULT_TASK,
        group: str | None = None,
        breakpoints=(),
        origin: str = ORIGIN_ROUTER,
        batch_id: str | None = None,
        provider: str | None = None,
        created: str | None = None,
    ) -> dict:
        """Store the answer of a call (replaces an earlier answer of the call)."""
        if result.error is not None or result.text is None:
            raise ValueError("Only answers are stored, not failed calls.")
        messages, new_pieces = [], []
        for message, pieces in zip(call.messages, split_pieces(call, breakpoints)):
            hashes = []
            for text, _ in pieces:
                digest = piece_hash(text)
                hashes.append(digest)
                if digest not in self._piece_hashes:
                    self._piece_hashes.add(digest)
                    new_pieces.append(
                        json.dumps({"hash": digest, "text": text}, ensure_ascii=False)
                    )
            messages.append({"role": message.role, "pieces": hashes})
        if new_pieces:
            if (
                self._pieces_file.exists()
                and self._pieces_file.stat().st_size > MAX_PIECES_FILE_BYTES
            ):
                number = int(self._pieces_file.stem.split("-")[1]) + 1
                self._pieces_file = self._pieces_dir / f"part-{number:03d}.jsonl"
            _append(self._pieces_file, new_pieces)

        record = {
            "key": call.key,
            "task": task,
            "group": group,
            "model": call.model,
            "params": call.params.as_dict(),
            "messages": messages,
            "tag": call.tag,
            "text": result.text,
            "finish_reason": result.finish_reason,
            "usage": asdict(result.usage),
            "route": result.route if result.route in ("live", "batch") else None,
            "batch_id": batch_id,
            "provider": provider,
            "created": created or _now(),
            "origin": origin,
        }
        file = self.file_for(call, task, group)
        file.parent.mkdir(parents=True, exist_ok=True)
        _append(file, [json.dumps(record, ensure_ascii=False, default=str)])
        self._records[call.key] = record
        return record

    def call(self, key: str) -> Call | None:
        """The call of a stored answer, rebuilt from its pieces."""
        record = self._records.get(key)
        if record is None:
            return None
        needed = {h for m in record["messages"] for h in m["pieces"]}
        texts = {}
        for file in sorted(self._pieces_dir.glob("part-*.jsonl")):
            with open(file, encoding="utf-8") as f:
                for line in f:
                    if line[10:74] in needed:
                        texts[line[10:74]] = json.loads(line)["text"]
        return Call(
            model=record["model"],
            messages=tuple(
                Message(m["role"], "".join(texts[h] for h in m["pieces"]))
                for m in record["messages"]
            ),
            params=Params(**record["params"]),
            tag=record.get("tag"),
        )
