"""Convert generation files written by `ff_modelgen` into call store records.

Each generation (a line of `generations/<model>.jsonl`, with the provider's full
response and usage) becomes the stored answer of the call that would produce it
(one user message with the prompt, the generation's `max_tokens` and
temperature), with `origin: "migrated"`, under `generation/<model>.jsonl`. The
source files are not changed.

    python -m feedback_forensics.router.migrate <dir with generations/> <store>
"""

import argparse
import json
from pathlib import Path

from loguru import logger

from feedback_forensics.router.datatypes import Call, Message, Params, Result
from feedback_forensics.router.store import ORIGIN_MIGRATED, Store
from feedback_forensics.router.wire import parse_usage

TASK = "generation"


def _finish_reason(full_response: dict) -> str | None:
    """Finish reason of a live (LangChain message) or batch (chat completion)
    generation, as `ff_modelgen.get_failure` reads it."""
    reason = (full_response.get("response_metadata") or {}).get("finish_reason")
    if reason is None:
        reason = (full_response.get("choices") or [{}])[0].get("finish_reason")
    return reason


def _usage(record: dict):
    """Usage of a generation, incl. cached tokens where the response has them."""
    full_response = record.get("full_response") or {}
    raw = (full_response.get("response_metadata") or {}).get(
        "token_usage"
    ) or full_response.get("usage")
    usage = parse_usage({"usage": raw if isinstance(raw, dict) else {}})
    saved = record.get("usage") or {}  # as saved by ff_modelgen (`get_usage`)
    fields = {
        name: (
            getattr(usage, name)
            if getattr(usage, name) is not None
            else saved.get(name)
        )
        for name in ("input_tokens", "output_tokens", "reasoning_tokens", "cost_usd")
    }
    return type(usage)(
        cached_tokens=usage.cached_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        **fields,
    )


def record_to_call(record: dict) -> tuple[Call, Result, dict]:
    """(call, answer, further store fields) of a generation record."""
    params = record.get("generation_params") or {}
    call = Call(
        model=record["model"],
        messages=(Message("user", record["prompt"]),),
        params=Params(
            # Not saved with older generations (None): their keys differ from
            # those of new calls, which state their settings
            max_tokens=params.get("max_tokens"),
            temperature=params.get("temperature"),
        ),
        tag={"prompt_id": record.get("prompt_id"), "failure": record.get("failure")},
    )
    route = params.get("api")
    if route not in ("live", "batch"):
        route = "batch" if record.get("batch_id") else None
    response = record.get("response")
    result = Result(
        key=call.key,
        tag=call.tag,
        text=response if isinstance(response, str) else "",
        finish_reason=_finish_reason(record.get("full_response") or {}),
        usage=_usage(record),
        route=route,
    )
    extra = dict(
        batch_id=record.get("batch_id"),
        provider=record.get("provider"),
        created=record.get("timestamp"),
    )
    return call, result, extra


def migrate_generations(source: str | Path, store: str | Path | Store) -> dict:
    """Add the generations in `source` (`generations/*.jsonl`, or the directory
    with these files itself) to a call store.

    Calls that already have an answer in the store are skipped, so migrating
    twice adds nothing.

    Returns:
        Counts: `migrated`, `skipped` (already in the store), `malformed` (lines
        that are not generation records).
    """
    source = Path(source)
    if (source / "generations").is_dir():
        source = source / "generations"
    files = sorted(source.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No generation files (*.jsonl) in {source}.")
    store = store if isinstance(store, Store) else Store(store)
    counts = {"migrated": 0, "skipped": 0, "malformed": 0}
    for file in files:
        with open(file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    call, result, extra = record_to_call(json.loads(line))
                except (ValueError, KeyError, TypeError, AttributeError):
                    counts["malformed"] += 1
                    continue
                if call.key in store:
                    counts["skipped"] += 1
                    continue
                store.put(call, result, task=TASK, origin=ORIGIN_MIGRATED, **extra)
                counts["migrated"] += 1
        logger.info(f"{file.name}: done ({counts})")
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="Directory with generations/*.jsonl")
    parser.add_argument("store", help="Directory of the call store")
    args = parser.parse_args(argv)
    counts = migrate_generations(args.source, args.store)
    logger.info(f"Migrated generations into {args.store}: {counts}")


if __name__ == "__main__":
    main()
