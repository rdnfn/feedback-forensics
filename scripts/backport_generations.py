"""Create generations of a new dataset version from an existing one.

Copies generations (and prompt config) unchanged, except for adding the
'failure' and 'usage' fields (see ff_modelgen.get_failure() and get_usage()) to
each record. Annotations are not copied, since they are recreated for the new
version.

Used to create v3 from v2 of the model personality dataset
(https://huggingface.co/datasets/rdnfn/ff-model-personality), dropping models
that were generated in v2 but never compared:

    python scripts/backport_generations.py hf-dataset/data/v2 hf-dataset/data/v3 \
        --exclude-models openrouter/mistralai/devstral-medium \
        openrouter/deepseek/deepseek-chat-v3-0324
"""

import argparse
import json
import pathlib
import shutil

from loguru import logger

from feedback_forensics.tools.ff_modelgen import get_failure, get_usage


def backport(
    source: pathlib.Path,
    target: pathlib.Path,
    exclude_models: list[str] = (),
):
    target_generations = target / "generations"
    if target_generations.exists() and any(target_generations.iterdir()):
        raise FileExistsError(f"{target_generations} is not empty, not overwriting.")

    # Generation files are named after the model (see ff_modelgen)
    excluded_files = {m.replace("/", "_") + ".jsonl" for m in exclude_models}
    source_files = sorted((source / "generations").glob("*.jsonl"))
    missing = excluded_files - {f.name for f in source_files}
    if missing:
        raise FileNotFoundError(f"No generations to exclude for: {sorted(missing)}")

    # Fails on git LFS pointer files (if LFS objects were not pulled)
    json.loads((source / "config/prompts.json").read_text())
    target_generations.mkdir(parents=True, exist_ok=True)
    (target / "config").mkdir(exist_ok=True)
    shutil.copyfile(source / "config/prompts.json", target / "config/prompts.json")

    for source_file in source_files:
        if source_file.name in excluded_files:
            logger.info(f"{source_file.stem}: excluded")
            continue
        records, num_malformed = [], 0
        for line in source_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                num_malformed += 1
                continue
            record["failure"] = get_failure(record["response"], record["full_response"])
            record["usage"] = get_usage(record["full_response"])
            records.append(record)

        (target_generations / source_file.name).write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
        failures = [r["failure"] for r in records if r["failure"]]
        log = logger.warning if failures or num_malformed else logger.info
        log(
            f"{source_file.stem}: {len(records)} generations, {len(failures)} failed"
            + (f", {num_malformed} malformed lines dropped" if num_malformed else "")
        )


def run():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=pathlib.Path, help="e.g. data/v2")
    parser.add_argument("target", type=pathlib.Path, help="e.g. data/v3")
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        default=[],
        help="Models whose generations are not copied, e.g. openrouter/x-ai/grok-4",
    )
    args = parser.parse_args()
    backport(args.source, args.target, args.exclude_models)


if __name__ == "__main__":
    run()
