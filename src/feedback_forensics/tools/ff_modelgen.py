"""Tools to generate responses from models for comparison of model personalities."""

import pathlib
import json
from datetime import datetime
import io
from tqdm import tqdm
import feedback_forensics.models.main
from feedback_forensics.data.ap_format import hash_string
import asyncio
import aiofiles
from typing import List
import time
import os
import functools
import concurrent.futures
import pandas as pd
import requests
from loguru import logger
from feedback_forensics.models import openrouter
from feedback_forensics.models.openrouter import (
    OpenRouterAccountError,
    OpenRouterError,
)

NONBATCH_SUFFIX = ":nonbatch"  # Our suffix to disable batch API for model


FAILED_FINISH_REASONS = {
    "length": "length",  # cut off at max tokens (e.g. all used for reasoning)
    "error": "error",  # provider error during generation
    "content_filter": "content_filter",  # stopped by provider's content filter
    "tool_calls": "tool_calls",  # model tried to call a tool (none offered)
}


def get_failure(response: str | None, full_response: dict) -> str | None:
    """Get failure reason of a generation, None if successful.

    Failed generations are saved (and thus not retried, e.g. a model running into
    its token limit would most likely do so again), but excluded from comparisons
    (except non-empty reference generations, see create_pairwise_datasets()).

    Failures: a finish reason in FAILED_FINISH_REASONS (whether or not the response
    has text), otherwise 'empty' if the response is empty or only whitespace.

    Args:
        response: Response text of the generation.
        full_response: Full response of live (langchain message) or batch
            (OpenAI chat completion) generation.
    """
    finish_reason = (full_response.get("response_metadata") or {}).get("finish_reason")
    if finish_reason is None:
        choices = full_response.get("choices") or [{}]
        finish_reason = choices[0].get("finish_reason")
    if finish_reason in FAILED_FINISH_REASONS:
        return FAILED_FINISH_REASONS[finish_reason]
    if not isinstance(response, str) or not response.strip():
        return "empty"
    return None


def get_usage(full_response: dict) -> dict:
    """Get token usage and cost of a generation (None where not reported).

    `cost_usd` is the cost reported by OpenRouter (incl. batch discount), which
    other providers do not report. Output tokens include reasoning tokens.

    Args:
        full_response: Full response of live (langchain message) or batch
            (OpenAI chat completion) generation.
    """
    # OpenAI-compatible usage (incl. OpenRouter): live and batch generations
    usage = (full_response.get("response_metadata") or {}).get(
        "token_usage"
    ) or full_response.get("usage")
    if isinstance(usage, dict) and usage.get("prompt_tokens") is not None:
        cost = usage.get("cost")
        return {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "cost_usd": float(cost) if cost is not None else None,
        }
    # Otherwise LangChain's provider-independent usage (e.g. Anthropic API)
    usage_metadata = full_response.get("usage_metadata") or {}
    return {
        "input_tokens": usage_metadata.get("input_tokens"),
        "output_tokens": usage_metadata.get("output_tokens"),
        "reasoning_tokens": (usage_metadata.get("output_token_details") or {}).get(
            "reasoning"
        ),
        "cost_usd": None,
    }


TEMPERATURE = 0.0  # Used for all generations (live and batch)

# HTTP status codes of requests rejected due to the API key or account, not the
# request: unauthorized (invalid key) and payment required (no credits left, or
# spending limit of the key reached). All further requests would be rejected too.
ACCOUNT_ERROR_STATUS_CODES = (401, 402)


class LiveGenerationAccountError(RuntimeError):
    """Live generation stopped: the API rejected the API key or account (see
    `ACCOUNT_ERROR_STATUS_CODES`). Generations completed before are saved."""


def _generation_params(api: str, max_tokens: int | None) -> dict:
    """Parameters a generation was created with (saved with each record)."""
    return {"api": api, "max_tokens": max_tokens, "temperature": TEMPERATURE}


def _get_provider_info(generation_id: str | None) -> dict:
    """Get provider and exact model version that served an OpenRouter generation.

    Best effort: OpenRouter may take a moment to register a generation, and
    missing provider info should never lose a (paid) generation.
    """
    if generation_id:
        error = None
        for attempt in range(5):
            time.sleep(2**attempt)  # 1s, 2s, 4s, ...
            try:
                data = openrouter.openrouter_request(
                    "GET", "/generation", params={"id": generation_id}
                )["data"]
            except (OpenRouterError, requests.RequestException) as e:
                error = e
                continue
            # Right after generation, fields may not be populated yet
            if data.get("provider_name"):
                return {
                    "provider": data["provider_name"],
                    "provider_model": data.get("model"),
                }
            error = "provider not populated yet"
        logger.warning(f"No provider info for generation {generation_id}: {error}")
    return {"provider": None, "provider_model": None}


def _log_failure(record: dict):
    logger.warning(
        f"Generation failed ({record['failure']}) for {record['model']} on prompt "
        f"{record['prompt_id']}: saved as failed, not retried."
    )


def run_model_on_prompts(prompts: list[str], model_name: str, output_path: str):

    output_path = pathlib.Path(output_path)

    output_file = output_path / "generations" / (model_name + ".jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Function to get already processed prompt IDs
    def get_processed_ids():
        processed_ids = set()
        if output_file.exists():
            with open(output_file, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        processed_ids.add(record["prompt_id"])
                    except:
                        pass
        return processed_ids

    # Get already processed prompt IDs
    processed_ids = get_processed_ids()

    model = feedback_forensics.models.main.get_model(model_name, max_tokens=16384)

    # Open file in append mode to continue where we left off
    with open(output_file, "a") as f:
        for prompt in tqdm(prompts, total=len(prompts)):
            prompt_id = hash_string(prompt)

            # Skip if already processed
            if prompt_id in processed_ids:
                continue

            try:
                # Create message and generate response
                msg = {"role": "user", "content": prompt}
                generation = model.invoke([msg])

                # Create a record with prompt, response, and metadata
                record = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "response": generation.content,
                    "full_response": dict(generation),
                    "model": model_name,
                }

                # Write the record to the file
                f.write(json.dumps(record) + "\n")

                # Add to processed set
                processed_ids.add(prompt_id)

            except Exception as e:
                print(f"Error processing prompt {prompt_id}: {e}")


async def run_model_on_prompts_async(
    prompts: List[str],
    model_name: str,
    output_path: str,
    max_concurrent: int = 5,
    max_tokens: int = 10000,
):
    """Process prompts concurrently using asyncio with controlled concurrency.

    Args:
        prompts: List of prompts to process
        model_name: Name of the model to use
        output_path: Directory to save results
        max_concurrent: Maximum number of concurrent tasks

    Raises:
        LiveGenerationAccountError: if the API rejected the API key or account
            (HTTP 401 or 402, e.g. out of credits). No further prompts are
            started, as they would fail too. Generations already completed (and
            those in flight) are saved first. Prompts failing with other errors
            are skipped (not saved, retried on next run).

    Example:

    ```python
    import asyncio

    # Call the async function with asyncio.run
    asyncio.run(run_model_on_prompts_async(
        prompts=your_prompts,
        model_name="your-model",
        output_path="output/directory",
        max_concurrent=10  # Adjust based on your needs
    ))
    ```
    """
    output_path = pathlib.Path(output_path)
    output_file = (
        output_path / "generations" / (model_name.replace("/", "_") + ".jsonl")
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Track stats
    start_time = time.time()
    total_prompts = len(prompts)
    completed = 0
    skipped = 0
    errors = {}  # error message -> number of prompts failed with it
    account_error = None  # message of first API key or account error
    num_not_started = 0  # prompts not started due to account error

    # Get already processed prompt IDs
    processed_ids = set()
    if output_file.exists():
        async with aiofiles.open(output_file, "r") as f:
            async for line in f:
                try:
                    record = json.loads(line)
                    processed_ids.add(record["prompt_id"])
                except:
                    pass

        # Print how many prompts we're skipping
        skipped = len(processed_ids.intersection([hash_string(p) for p in prompts]))
        if skipped > 0:
            print(f"Skipping {skipped} already processed prompts")

    _ensure_trailing_newline(output_file)

    # Create file lock to prevent race conditions
    file_lock = asyncio.Lock()
    model = feedback_forensics.models.main.get_model(
        model_name, temp=TEMPERATURE, max_tokens=max_tokens
    )

    # Create a thread pool for running synchronous model calls
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)

    async def process_prompt(prompt, index):
        nonlocal completed, account_error, num_not_started
        prompt_id = hash_string(prompt)

        # Skip if already processed
        if prompt_id in processed_ids:
            return None

        if account_error is not None:
            # Would be rejected like the prompts before
            num_not_started += 1
            return {"index": index, "prompt_id": prompt_id, "success": False}

        try:
            # Create message
            msg = {"role": "user", "content": prompt}

            # Run model in thread pool to not block the event loop
            loop = asyncio.get_event_loop()
            generation = await loop.run_in_executor(
                thread_pool, functools.partial(model.invoke, [msg])
            )

            # Create a record with prompt, response, and metadata
            full_response = dict(generation)
            record = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": generation.content,
                "full_response": full_response,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "failure": get_failure(generation.content, full_response),
                "usage": get_usage(full_response),
                "generation_params": _generation_params("live", max_tokens),
            }
            if model_name.startswith("openrouter/"):
                record.update(
                    await loop.run_in_executor(
                        thread_pool,
                        _get_provider_info,
                        full_response.get("response_metadata", {}).get("id"),
                    )
                )
            if record["failure"]:
                _log_failure(record)

            # Write the record to the file with lock to prevent race conditions
            async with file_lock:
                async with aiofiles.open(output_file, "a") as f:
                    await f.write(json.dumps(record) + "\n")
                    await f.flush()

            # Update counters
            completed += 1
            processed_ids.add(prompt_id)

            # Return success message
            return {
                "index": index,
                "prompt_id": prompt_id,
                "success": True,
                "time": time.time(),
            }

        except Exception as e:
            # Not aborting: other prompts (and other models) may still succeed,
            # except after API key or account errors (see below).
            # Each distinct error is only shown once (e.g. key limit hits all prompts)
            message = _describe_live_error(e, model_name)
            errors[message] = errors.get(message, 0) + 1
            if getattr(e, "status_code", None) in ACCOUNT_ERROR_STATUS_CODES:
                if account_error is None:
                    account_error = message
                    logger.error(
                        f"Generation failed for {model_name} on prompt "
                        f"{prompt_id[:8]}, not starting further prompts: {message}"
                    )
            elif errors[message] == 1:
                logger.error(
                    f"Generation failed for {model_name} on prompt {prompt_id[:8]} "
                    f"(shown once, continuing with other prompts): {message}"
                )
            return {
                "index": index,
                "prompt_id": prompt_id,
                "success": False,
                "error": str(e),
                "time": time.time(),
            }

    # Process prompts with a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_process_prompt(prompt, index):
        async with semaphore:
            return await process_prompt(prompt, index)

    # Create tasks with progress tracking
    tasks = []
    for i, prompt in enumerate(prompts):
        task = asyncio.create_task(bounded_process_prompt(prompt, i))
        tasks.append(task)

    # Setup progress bar that updates as tasks complete
    pbar = tqdm(total=len(prompts) - skipped)

    # Process results as they complete
    for future in asyncio.as_completed(tasks):
        result = await future
        if result is not None:  # Skip already processed prompts
            pbar.update(1)

    pbar.close()
    thread_pool.shutdown()

    # Print stats
    elapsed = time.time() - start_time
    print(f"Completed processing {completed} prompts with model {model_name}")
    for message, count in errors.items():
        logger.error(
            f"{count} of {total_prompts} prompts failed for {model_name} (not "
            f"saved, retried on next run): {message}"
        )
    print(
        f"Total time: {elapsed:.2f}s, Average: {elapsed/max(1, completed):.2f}s per prompt"
    )
    if account_error is not None:
        raise LiveGenerationAccountError(
            f"Live generation for {model_name} stopped by an API key or account "
            f"error. {completed} generations of this run were saved, "
            f"{num_not_started} prompts were not started.\n{account_error}"
        )


def strip_nonbatch_suffix(model_name: str) -> str:
    """Remove suffix, e.g. 'openrouter/openai/gpt-6-sol:nonbatch' -> 'openrouter/openai/gpt-6-sol'."""
    return model_name.removesuffix(NONBATCH_SUFFIX)


def get_pending_batch_models(output_path: str) -> List[str]:
    """Get all models with a pending batch (according to state files)."""
    state_dir = pathlib.Path(output_path) / "batches"
    models = []
    for state_file in sorted(state_dir.glob("*.json")):
        with open(state_file) as f:
            models.append(json.load(f)["model"])
    return models


def get_batch_models(model_names: List[str], output_path: str) -> List[str]:
    """Get models to generate via OpenRouter batch API (at reduced price).

    By default, uses batch API for new OpenRouter models (without any generations
    yet) if a batch variant is available. Models with suffix ':nonbatch' use live
    API. Missing generations of existing models (e.g. failed prompts) use live API
    to avoid long waits. Models with a pending batch are always included (even with
    ':nonbatch' suffix or if batch variant is no longer listed), so that already
    paid batch results are never lost.

    Returns:
        List of model names (without suffix) to generate via batch API.
    """
    pending = set(get_pending_batch_models(output_path))
    batch_models, candidates = [], []
    for model_name in model_names:
        stripped_name = strip_nonbatch_suffix(model_name)
        if stripped_name in pending:
            batch_models.append(stripped_name)
            continue
        if model_name.endswith(NONBATCH_SUFFIX) or not model_name.startswith(
            "openrouter/"
        ):
            continue
        output_file, _ = _get_batch_paths(output_path, model_name)
        # (an empty file, e.g. left by a failed batch, is no generation either)
        if not _get_processed_ids(output_file):
            candidates.append(model_name)

    if not candidates:
        return batch_models
    with_batch_variant = openrouter.get_models_with_batch_variant(candidates)
    for model_name in candidates:
        if model_name in with_batch_variant:
            batch_models.append(model_name)
        else:
            logger.info(f"No batch variant available for {model_name}, using live API.")
    return batch_models


def _get_batch_paths(output_path: str, model_name: str):
    """Get generations file and batch state file for a model."""
    output_path = pathlib.Path(output_path)
    file_stem = model_name.replace("/", "_")
    output_file = output_path / "generations" / (file_stem + ".jsonl")
    state_file = output_path / "batches" / (file_stem + ".json")
    return output_file, state_file


def _get_processed_ids(output_file: pathlib.Path) -> set:
    processed_ids = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["prompt_id"])
                except:
                    pass
    return processed_ids


def _ensure_trailing_newline(file: pathlib.Path):
    """Terminate a line torn by an interrupted write, so appends don't corrupt it."""
    if file.exists() and file.stat().st_size > 0:
        with open(file, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")


def _describe_live_error(error: Exception, model_name: str) -> str:
    """Get legible message for error raised by a live (non-batch) generation."""
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if model_name.startswith("openrouter/") and isinstance(status_code, int):
        return str(
            openrouter.get_openrouter_error(status_code, getattr(response, "text", ""))
        )
    return f"{type(error).__name__}: {error}"


def _recover_batch(state: dict, state_file: pathlib.Path, prompts: List[str]) -> bool:
    """Find the batch of a state whose submission was interrupted before its batch
    id was saved (see openrouter.recover_batch()).

    Returns:
        False if it is not known yet whether the batch was created. Otherwise True:
        the batch id is saved to the state, or the state is removed (not created).
    """
    model_name = state["model"]
    resolved, batch_id = openrouter.recover_batch(
        model_name, state, custom_ids=[hash_string(p) for p in prompts]
    )
    if batch_id:
        state["batch_id"] = batch_id
        openrouter.write_json_atomic(state_file, state)
        logger.info(f"Recovered batch {batch_id} for {model_name}.")
    elif resolved:
        state_file.unlink()
        logger.info(f"No batch found for {model_name}, to be submitted again.")
    else:
        openrouter.write_json_atomic(state_file, state)
        logger.info(
            f"Not known yet whether a batch was created for {model_name} (batches "
            "that may be it have no results yet)."
        )
    return resolved


def submit_batch(
    prompts: List[str], model_name: str, output_path: str, max_tokens: int = 10000
):
    """Submit prompts without generations as OpenRouter batch (at reduced price).

    A state file is written before submitting and updated with the batch id after,
    so that a batch is never lost or submitted twice: if a batch is already pending,
    nothing is submitted. If a previous submission was interrupted before its batch
    id was saved, the batch is recovered from OpenRouter's batch list.
    Use collect_batch() to retrieve the results.

    Args:
        prompts: List of prompts to process
        model_name: Name of the model (format 'openrouter/provider/model')
        output_path: Directory to save results
        max_tokens: Maximum number of output tokens per generation
    """
    output_file, state_file = _get_batch_paths(output_path, model_name)
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        if state.get("batch_id"):
            logger.info(f"Batch already pending for {model_name}, not resubmitting.")
            return
        logger.warning(
            f"Previous batch submission for {model_name} was interrupted, "
            "checking whether batch was created."
        )
        if not _recover_batch(state, state_file, prompts) or state_file.exists():
            return

    processed_ids = _get_processed_ids(output_file)
    requests_list = []
    for prompt in prompts:
        prompt_id = hash_string(prompt)
        if prompt_id not in processed_ids:
            requests_list.append(
                {
                    "custom_id": prompt_id,
                    # Same parameters as feedback_forensics.models.main.get_model() in live mode
                    "body": {
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": TEMPERATURE,
                        # Report cost of each request (see get_usage())
                        "usage": {"include": True},
                    },
                }
            )
    if not requests_list:
        logger.info(f"All prompts already processed for {model_name}.")
        state_file.unlink(missing_ok=True)
        return

    state = {"model": model_name, "max_tokens": max_tokens}
    batch = openrouter.submit_batch(
        model_name,
        requests_list,
        entry=state,
        save=lambda: openrouter.write_json_atomic(state_file, state),
        discard=state_file.unlink,
    )
    logger.info(
        f"Submitted batch {batch['id']} with {len(requests_list)} prompts for {model_name}"
    )


def _parse_batch_result(
    result: dict, prompts_by_id: dict, model_name: str, batch: dict, state: dict
):
    """Convert successful batch result to generation record (None if failed)."""
    body = openrouter.get_result_body(result)
    if body is None:
        return None
    message = body["choices"][0]["message"]
    return {
        "prompt_id": result["custom_id"],
        "prompt": prompts_by_id[result["custom_id"]],
        "response": message["content"] or "",
        "full_response": body,
        "model": model_name,
        "timestamp": openrouter.now().isoformat(),
        "batch_id": batch["id"],
        "failure": get_failure(message.get("content"), body),
        "usage": get_usage(body),
        "generation_params": _generation_params("batch", state.get("max_tokens")),
        "provider": body.get("provider") or batch.get("provider"),
        "provider_model": body.get("model") or batch.get("model"),
    }


def collect_batch(
    prompts: List[str],
    model_name: str,
    output_path: str,
    deadline: float,
    poll_interval: int = 60,
) -> bool:
    """Wait for pending batch (until deadline) and save its results to generations file.

    Results of failed requests within the batch are not saved, so that they can be
    retried with run_model_on_prompts_async(). Successful results that cannot be
    parsed (or whose prompt is no longer in prompts) are saved as raw results to
    'batches/unparsed/', so that no paid generation is lost.

    Args:
        prompts: List of prompts (same as used for submit_batch)
        model_name: Name of the model
        output_path: Directory to save results
        deadline: Unix time after which to stop waiting
        poll_interval: Seconds between status checks

    Returns:
        True if no batch is pending anymore, False if batch is still pending
        (or needs to be resubmitted).
    """
    output_file, state_file = _get_batch_paths(output_path, model_name)
    if not state_file.exists():
        return True

    with open(state_file) as f:
        state = json.load(f)
    while True:
        if not state["batch_id"] and not _recover_batch(state, state_file, prompts):
            # Submission was interrupted, not known yet whether batch was created
            batch, status = None, "unknown"
        elif not state_file.exists():
            return False  # Batch was not created: to be submitted again
        else:
            batch, status = openrouter.poll_batch(
                state["batch_id"], state["submitted_at"]
            )
        if batch is not None:
            break
        if time.time() + poll_interval > deadline:
            logger.info(f"Batch for {model_name} still {status}.")
            return False
        time.sleep(poll_interval)
    batch_id = batch["id"]

    # Save results (also for expired/cancelled batches, which may have partial results)
    prompts_by_id = {hash_string(p): p for p in prompts}
    processed_ids = _get_processed_ids(output_file)
    records, unparsed, num_failed = [], [], 0
    for result in batch.get("results") or []:
        try:
            record = _parse_batch_result(
                result, prompts_by_id, model_name, batch, state
            )
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Could not parse result in batch {batch_id}: {e!r}")
            unparsed.append(result)
            continue
        if record is None:
            num_failed += 1
        elif record["prompt_id"] not in processed_ids:
            # (otherwise already saved in previous, interrupted run)
            records.append(record)
            processed_ids.add(record["prompt_id"])
            if record["failure"]:
                _log_failure(record)

    if unparsed:
        unparsed_file = state_file.parent / "unparsed" / f"{batch_id}.json"
        openrouter.write_json_atomic(
            unparsed_file, {"model": model_name, "results": unparsed}
        )
        logger.warning(f"Saved {len(unparsed)} unparsed results to {unparsed_file}")

    # Not creating an empty file (e.g. failed batch): the model must still count
    # as new, see get_batch_models()
    if records:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        _ensure_trailing_newline(output_file)
        with open(output_file, "a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in records))
            f.flush()
            os.fsync(f.fileno())

    # Only remove state once results are safely saved
    state_file.unlink()
    logger.info(
        f"Batch {batch_id} for {model_name} {status}: saved {len(records)} generations "
        f"({sum(bool(r['failure']) for r in records)} failed generations), "
        f"{num_failed} failed requests. Cost: {(batch.get('usage') or {}).get('cost')}"
    )
    if status in ("failed", "gone"):
        # E.g. batch variant listed by OpenRouter, but deprecated by provider, or
        # results deleted by OpenRouter (after 30 days)
        logger.error(
            f"Batch {batch_id} for {model_name} {status}: {batch.get('error')}. "
            "Falling back to live API for missing prompts."
        )
        return True
    # Expired/cancelled batches: missing prompts are generated in next run
    return status == "completed"


def load_generations(output_path: str, model_name: str):
    """Load generations from output path.

    Column 'failure' is (re)derived from full responses (see get_failure()),
    so that it is also set for generations saved before failures were recorded.
    """

    output_path = pathlib.Path(output_path)
    output_file = (
        output_path / "generations" / (model_name.replace("/", "_") + ".jsonl")
    )
    # Skip lines torn by interrupted writes instead of failing on whole file
    valid_lines = []
    with open(output_file, "r") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                json.loads(line)
                valid_lines.append(line.rstrip("\n") + "\n")
            except json.JSONDecodeError:
                if line.strip():
                    logger.warning(
                        f"Skipping malformed line {line_num} in {output_file}"
                    )
    generations = pd.read_json(io.StringIO("".join(valid_lines)), lines=True)
    generations["failure"] = [
        get_failure(r, fr)
        for r, fr in zip(generations["response"], generations["full_response"])
    ]
    num_failed = generations["failure"].notna().sum()
    if num_failed:
        logger.warning(
            f"{model_name}: {num_failed}/{len(generations)} generations failed "
            f"({generations['failure'].value_counts().to_dict()})."
        )
    return generations
