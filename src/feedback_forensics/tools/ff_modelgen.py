"""Tools to generate responses from models for comparison of model personalities."""

import pathlib
import json
from datetime import datetime, timezone
import io
from tqdm import tqdm
import inverse_cai.models
import inverse_cai.data.annotated_pairs_format
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

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
BATCH_SUFFIX = ":batch"  # OpenRouter suffix of batch model variants
NONBATCH_SUFFIX = ":nonbatch"  # Our suffix to disable batch API for model
BATCH_NOT_FOUND_GRACE = 600  # Seconds after submission to treat 404 as pending


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

    model = inverse_cai.models.get_model(model_name, max_tokens=16384)

    # Open file in append mode to continue where we left off
    with open(output_file, "a") as f:
        for prompt in tqdm(prompts, total=len(prompts)):
            prompt_id = inverse_cai.data.annotated_pairs_format.hash_string(prompt)

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
        skipped = len(
            processed_ids.intersection(
                [
                    inverse_cai.data.annotated_pairs_format.hash_string(p)
                    for p in prompts
                ]
            )
        )
        if skipped > 0:
            print(f"Skipping {skipped} already processed prompts")

    _ensure_trailing_newline(output_file)

    # Create file lock to prevent race conditions
    file_lock = asyncio.Lock()
    model = inverse_cai.models.get_model(model_name, max_tokens=max_tokens)

    # Create a thread pool for running synchronous model calls
    thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)

    async def process_prompt(prompt, index):
        nonlocal completed
        prompt_id = inverse_cai.data.annotated_pairs_format.hash_string(prompt)

        # Skip if already processed
        if prompt_id in processed_ids:
            return None

        try:
            # Create message
            msg = {"role": "user", "content": prompt}

            # Run model in thread pool to not block the event loop
            loop = asyncio.get_event_loop()
            generation = await loop.run_in_executor(
                thread_pool, functools.partial(model.invoke, [msg])
            )

            # Create a record with prompt, response, and metadata
            record = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "response": generation.content,
                "full_response": dict(generation),
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
            }

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
            print(f"Error processing prompt {prompt_id}: {e}")
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
    print(
        f"Total time: {elapsed:.2f}s, Average: {elapsed/max(1, completed):.2f}s per prompt"
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
        if not output_file.exists():
            candidates.append(model_name)

    if not candidates:
        return batch_models
    available = {m["id"] for m in _openrouter_request("GET", "/models")["data"]}
    for model_name in candidates:
        if model_name.split("/", 1)[1] + BATCH_SUFFIX in available:
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


def _write_json_atomic(path: pathlib.Path, data: dict):
    """Write JSON so that path never contains a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    """Parse ISO time (naive values are interpreted as local time)."""
    return datetime.fromisoformat(value).astimezone(timezone.utc)


class OpenRouterError(RuntimeError):
    def __init__(self, status_code: int, text: str):
        super().__init__(f"OpenRouter batch API error {status_code}: {text}")
        self.status_code = status_code


def _openrouter_request(method: str, path: str, **kwargs) -> dict:
    """Send request to OpenRouter API, retrying on rate limits and server errors.

    Non-GET requests are only retried on rate limits (429), since other errors
    (5xx, timeouts) are ambiguous: the request may have been processed, and
    retrying could e.g. submit (and pay for) a batch twice.
    """
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        **inverse_cai.models.OPENROUTER_HEADERS,
    }
    retry_all = method == "GET"
    for attempt in range(5):
        last_attempt = attempt == 4
        try:
            response = requests.request(
                method,
                OPENROUTER_API_URL + path,
                headers=headers,
                timeout=300,
                **kwargs,
            )
        except requests.RequestException as e:
            if not retry_all or last_attempt:
                raise
            logger.warning(f"OpenRouter request failed ({e}), retrying...")
            time.sleep(60)
            continue
        retryable = response.status_code == 429 or (
            retry_all and response.status_code >= 500
        )
        if not retryable or last_attempt:
            break
        logger.warning(f"OpenRouter returned {response.status_code}, retrying...")
        time.sleep(60)
    if not response.ok:
        raise OpenRouterError(response.status_code, response.text)
    return response.json()


def _find_submitted_batch(state: dict) -> str | None:
    """Find id of batch submitted for state whose submission response was lost.

    Returns:
        Batch id, or None if no such batch exists on OpenRouter.
    """
    created_after = int(_parse_time(state["submitted_at"]).timestamp()) - 300
    model = state["model"].split("/", 1)[1]
    matches, after = [], None
    while True:
        params = {"limit": 100, "created_after": created_after}
        if after:
            params["after"] = after
        page = _openrouter_request("GET", "/batches", params=params)
        for batch in page.get("data") or []:
            total = (batch.get("request_counts") or {}).get("total")
            if batch.get("model") == model and total in (None, state["num_requests"]):
                matches.append(batch["id"])
        if not page.get("has_more") or not page.get("last_id"):
            break
        after = page["last_id"]
    if len(matches) > 1:
        logger.warning(
            f"Found multiple batches {matches} for {state['model']}, using "
            f"{matches[-1]}. Others contain duplicate requests and can be ignored."
        )
    return matches[-1] if matches else None


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
        batch_id = _find_submitted_batch(state)
        if batch_id:
            state["batch_id"] = batch_id
            _write_json_atomic(state_file, state)
            logger.info(f"Recovered batch {batch_id} for {model_name}.")
            return
        logger.info(f"No batch found for {model_name}, submitting again.")

    processed_ids = _get_processed_ids(output_file)
    requests_list = []
    for prompt in prompts:
        prompt_id = inverse_cai.data.annotated_pairs_format.hash_string(prompt)
        if prompt_id not in processed_ids:
            requests_list.append(
                {
                    "custom_id": prompt_id,
                    # Same parameters as inverse_cai.models.get_model() in live mode
                    "body": {
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                    },
                }
            )
    if not requests_list:
        logger.info(f"All prompts already processed for {model_name}.")
        state_file.unlink(missing_ok=True)
        return

    assert model_name.startswith(
        "openrouter/"
    ), f"Batch mode only supported for OpenRouter models, got {model_name}"

    # Save intent first: if response gets lost (crash, timeout, 5xx), next run
    # finds the batch via OpenRouter's batch list instead of submitting again.
    state = {
        "batch_id": None,
        "model": model_name,
        "num_requests": len(requests_list),
        "submitted_at": _now().isoformat(),
    }
    _write_json_atomic(state_file, state)

    # Note: OpenRouter requires 'requests' to be the last key
    try:
        batch = _openrouter_request(
            "POST",
            "/batches",
            json={
                "endpoint": "/v1/chat/completions",
                "model": model_name.split("/", 1)[1],
                "requests": requests_list,
            },
        )
    except OpenRouterError as e:
        if e.status_code < 500:
            # Definitely rejected (not created), safe to resubmit later
            state_file.unlink()
        raise

    state["batch_id"] = batch["id"]
    _write_json_atomic(state_file, state)
    logger.info(
        f"Submitted batch {batch['id']} with {len(requests_list)} prompts for {model_name}"
    )


def _parse_batch_result(result: dict, prompts_by_id: dict, model_name, batch_id):
    """Convert successful batch result to generation record (None if failed)."""
    response = result.get("response")
    if result.get("error") or not response or response.get("status_code") != 200:
        return None
    body = response["body"]
    return {
        "prompt_id": result["custom_id"],
        "prompt": prompts_by_id[result["custom_id"]],
        "response": body["choices"][0]["message"]["content"] or "",
        "full_response": body,
        "model": model_name,
        "timestamp": _now().isoformat(),
        "batch_id": batch_id,
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
    batch_id = state["batch_id"]
    assert batch_id, f"Batch id unknown for {model_name}, run submit_batch() first."
    submitted_at = _parse_time(state["submitted_at"])

    while True:
        try:
            batch = _openrouter_request("GET", f"/batches/{batch_id}")
            status = batch["status"]
        except OpenRouterError as e:
            age = (_now() - submitted_at).total_seconds()
            if e.status_code == 404 and age <= BATCH_NOT_FOUND_GRACE:
                # Guard against newly submitted batches briefly not being found
                batch, status = {}, "not_found_yet"
            elif e.status_code == 410:
                # Results deleted by OpenRouter (after 30 days)
                logger.error(
                    f"Results of batch {batch_id} for {model_name} expired on "
                    "OpenRouter. Missing prompts will be generated via live API."
                )
                state_file.unlink()
                return True
            else:
                raise
        if status in ("failed", "expired", "cancelled"):
            break
        if status == "completed" and batch.get("results") is not None:
            break
        if time.time() + poll_interval > deadline:
            logger.info(f"Batch {batch_id} for {model_name} still {status}.")
            return False
        time.sleep(poll_interval)

    # Save results (also for expired/cancelled batches, which may have partial results)
    prompts_by_id = {
        inverse_cai.data.annotated_pairs_format.hash_string(p): p for p in prompts
    }
    processed_ids = _get_processed_ids(output_file)
    records, unparsed, num_failed = [], [], 0
    for result in batch.get("results") or []:
        try:
            record = _parse_batch_result(result, prompts_by_id, model_name, batch_id)
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

    if unparsed:
        unparsed_file = state_file.parent / "unparsed" / f"{batch_id}.json"
        _write_json_atomic(unparsed_file, {"model": model_name, "results": unparsed})
        logger.warning(f"Saved {len(unparsed)} unparsed results to {unparsed_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    _ensure_trailing_newline(output_file)
    with open(output_file, "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in records))
        f.flush()
        os.fsync(f.fileno())

    # Only remove state once results are safely saved
    state_file.unlink()
    logger.info(
        f"Batch {batch_id} for {model_name} {status}: saved {len(records)} generations, "
        f"{num_failed} failed requests. Cost: {(batch.get('usage') or {}).get('cost')}"
    )
    if status == "failed":
        raise RuntimeError(
            f"Batch {batch_id} for {model_name} failed: {batch.get('error')}"
        )
    # Expired/cancelled batches: missing prompts are generated in next run
    return status == "completed"

def load_generations(output_path: str, model_name: str):
    """Load generations from output path."""

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
                    logger.warning(f"Skipping malformed line {line_num} in {output_file}")
    return pd.read_json(io.StringIO("".join(valid_lines)), lines=True)
