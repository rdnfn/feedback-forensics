"""Tools to generate responses from models for comparison of model personalities."""

import pathlib
import json
from datetime import datetime
from tqdm import tqdm
import inverse_cai.models
import inverse_cai.data.annotated_pairs_format
import asyncio
import aiofiles
from typing import List
import time
import functools
import concurrent.futures
import pandas as pd


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


def load_generations(output_path: str, model_name: str):
    """Load generations from output path."""

    output_path = pathlib.Path(output_path)
    output_file = (
        output_path / "generations" / (model_name.replace("/", "_") + ".jsonl")
    )
    return pd.read_json(output_file, lines=True)
