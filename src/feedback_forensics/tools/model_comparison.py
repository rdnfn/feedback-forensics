"""Tools to generate responses from models for comparison of model personalities."""

import pathlib
import json
from tqdm import tqdm
import inverse_cai.models
import inverse_cai.data.annotated_pairs_format


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
