"""Prompt for principle-following annotation and prompt parsing utilities.

Ported from inverse-cai v0.4.8 (`experiment/config/prompts.py` and
`algorithm/utils.py`). The voting prompt must stay identical to keep results
comparable with annotations created with ICAI.
"""

import re

from loguru import logger

# NOTE: keep byte-identical to ICAI (incl. the trailing space after the "Put \"A\"" line).
VOTING_PROMPT = """<|im_start|>system
Your job is to check which sample is should be selected according to the given rules. You're an expert at this.
<|im_end|>
<|im_start|>user
Sample A:
{sample_a}

Sample B:
{sample_b}

Given the samples data above, check for each rule below which sample should be selected:
{summaries}

Answer in json format, e.g. {{0: "A", 1: "B", 2: "None", 3: "Both",...}}.
Put "A" if A is selected according to that rule. 
Put "B" if B is selected according to that rule.
Put "Both" if both A and B should be selected, and the rule is categorical so it is impossible to select only one.
Put "None" if a rule is not applicable to the two samples.
Otherwise, no ties are allowed, only one of "A", "B", "Both" or "None".
Vote for all rules, even if you are unsure.
DO NOT respond with any text apart from the json format above!
DO NOT add markdown formatting around JSON.
ONLY REPLY IN JSON FORMAT
<|im_end|>"""


def parse_prompt(prompt_str: str, prompt_kwargs: dict) -> list[dict]:
    """Parse a ChatML prompt string into a list of messages, filling in kwargs."""

    for key in prompt_kwargs:
        if f"{{{key}}}" not in prompt_str:
            logger.error(f"Key '{key}' not found in prompt_str. Check your prompts.")

    for m in re.finditer(r"{([^{} ]+)}", prompt_str):
        key = m.group(1)
        if key not in prompt_kwargs:
            logger.error(f"Key '{key}' is not a valid key. Check your prompts.")
            prompt_str = prompt_str.replace(f"{{{key}}}", "")

    messages = prompt_to_chatml(prompt_str)

    # Fill in values AFTER converting to chatml, so that values containing
    # chatml syntax are not parsed as chatml.
    for message in messages:
        if "content" in message:
            content = message["content"]
            filtered_kwargs = {
                k: v for k, v in prompt_kwargs.items() if f"{{{k}}}" in content
            }
            message["content"] = content.format(**filtered_kwargs)

    return messages


# `string_to_dict` and `prompt_to_chatml` are adapted from alpaca_eval.utils
# Licensed under Apache License 2.0 (https://github.com/tatsu-lab/alpaca_eval/blob/cd543a149df89434d8a54582c0151c0b945c3d20/LICENSE)
# Copyright 2023 Yann Dubois and Xuechen Li and Rohan Taori and Tianyi Zhang and Ishaan Gulrajani


def _string_to_dict(to_convert: str) -> dict:
    """Converts a string with equal signs to dictionary, e.g. "a=1 b=2"."""
    return {
        s.split("=", 1)[0]: s.split("=", 1)[1]
        for s in to_convert.split(" ")
        if len(s) > 0
    }


def prompt_to_chatml(
    prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>"
) -> list[dict]:
    """Convert a ChatML text prompt to a list of message dicts."""
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    messages = []
    for p in prompt.split(start_token)[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            other_params = _string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        messages.append(dict(content=content, role=role, **other_params))

    return messages
