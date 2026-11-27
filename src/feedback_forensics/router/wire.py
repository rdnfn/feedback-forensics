"""The request and response format (OpenRouter chat completions), shared by the
live and the batch lane."""

from feedback_forensics.router.datatypes import TTL_1H, Call, Usage


def split_pieces(call: Call, breakpoints=()) -> list[list[tuple[str, bool]]]:
    """The prompt cut at the breakpoints: for each message, its pieces as
    (text, whether a breakpoint follows it).

    Breakpoints are character offsets into `Call.text`. A message is only cut
    where a breakpoint lies within it.
    """
    breakpoints = sorted(set(breakpoints))
    messages, start = [], 0
    for message in call.messages:
        end = start + len(message.text)
        cuts = [b - start for b in breakpoints if start < b < end]
        bounds = [0] + cuts + [len(message.text)]
        pieces = [
            (message.text[a:b], i < len(cuts))
            for i, (a, b) in enumerate(zip(bounds, bounds[1:]))
        ]
        if end in breakpoints and end > start:
            pieces[-1] = (pieces[-1][0], True)
        messages.append(pieces)
        start = end
    return messages


def build_body(call: Call, breakpoints=(), cache_ttl: str | None = None) -> dict:
    """Body of the chat completions request of a call (without the model).

    A breakpoint is expressed by sending the content of its message as a list
    of text blocks, with `cache_control` on the block that ends the cached
    prefix. Messages without breakpoint are sent as plain strings: splitting a
    prompt into blocks can change how it is tokenised.
    """
    cache_control = {"type": "ephemeral"}
    if cache_ttl == TTL_1H:
        cache_control["ttl"] = "1h"
    messages = []
    for message, pieces in zip(call.messages, split_pieces(call, breakpoints)):
        if not any(marked for _, marked in pieces):
            messages.append({"role": message.role, "content": message.text})
            continue
        blocks = []
        for text, marked in pieces:
            block = {"type": "text", "text": text}
            if marked:
                block["cache_control"] = dict(cache_control)
            blocks.append(block)
        messages.append({"role": message.role, "content": blocks})

    params = call.params
    body: dict = {"messages": messages}
    if params.max_tokens is not None:
        body["max_tokens"] = params.max_tokens
    if params.temperature is not None:
        body["temperature"] = params.temperature
    # Report token usage (incl. cached tokens) and cost of the request
    body["usage"] = {"include": True}
    if params.reasoning_effort is not None:
        body["reasoning"] = {"effort": params.reasoning_effort}
    return body


def parse_usage(body: dict) -> Usage:
    """Token usage and cost reported in a chat completion."""
    usage = body.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cost = usage.get("cost")
    return Usage(
        input_tokens=usage.get("prompt_tokens"),
        cached_tokens=prompt_details.get("cached_tokens"),
        cache_write_tokens=prompt_details.get("cache_write_tokens"),
        output_tokens=usage.get("completion_tokens"),
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        cost_usd=float(cost) if cost is not None else None,
    )


def parse_completion(body: dict) -> tuple[str, str | None, Usage, str | None]:
    """(text, finish reason, usage, provider) of a chat completion.

    Raises:
        KeyError, IndexError, TypeError: The completion cannot be read.
    """
    choice = body["choices"][0]
    text = choice["message"]["content"]
    if text is None:
        text = ""  # e.g. all output tokens used for reasoning
    if not isinstance(text, str):
        raise TypeError(f"content is not a string: {text!r}")
    return text, choice.get("finish_reason"), parse_usage(body), body.get("provider")
