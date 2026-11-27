"""Shared prompt prefixes of a set of calls (pure: no network, no disk).

The prompts are sorted as text, which puts prompts with a common start next to
each other, and the longest common prefix of each pair of neighbours gives the
tree: callers do not mark shared parts.
"""

import math
from bisect import bisect_left
from typing import Iterable

from feedback_forensics.router.datatypes import (
    Call,
    ModelProfile,
    PrefixNode,
    PrefixTree,
)

# Prompts are compared as one text in which every message starts with a header
# (`_SEP` role `_END`), and which ends with `_SEP`: two prompts with a common
# prefix then also agree on the roles and ends of the messages within it.
_SEP, _END = "\x1e", "\x1f"


def estimate_tokens(chars: int, profile: ModelProfile) -> int:
    return math.ceil(chars / profile.chars_per_token)


class _Prompt:
    """A call's prompt as one text, and where its messages end in it."""

    __slots__ = ("call", "text", "bounds", "head_chars")

    def __init__(self, call: Call):
        self.call = call
        parts, bounds, head_chars, pos, heads = [], [], [], 0, 0
        for message in call.messages:
            bounds.append(pos)
            head_chars.append(heads)
            head = f"{_SEP}{message.role}{_END}"
            parts += [head, message.text]
            pos += len(head) + len(message.text)
            heads += len(head)
        bounds.append(pos)
        head_chars.append(heads)
        parts.append(_SEP)
        self.text = "".join(parts)
        # Positions of the message headers, and of the end of the last message
        self.bounds = bounds
        # Header characters before each of these positions
        self.head_chars = head_chars


def common_prefix_length(a: str, b: str) -> int:
    """Length of the longest common prefix (comparing slices, not characters)."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    low, high = 0, n  # a[:low] == b[:low], a[:high] != b[:high]
    while high - low > 1:
        mid = (low + high) // 2
        if a[low:mid] == b[low:mid]:
            low = mid
        else:
            high = mid
    return low


def _cut(a: _Prompt, b: _Prompt, common: int) -> int:
    """Longest prefix (characters of `Call.text`) two neighbouring prompts with
    `common` characters in common may share: it ends at a line break or at the
    end of a message. 0 if there is none."""
    after_line_break = a.text.rfind("\n", 0, common) + 1
    i = bisect_left(a.bounds, common) - 1  # last message end before `common`
    message_end = a.bounds[i] if i > 0 else 0
    cut = max(after_line_break, message_end)
    if cut == 0:
        return 0
    # Both prompts must have the same messages up to the cut (not the case if
    # the text of one contains what looks like the header of the other)
    n = bisect_left(a.bounds, cut)
    if a.bounds[:n] != b.bounds[:n] or bisect_left(b.bounds, cut) != n:
        return 0
    if cut == message_end and cut not in b.bounds:
        return 0
    return cut - a.head_chars[min(n, len(a.head_chars) - 1)]


def build_prefix_tree(calls: Iterable[Call], profile: ModelProfile) -> PrefixTree:
    """The tree of prefixes shared by the calls (of one model).

    A prefix is kept if at least two calls share it and it is long enough to be
    cached for the model (`profile.min_cacheable_tokens`). Prefixes end at line
    breaks or at the end of a message.
    """
    unique = {call.key: call for call in calls}
    models = {call.model for call in unique.values()}
    if len(models) > 1:
        raise ValueError(f"A prefix tree is built per model, got {sorted(models)}.")
    prompts = sorted(
        (_Prompt(call) for call in unique.values()),
        key=lambda p: (p.text, p.call.key),
    )
    keys = tuple(p.call.key for p in prompts)
    cuts = [
        _cut(a, b, common_prefix_length(a.text, b.text))
        for a, b in zip(prompts, prompts[1:])
    ]

    # Stack of open nodes: [end, index of first call, finished children]
    stack: list[list] = [[0, 0, []]]

    def close(node: list, last: int) -> list[PrefixNode]:
        end, first, children = node
        tokens = estimate_tokens(end, profile)
        if end > 0 and tokens < profile.min_cacheable_tokens:
            return children  # too short to cache: its children move up
        return [PrefixNode(end, tokens, keys[first : last + 1], tuple(children))]

    for i, cut in enumerate(cuts + [0]):
        first, closed = i, []
        while stack[-1][0] > cut:
            node = stack.pop()
            node[2].extend(closed)
            closed = close(node, i)
            first = node[1]
        if stack[-1][0] < cut:
            stack.append([cut, first, closed])
        else:
            stack[-1][2].extend(closed)
    (root,) = close(stack[0], len(prompts) - 1)
    model = next(iter(models), "")
    return PrefixTree(model, root, tuple(p.call for p in prompts))


def node_paths(tree: PrefixTree) -> dict[str, tuple[PrefixNode, ...]]:
    """For each call (key), the shared prefixes it starts with, shortest first."""
    paths = {key: () for key in tree.root.keys}
    stack = [(child, (child,)) for child in tree.root.children]
    while stack:
        node, path = stack.pop()
        for key in node.keys:
            paths[key] = path  # (replaced by longer paths of children below)
        stack.extend((child, path + (child,)) for child in node.children)
    return paths


def node_id(tree: PrefixTree, node: PrefixNode) -> tuple:
    """Identifies the prefix of a node (the cache entry written for it)."""
    return (tree.model, node.end, node.keys[0])
