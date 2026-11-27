"""Tests of prefix detection and breakpoint choice (pure)."""

import random
from dataclasses import replace

from feedback_forensics.router.datatypes import Call, Message
from feedback_forensics.router.prefix import (
    build_prefix_tree,
    common_prefix_length,
    node_paths,
)
from feedback_forensics.router.strategies import choose_breakpoints
from feedback_forensics.router.testutils import (
    FIXED,
    FIXED_END,
    MIDDLE_END,
    MODEL,
    PROFILE,
    SYSTEM,
    lines,
    make_call,
    make_calls,
)


def shape(node) -> tuple:
    """(end, number of calls, children) of a node."""
    return (node.end, len(node.keys), [shape(child) for child in node.children])


def test_common_prefix_length():
    assert common_prefix_length("", "abc") == 0
    assert common_prefix_length("abc", "abc") == 3
    assert common_prefix_length("abc", "abd") == 2
    assert common_prefix_length("xbc", "abc") == 0
    a = "a" * 10_000 + "b" + "c" * 100
    assert common_prefix_length(a, "a" * 10_000 + "x") == 10_000


def test_annotation_shaped_calls():
    """A fixed start shared by all, a middle part shared by groups, a unique end."""
    calls = make_calls(groups=3, per_group=4)
    random.Random(0).shuffle(calls)
    tree = build_prefix_tree(calls, PROFILE)
    assert shape(tree.root) == (
        0,
        12,
        [
            (
                FIXED_END,
                12,
                [(MIDDLE_END, 4, []), (MIDDLE_END, 4, []), (MIDDLE_END, 4, [])],
            )
        ],
    )
    fixed = tree.root.children[0]
    assert fixed.tokens == FIXED_END  # one character per token
    assert tree.calls[0].text[: fixed.end] == SYSTEM + FIXED
    # Tree order: the calls of a group are next to each other
    groups = [call.tag["group"] for call in tree.calls]
    assert groups == sorted(groups)
    assert tree.root.keys == tuple(call.key for call in tree.calls)
    for group_node in fixed.children:
        tags = [c.tag["group"] for c in tree.calls if c.key in group_node.keys]
        assert len(set(tags)) == 1
    paths = node_paths(tree)
    assert all([n.end for n in paths[c.key]] == [FIXED_END, MIDDLE_END] for c in calls)


def test_prefixes_end_at_line_breaks():
    # The unique ends of a group start alike ("response 0-"), and their common
    # text ends within a line: the prefix ends at the line break before
    calls = make_calls(groups=1, per_group=3)
    text = calls[0].text
    assert text[MIDDLE_END:].startswith("response 0-")
    tree = build_prefix_tree(calls, PROFILE)
    (node,) = tree.root.children
    assert node.end == MIDDLE_END and text[node.end - 1] == "\n"

    # Without any line break in the common text, there is no prefix
    one_line = [Call(MODEL, (Message("user", "x" * 500 + str(i)),)) for i in range(3)]
    assert build_prefix_tree(one_line, PROFILE).root.children == ()


def test_prefixes_may_end_where_a_message_ends():
    calls = [
        Call(MODEL, (Message("system", "s" * 300), Message("user", f"question {i}")))
        for i in range(3)
    ]
    (node,) = build_prefix_tree(calls, PROFILE).root.children
    assert node.end == 300

    # The same text in messages that end elsewhere is not a common prefix
    other = Call(MODEL, (Message("system", "s" * 299), Message("user", "s question")))
    tree = build_prefix_tree([calls[0], other], PROFILE)
    assert tree.root.children == ()
    # ... nor is the same text with another role
    other = Call(MODEL, (Message("user", "s" * 300), Message("user", "question 0")))
    assert build_prefix_tree([calls[0], other], PROFILE).root.children == ()


def test_calls_differing_only_in_params_share_the_whole_prompt():
    call = make_call(0, 0)
    other = Call(call.model, call.messages, replace(call.params, max_tokens=7))
    (node,) = build_prefix_tree([call, other], PROFILE).root.children
    assert node.end == len(call.text)


def test_minimum_length():
    """Prefixes too short to be cached for the model are dropped."""
    calls = make_calls(groups=2, per_group=3)
    # Long enough for the middle parts only
    profile = replace(PROFILE, min_cacheable_tokens=FIXED_END + 1)
    tree = build_prefix_tree(calls, profile)
    assert shape(tree.root) == (0, 6, [(MIDDLE_END, 3, []), (MIDDLE_END, 3, [])])
    # Too long for any
    profile = replace(PROFILE, min_cacheable_tokens=MIDDLE_END + 1)
    assert build_prefix_tree(calls, profile).root.children == ()
    # The length is estimated from the characters per token of the profile
    profile = replace(PROFILE, chars_per_token=4.0, min_cacheable_tokens=200)
    tree = build_prefix_tree(calls, profile)
    assert shape(tree.root) == (0, 6, [(MIDDLE_END, 3, []), (MIDDLE_END, 3, [])])
    assert tree.root.children[0].tokens == MIDDLE_END / 4


def test_two_call_minimum():
    """A prefix is only kept if at least two calls share it."""
    calls = make_calls(groups=2, per_group=2) + [make_call(7, 0)]
    tree = build_prefix_tree(calls, PROFILE)
    (fixed,) = tree.root.children
    # The call that is alone in its group only shares the fixed start
    assert shape(fixed) == (FIXED_END, 5, [(MIDDLE_END, 2, []), (MIDDLE_END, 2, [])])
    assert [n.end for n in node_paths(tree)[calls[-1].key]] == [FIXED_END]
    # A single call shares nothing
    assert build_prefix_tree(calls[:1], PROFILE).root.children == ()
    assert build_prefix_tree([], PROFILE).root.keys == ()


def test_text_like_message_headers_is_harmless():
    a = Call(MODEL, (Message("system", "s" * 200 + "\n"), Message("user", "one")))
    b = Call(MODEL, (Message("system", "s" * 200 + "\n\x1euser\x1fone two"),))
    tree = build_prefix_tree([a, b], PROFILE)
    assert all(node.end <= 201 for node in tree.root.children)


def nested_calls(levels: int) -> list[Call]:
    """Calls with `levels` nested shared prefixes: at each level, two calls
    leave with their own end."""
    calls, prefix = [], ""
    for level in range(levels):
        prefix += lines(f"level {level}", 10)
        calls += [
            Call(MODEL, (Message("user", prefix + f"end {level}-{i}"),))
            for i in range(2)
        ]
    return calls


def test_breakpoint_limit():
    """At most `max_breakpoints` per call: those with the largest saving."""
    calls = nested_calls(levels=6)
    tree = build_prefix_tree(calls, PROFILE)
    deepest = calls[-1].key
    path = node_paths(tree)[deepest]
    assert [node.end for node in path] == [300 * n for n in range(1, 7)]

    chosen = choose_breakpoints(tree, PROFILE)
    assert all(len(nodes) <= PROFILE.max_breakpoints == 4 for nodes in chosen.values())
    # Every level adds the same number of tokens, and is shared by more calls
    # the shorter it is: the shortest four save most
    assert [node.end for node in chosen[deepest]] == [300, 600, 900, 1200]
    assert [node.end for node in chosen[calls[0].key]] == [300]

    two = choose_breakpoints(tree, replace(PROFILE, max_breakpoints=2))
    assert [node.end for node in two[deepest]] == [300, 600]
    none = choose_breakpoints(tree, replace(PROFILE, max_breakpoints=0))
    assert set(none.values()) == {()}


def test_breakpoints_only_where_they_save():
    # Shared by two calls: read once at 10% of the price (saving 90%), written
    # once for 25% more: pays off. Not with a write surcharge of 100%.
    calls = make_calls(groups=1, per_group=2)
    tree = build_prefix_tree(calls, PROFILE)
    assert [n.end for n in choose_breakpoints(tree, PROFILE)[calls[0].key]] == [
        MIDDLE_END
    ]
    costly = replace(PROFILE, cache_write_multiplier_5m=2.0)
    assert choose_breakpoints(tree, costly)[calls[0].key] == ()
