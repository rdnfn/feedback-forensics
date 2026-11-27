"""Layout check: text that repeats across calls where no cache can reach it.

A cache only covers the start of a prompt. Text that many calls share, but that
comes after text unique to each call, is paid in full every time. The check
reports such text; it never changes a prompt. Pure (no network, no disk).
"""

import re
import statistics
from collections import Counter
from typing import Iterable, Mapping

from feedback_forensics.router.prefix import (
    build_prefix_tree,
    estimate_tokens,
    node_paths,
)
from feedback_forensics.router.profiles import get_profile
from feedback_forensics.router.datatypes import (
    Call,
    LayoutFinding,
    LayoutReport,
    ModelProfile,
)

_BLANK_LINES = re.compile(r"\n\s*\n")


def _paragraphs(text: str, start: int) -> list[tuple[int, str]]:
    """(offset, text) of the paragraphs (separated by blank lines) of
    `text[start:]`."""
    paragraphs, position = [], start
    ends = [(m.start(), m.end()) for m in _BLANK_LINES.finditer(text, start)]
    for end, next_start in ends + [(len(text), len(text))]:
        paragraph = text[position:end].rstrip()
        if paragraph.strip():
            paragraphs.append((position, paragraph))
        position = next_start
    return paragraphs


def check_layout(
    calls: Iterable[Call],
    profiles: Mapping[str, ModelProfile],
    min_calls: int = 2,
    max_findings: int = 20,
) -> LayoutReport:
    """Report text that repeats across calls behind text unique to each call.

    Method: take each prompt after its shared start (the longest prefix in the
    prefix tree), split it into paragraphs (at blank lines), and count in how
    many calls of the model each paragraph appears. A finding is a run of
    consecutive paragraphs that each appear in at least `min_calls` calls,
    behind a paragraph that appears in only one. Its saving is that of caching
    it (see `choose_breakpoints`): reads by all calls but one, minus one write.
    Findings are sorted by saving; those without one are left out (so nothing
    is reported for models that do not cache).
    """
    by_model: dict[str, dict[str, Call]] = {}
    for call in calls:
        by_model.setdefault(call.model, {}).setdefault(call.key, call)
    findings = []
    for model in sorted(by_model):
        profile = get_profile(profiles, model)
        if profile.max_breakpoints < 1:
            continue
        tree = build_prefix_tree(by_model[model].values(), profile)
        paths = node_paths(tree)
        tails = []  # per call: (offset, paragraph) after the shared start
        counts: Counter = Counter()
        for call in tree.calls:
            path = paths[call.key]
            start = path[-1].end if path else 0
            paragraphs = _paragraphs(call.text, start)
            tails.append(paragraphs)
            counts.update({text for _, text in paragraphs})

        runs: dict[tuple, list[int]] = {}  # paragraphs -> offsets (one per call)
        for paragraphs in tails:
            seen, run, behind_unique = set(), [], False

            def flush():
                texts = tuple(text for _, text in run)
                if run and texts not in seen:
                    seen.add(texts)
                    runs.setdefault(texts, []).append(run[0][0])
                run.clear()

            for offset, text in paragraphs:
                if counts[text] >= min_calls:
                    if behind_unique:
                        run.append((offset, text))
                    continue
                flush()
                behind_unique = behind_unique or counts[text] == 1
            flush()

        for texts, offsets in runs.items():
            text = "\n\n".join(texts)
            tokens = estimate_tokens(len(text), profile)
            saving = (
                tokens
                * profile.input_price
                * (
                    (len(offsets) - 1) * (1 - profile.cache_read_multiplier)
                    - (profile.cache_write_multiplier_5m - 1)
                )
                / 1e6
            )
            if saving > 0:
                findings.append(
                    LayoutFinding(
                        model=model,
                        calls=len(offsets),
                        offset=int(statistics.median(offsets)),
                        tokens=tokens,
                        saving_usd=saving,
                        preview=text[:80],
                    )
                )
    findings.sort(key=lambda f: (-f.saving_usd, f.model, f.offset, f.preview))
    return LayoutReport(tuple(findings[:max_findings]))
