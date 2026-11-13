"""Rigorous equivalence check: native FF annotation vs. the former ICAI pipeline.

Both pipelines annotate the same datasets against the same deterministic mock LLM
API (see `mock_api.py`). The mock's answers depend only on content (never on which
response is shown first), so the outputs MUST be identical: any difference is a
bug in the port. Additionally, the API requests must be identical up to the order
of the two responses in each prompt (system prompt, principles and their order,
model parameters). The A/B order itself is not compared: FF assigns its random A/B
orders to the comparisons in row order, ICAI in the (non-reproducible) order in
which its async tasks start, so the A/B order of a comparison matches ICAI's only
by chance, about half of the time (see `canonical_requests`).

ICAI is run exactly as `ff-annotate` invoked it before annotation became native.
Requires the `inverse-cai` package (`icai-exp` command), skipped otherwise.
"""

import ast
import json
import os
import shutil
import subprocess
from collections import Counter

import pandas as pd
import pytest

from feedback_forensics.annotation import annotate_dataset, get_default_principles
from feedback_forensics.annotation.mock_api import VOTING_PROMPT_REGEX
from feedback_forensics.annotation.testutils import MOCK_MODEL, make_pairs_df
from feedback_forensics.data.ap_format import (
    create_annotated_pairs,
    save_annotated_pairs_to_file,
)

pytestmark = pytest.mark.skipif(
    shutil.which("icai-exp") is None,
    reason="inverse-cai (icai-exp) not installed",
)


def run_icai_annotation(datapath, output_dir, principles, model, base_url, cwd):
    """Run ICAI with the exact configuration formerly used by `ff-annotate`."""
    cmd = [
        "icai-exp",
        f"data_path={datapath}",
        "annotator.skip=true",
        "s0_skip_principle_generation=true",
        "async_task_num=400",
        "s3_num_seeds_to_reannotate_with=1",
        f"hydra.run.dir={output_dir}",
        f"alg_model={model}",
        f"s0_added_principles_to_test={json.dumps(principles)}",
    ]
    env = {
        **os.environ,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_BASE": base_url,
        "OPENAI_API_KEY": "mock-key",
    }
    process = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=900
    )
    assert process.returncode == 0, process.stderr[-5000:]


def normalise_ap(ap: dict) -> dict:
    """Remove metadata that legitimately differs (timestamps, names, descriptions)."""
    ap = json.loads(json.dumps(ap))
    for key in (
        "created_at",
        "dataset_name",
        "description",
        "annotation",
        "annotation_usage",
    ):
        ap["metadata"].pop(key, None)
    return ap


def load_raw_votes(path) -> dict:
    df = pd.read_csv(path)
    vote_col = [c for c in df.columns if c != "index"][0]  # ICAI: "0", FF: "votes"
    return {row["index"]: ast.literal_eval(row[vote_col]) for _, row in df.iterrows()}


def canonical_requests(requests: list[dict]) -> list[str]:
    """Requests with the order of the two samples normalised.

    ICAI's A/B order is not reproducible: tasks are started via
    `asyncio.as_completed` (used by `tqdm.gather`), which iterates over a *set*
    of coroutines, so random A/B flips get assigned to comparisons in set
    iteration order (which depends on the coroutines' memory addresses), and
    the backoff jitter of retries also draws random numbers. Hence requests can
    only be compared modulo A/B order; all else (system prompt, principles and
    their order, model parameters) must be identical. Vote equality is
    unaffected since the mock is order-invariant.
    """
    canonical = []
    for request in requests:
        request = json.loads(json.dumps(request))
        user_message = request["messages"][1]
        match = VOTING_PROMPT_REGEX.search(user_message["content"])
        assert match is not None
        sample_1, sample_2 = sorted([match["a"], match["b"]])
        user_message["content"] = (
            user_message["content"][: match.start()]
            + f"SAMPLES:{sample_1}\x00{sample_2}\nPRINCIPLES:{match['principles']}"
            + user_message["content"][match.end() :]
        )
        canonical.append(json.dumps(request, sort_keys=True))
    return sorted(canonical)


EXTRA_PRINCIPLES = [
    "Select the response that is more confident",
    "Select the response that sounds more like a pirate",
]


@pytest.mark.parametrize(
    "case",
    [
        # >40 principles: two passes over the data (two prompts per comparison);
        # prompt column (merged into texts), extra annotator + metadata columns
        dict(name="prompts_two_passes", extra_principles=True, fail_first=False),
        # no preferred_text (pseudo-label added), transient HTTP 500 errors
        dict(name="no_pref_with_retries", extra_principles=False, fail_first=True),
        # AnnotatedPairs JSON input
        dict(name="ap_json_input", extra_principles=False, fail_first=False),
    ],
    ids=lambda c: c["name"],
)
def test_native_annotation_identical_to_icai(tmp_path, mock_llm_server, case):
    server = mock_llm_server(fail_first_attempts=case["fail_first"])

    if case["name"] == "prompts_two_passes":
        df = make_pairs_df(80, with_prompt=True, with_extra_columns=True, seed="p")
        datapath = tmp_path / "data.csv"
        df.to_csv(datapath, index=False)
    elif case["name"] == "no_pref_with_retries":
        df = make_pairs_df(80, with_preferred=False, seed="n")
        datapath = tmp_path / "data.csv"
        df.to_csv(datapath, index=False)
    else:
        df = make_pairs_df(60, with_extra_columns=True, seed="j")
        df = df.drop(columns=["human_label"])
        ap = create_annotated_pairs(df, dataset_name="input")
        datapath = tmp_path / "data.json"
        save_annotated_pairs_to_file(ap, datapath)

    principles = get_default_principles()
    if case["extra_principles"]:
        principles += EXTRA_PRINCIPLES

    # ICAI
    icai_dir = tmp_path / "icai"
    run_icai_annotation(
        datapath, icai_dir, principles, MOCK_MODEL, server.base_url, tmp_path
    )
    icai_requests = server.requests
    icai_failed_attempts = server.num_failed_attempts
    server.reset()

    # Native FF
    ff_dir = tmp_path / "ff"
    ff_ap_path = annotate_dataset(
        datapath=datapath, principles=principles, model=MOCK_MODEL, output_dir=ff_dir
    )
    ff_requests = server.requests

    # 1. Identical annotated datasets
    icai_ap = json.load(open(icai_dir / "results" / "070_annotations_train_ap.json"))
    ff_ap = json.load(open(ff_ap_path))
    assert normalise_ap(ff_ap) == normalise_ap(icai_ap)

    # 2. Identical raw votes and annotated input data
    assert load_raw_votes(ff_dir / "results" / "040_votes_per_comparison.csv") == (
        load_raw_votes(icai_dir / "results" / "040_votes_per_comparison.csv")
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(ff_dir / "results" / "000_train_data.csv"),
        pd.read_csv(icai_dir / "results" / "000_train_data.csv"),
    )
    assert json.load(
        open(ff_dir / "results" / "030_distilled_principles_per_cluster.json")
    ) == json.load(
        open(icai_dir / "results" / "030_distilled_principles_per_cluster.json")
    )

    # 3. Identical API requests (prompts incl. principle order, params),
    #    modulo A/B order (see `canonical_requests`)
    num_passes = 2 if case["extra_principles"] else 1
    assert len(ff_requests) == len(df) * num_passes
    assert canonical_requests(ff_requests) == canonical_requests(icai_requests)

    # Sanity: the test data exercises all vote outcomes and error paths
    outcomes = Counter(
        annotation.get("no_pref_reason", "pref")
        for comparison in ff_ap["comparisons"]
        for annotation in comparison["annotations"].values()
    )
    for outcome in ("pref", "not_applicable", "both", "neither", "invalid"):
        assert outcomes[outcome] > 0, f"No '{outcome}' votes: {outcomes}"
    finish_reasons = Counter(
        json.loads(line).get("finish_reason")
        for line in open(ff_dir / "api_calls.jsonl")
    )
    assert finish_reasons["length"] > 0
    if case["fail_first"]:
        assert icai_failed_attempts > 0 and server.num_failed_attempts > 0
