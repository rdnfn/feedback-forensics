"""Tests for the HTTP layer of annotation (against the mock LLM API): per-run HTTP
clients, timeouts, TCP keepalive, proxy settings and retries.

FF's own retries (`voting.run_with_http_retries`) only come into play once the
API clients' own retries are exhausted, hence the mock server fails the first
`SDK_ATTEMPTS` attempts of every request in the retry tests.
"""

import asyncio
import contextlib
import logging
import random
import socket
import sys

import anthropic
import httpx
import openai
import pytest

from feedback_forensics import models
from feedback_forensics.annotation import annotate_dataset, voting
from feedback_forensics.annotation.mock_api import (
    MockLLMServer,
    anthropic_to_openai_messages,
)
from feedback_forensics.annotation.testutils import (
    MOCK_MODEL,
    expected_vote,
    make_pairs_df,
)
from feedback_forensics.annotation.usage import UsageTracker
from feedback_forensics.annotation.voting import (
    INVALID_VOTE,
    VotingConfig,
    get_preferred_text,
    get_rejected_text,
    get_votes_for_principles,
    get_votes_for_principles_async,
    run_with_http_retries,
)

ANTHROPIC_MODEL = "anthropic/mock-annotator"
OPENROUTER_MODEL = "openrouter/mock/annotator"
MODELS = [MOCK_MODEL, OPENROUTER_MODEL, ANTHROPIC_MODEL]
PRINCIPLES = [
    "Select the response that is more concise",
    "Select the response that is more friendly",
    "Select the response that uses more formatting",
]
# Attempts per API call made by the OpenAI and Anthropic clients (2 retries)
SDK_ATTEMPTS = 3
# API host that can only be reached via a proxy (the mock server)
PROXIED_API_URL = "http://mock-llm-api.invalid"
# Proxy that cannot be reached
DEAD_PROXY_URL = "http://mock-proxy.invalid:8080"
PROXY_ENV_VARS = ["OPENAI_PROXY", "ANTHROPIC_PROXY"] + [
    var
    for scheme in ("http", "https", "all", "no")
    for var in (f"{scheme}_proxy", f"{scheme.upper()}_PROXY")
]


@pytest.fixture
def llm_server(monkeypatch):
    """Factory fixture: start a mock LLM server and point the API clients to it."""
    for var in PROXY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    with contextlib.ExitStack() as stack:

        def start(**kwargs) -> MockLLMServer:
            server = stack.enter_context(MockLLMServer(**kwargs))
            set_api_urls(monkeypatch, server.base_url, server.anthropic_base_url)
            monkeypatch.setattr(models, "OPENROUTER_API_BASE", server.base_url)
            monkeypatch.setenv("OPENAI_API_KEY", "mock-key")
            monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-key")
            monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")
            return server

        yield start


def set_api_urls(monkeypatch, openai_url: str, anthropic_url: str) -> None:
    for var in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
        monkeypatch.setenv(var, openai_url)
    for var in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_URL"):
        monkeypatch.setenv(var, anthropic_url)


def annotate(df, model: str, max_duration_s: float = 60) -> tuple[dict, UsageTracker]:
    """Annotate `df` with `PRINCIPLES` in a new event loop, returns (votes, usage)."""
    usage_tracker = UsageTracker()
    result = asyncio.run(
        asyncio.wait_for(
            get_votes_for_principles_async(
                feedback_df=df,
                principles=dict(enumerate(PRINCIPLES)),
                config=VotingConfig(model=model),
                usage_tracker=usage_tracker,
            ),
            max_duration_s,  # fail instead of hanging
        )
    )
    return result.votes, usage_tracker


def expected_votes(df) -> dict:
    """Votes that the (content-deterministic) mock API implies."""
    return {
        idx: {
            key: expected_vote(
                principle,
                get_preferred_text(row),
                get_rejected_text(row),
                position_debias=False,
            )
            for key, principle in enumerate(PRINCIPLES)
        }
        for idx, row in df.iterrows()
    }


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("failure", [500, 429, "disconnect", "stall"])
def test_errors_beyond_api_client_retries_are_retried(
    llm_server, retry_waits, monkeypatch, model, failure
):
    """Server errors, rate limits, connection errors and timeouts are retried (for
    all providers), also when they persist beyond the API clients' own retries."""
    # Requests to a stalled server time out (and are retried)
    monkeypatch.setattr(models, "REQUEST_TIMEOUT_S", 1.0)
    server = llm_server(fail_attempts=SDK_ATTEMPTS, failure=failure, stall_s=120)
    df = make_pairs_df(4, seed="e")

    votes, usage = annotate(df, model)

    assert votes == expected_votes(df)
    assert usage.num_failed_calls == 0
    # Each call: all attempts of the API client fail, FF retries once
    assert len(retry_waits) == len(df)
    assert server.num_failed_attempts == SDK_ATTEMPTS * len(df)
    assert server.num_attempts == (SDK_ATTEMPTS + 1) * len(df)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("status", [400, 401])
def test_client_errors_are_not_retried(llm_server, retry_waits, model, status):
    """Other client errors (e.g. invalid request or API key) would fail again: no
    retries, the votes become invalid."""
    server = llm_server(fail_attempts=1, failure=status)  # a retry would succeed
    df = make_pairs_df(4, seed="c")

    votes, usage = annotate(df, model)

    assert all(v == INVALID_VOTE for vs in votes.values() for v in vs.values())
    assert usage.num_failed_calls == len(df)
    assert retry_waits == []
    assert server.num_attempts == len(df)


API_REQUEST = httpx.Request("POST", "https://api.invalid/v1/chat/completions")


def status_error(sdk, status: int, message: str | None = None):
    """An API client's (`sdk`: openai or anthropic) error for an HTTP status."""
    response = httpx.Response(status, request=API_REQUEST)
    return sdk.APIStatusError(
        message or f"Error code: {status}", response=response, body=None
    )


@pytest.mark.parametrize(
    "error, retried",
    [
        # Rate limits, server errors and other transient HTTP errors
        *[
            (status_error(sdk, status), True)
            for sdk in (openai, anthropic)
            for status in (403, 408, 409, 420, 429, 444, 498, 499, 500, 503, 529)
        ],
        # Other client errors would fail again
        *[
            (status_error(sdk, status), False)
            for sdk in (openai, anthropic)
            for status in (400, 401, 402, 404, 422)
        ],
        # HTTP status not in the message
        (status_error(anthropic, 529, message="Overloaded"), True),
        (status_error(openai, 400, message="Bad request"), False),
        # Connection errors and timeouts
        (openai.APIConnectionError(request=API_REQUEST), True),
        (openai.APITimeoutError(API_REQUEST), True),
        (anthropic.APIConnectionError(request=API_REQUEST), True),
        (anthropic.APITimeoutError(API_REQUEST), True),
        (httpx.ConnectError("Connection refused"), True),
        (asyncio.TimeoutError(), True),
        # Errors of other libraries, with the HTTP status in the args or message
        (Exception({"code": 429, "message": "Rate limited"}), True),
        (Exception({"code": 400, "message": "Bad request"}), False),
        (Exception("Error code: 503 - Service unavailable"), True),
        (Exception("Error code: 401 - Unauthorized"), False),
        # Not API errors (e.g. bugs)
        (ValueError("Invalid value"), False),
        (KeyError("choices"), False),
    ],
    ids=repr,
)
def test_retried_errors(error, retried):
    assert voting._fatal_model_error(error) is not retried


def test_retried_http_errors_are_icais():
    """The same HTTP errors are retried as by ICAI, apart from HTTP 409 (see the
    `voting` module docstring)."""
    icai_utils = pytest.importorskip("inverse_cai.algorithm.utils")
    for status in range(400, 600):
        error = status_error(openai, status)
        assert voting._fatal_model_error(error) == (
            icai_utils._fatal_model_error(error) and status != 409
        ), status


def rate_limit_error():
    response = httpx.Response(429, request=API_REQUEST)
    return openai.RateLimitError("Error code: 429", response=response, body=None)


def server_error():
    response = httpx.Response(500, request=API_REQUEST)
    return openai.InternalServerError("Error code: 500", response=response, body=None)


def timeout_error():
    return openai.APITimeoutError(API_REQUEST)


def run_until_giving_up(retry_fn, make_error) -> int:
    """Run a call that always fails with `retry_fn`, returns the number of tries."""
    tries = 0

    async def failing_call():
        nonlocal tries
        tries += 1
        raise make_error()

    with pytest.raises(type(make_error())):
        asyncio.run(retry_fn(failing_call))
    return tries


MAX_RETRY_WAITS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


@pytest.fixture
def jitter_ranges(monkeypatch) -> list[tuple[float, float]]:
    """Make the random backoff jitter return the longest wait, returns the ranges
    the waits were drawn from."""
    ranges = []

    def uniform(a, b):
        ranges.append((a, b))
        return b

    monkeypatch.setattr(random, "uniform", uniform)
    return ranges


@pytest.mark.parametrize("make_error", [rate_limit_error, server_error, timeout_error])
def test_retry_budget(retry_waits, jitter_ranges, make_error):
    """Up to 10 tries with exponential backoff without cap and with full jitter, as
    in ICAI (see `test_retry_budget_is_icais`): persistent rate limits or server
    errors only make votes invalid after FF's waits between tries of about 4
    minutes on average (8.5 minutes at most). The API clients' own retries
    (incl. their waits, e.g. for a `Retry-After` header) and the duration of the
    requests themselves add to this."""
    assert run_until_giving_up(run_with_http_retries, make_error) == 10
    assert retry_waits == MAX_RETRY_WAITS
    assert jitter_ranges == [(0, wait) for wait in MAX_RETRY_WAITS]


def test_retry_budget_is_icais(retry_waits, jitter_ranges):
    """The retry budget tested in `test_retry_budget` is ICAI's."""
    icai_utils = pytest.importorskip("inverse_cai.algorithm.utils")

    for make_error in (rate_limit_error, server_error):
        retry_waits.clear()
        jitter_ranges.clear()
        assert run_until_giving_up(icai_utils.run_with_http_retries, make_error) == 10
        assert retry_waits == MAX_RETRY_WAITS
        assert jitter_ranges == [(0, wait) for wait in MAX_RETRY_WAITS]


@pytest.mark.parametrize("model", MODELS)
def test_each_run_uses_its_own_http_client(llm_server, model):
    """Every run (event loop) uses its own HTTP client, also for `anthropic/` models.

    Otherwise later runs reuse keep-alive connections of an earlier run's closed
    event loop ("Event loop is closed"): the API client silently re-sends the
    requests (billed but not logged), and calls whose attempts all fail make
    their votes invalid (e.g. small runs after a large one, as in
    `ff-model-comparison`).
    """
    server = llm_server(keep_alive=True)
    datasets = [make_pairs_df(n, seed=f"r{n}") for n in (60, 8, 8)]
    for df in datasets:
        num_attempts = server.num_attempts

        votes, usage = annotate(df, model)

        assert votes == expected_votes(df)
        assert usage.num_failed_calls == 0
        assert server.num_attempts - num_attempts == len(df)  # no re-sent requests


@pytest.mark.parametrize("model", MODELS)
def test_only_incomplete_responses_reported(llm_server, log_messages, model):
    """Regression: every `anthropic/` response was reported as not finished with
    'stop' reason (Anthropic reports the stop reason "end_turn" instead)."""
    llm_server()
    df = make_pairs_df(10, seed="f")

    _, usage = annotate(df, model)

    unfinished = [m for _, m in log_messages if "did not finish" in m]
    assert len(unfinished) == usage.num_truncated_calls > 0
    truncated_reason = "max_tokens" if model == ANTHROPIC_MODEL else "length"
    assert all(f"instead with '{truncated_reason}' reason" in m for m in unfinished)


@pytest.mark.parametrize("api", ["annotate_dataset", "get_votes_for_principles"])
def test_reasoning_effort_not_supported_for_anthropic_models(tmp_path, llm_server, api):
    """Fails before any API call or output file, rather than silently ignoring the
    setting (while recording it in the output metadata)."""
    server = llm_server()
    df = make_pairs_df(4, seed="a")
    df.to_csv(tmp_path / "data.csv", index=False)

    with pytest.raises(ValueError, match="reasoning_effort"):
        if api == "annotate_dataset":
            annotate_dataset(
                tmp_path / "data.csv",
                PRINCIPLES,
                ANTHROPIC_MODEL,
                output_dir=tmp_path / "out",
                reasoning_effort="high",
            )
        else:
            get_votes_for_principles(
                feedback_df=df,
                principles=dict(enumerate(PRINCIPLES)),
                config=VotingConfig(model=ANTHROPIC_MODEL, reasoning_effort="high"),
                cache_path=tmp_path / "out" / "cache" / "votes",
            )
    assert server.num_attempts == 0
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "model, error",
    [
        ("mock-annotator", "Unsupported model name"),
        ("openrouter/mock-annotator", "openrouter/provider/model"),
    ],
)
def test_unsupported_model_names_rejected_before_writing_files(
    tmp_path, llm_server, model, error
):
    server = llm_server()
    make_pairs_df(4, seed="a").to_csv(tmp_path / "data.csv", index=False)

    with pytest.raises(ValueError, match=error):
        annotate_dataset(
            tmp_path / "data.csv", PRINCIPLES, model, output_dir=tmp_path / "out"
        )
    assert server.num_attempts == 0
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "model, proxy_var",
    [
        (MOCK_MODEL, "OPENAI_PROXY"),
        ("openrouter/mock/annotator", "OPENAI_PROXY"),
        (MOCK_MODEL, "HTTP_PROXY"),
        (ANTHROPIC_MODEL, "ANTHROPIC_PROXY"),
        (ANTHROPIC_MODEL, "HTTP_PROXY"),
    ],
)
def test_proxy_settings(llm_server, retry_waits, monkeypatch, caplog, model, proxy_var):
    """Proxy settings from the environment are honoured (as by LangChain's own HTTP
    clients). The mock server acts as the proxy for an API that can't be reached
    directly."""
    server = llm_server()
    set_api_urls(monkeypatch, f"{PROXIED_API_URL}/v1", PROXIED_API_URL)
    monkeypatch.setattr(models, "OPENROUTER_API_BASE", f"{PROXIED_API_URL}/v1")
    monkeypatch.setenv(proxy_var, server.anthropic_base_url)  # the server's root URL
    # LangChain warns about ignored proxy settings only once per process
    monkeypatch.setattr(
        "langchain_openai.chat_models._client_utils._proxy_env_warning_emitted",
        False,
        raising=False,
    )
    df = make_pairs_df(4, seed="p")

    with caplog.at_level(logging.WARNING):
        votes, usage = annotate(df, model)

    assert votes == expected_votes(df)
    assert usage.num_failed_calls == 0
    assert len(server.paths) == len(df)
    assert all(path.startswith(f"{PROXIED_API_URL}/v1/") for path in server.paths)
    # No (misleading) warnings of LangChain, e.g. that proxy settings are ignored
    assert [r.getMessage() for r in caplog.records if r.name.startswith("lang")] == []


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("env_var", ["HTTP_PROXY", "NO_PROXY"])
def test_proxy_settings_from_environment_take_precedence(
    llm_server, retry_waits, monkeypatch, model, env_var
):
    """As with LangChain's default clients (used by ICAI): `HTTP(S)_PROXY` and
    `NO_PROXY` take precedence over `OPENAI_PROXY` / `ANTHROPIC_PROXY`."""
    server = llm_server()
    provider_proxy_var = (
        "ANTHROPIC_PROXY" if model.startswith("anthropic/") else "OPENAI_PROXY"
    )
    monkeypatch.setenv(provider_proxy_var, DEAD_PROXY_URL)
    if env_var == "HTTP_PROXY":
        # The mock server acts as the proxy for an API that can't be reached
        set_api_urls(monkeypatch, f"{PROXIED_API_URL}/v1", PROXIED_API_URL)
        monkeypatch.setattr(models, "OPENROUTER_API_BASE", f"{PROXIED_API_URL}/v1")
        monkeypatch.setenv("HTTP_PROXY", server.anthropic_base_url)
        expected_path_prefix = f"{PROXIED_API_URL}/v1/"
    else:
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")  # the mock server's host
        expected_path_prefix = "/v1/"
    df = make_pairs_df(4, seed="p")

    votes, usage = annotate(df, model, max_duration_s=20)

    assert votes == expected_votes(df)
    assert usage.num_failed_calls == 0
    assert len(server.paths) == len(df)
    assert all(path.startswith(expected_path_prefix) for path in server.paths)


# LangChain's defaults (langchain-openai's `_default_socket_options`), as far as
# the platform has these options
EXPECTED_TCP_KEEPALIVE_OPTIONS = {
    "TCP_KEEPIDLE": 60,
    "TCP_KEEPALIVE": 60,  # macOS name of TCP_KEEPIDLE
    "TCP_KEEPINTVL": 10,
    "TCP_KEEPCNT": 3,
    "TCP_USER_TIMEOUT": 120_000,  # milliseconds
}


@pytest.mark.parametrize("model", MODELS)
def test_http_client_enables_tcp_keepalive(llm_server, model):
    """Like LangChain's default OpenAI client (used by ICAI): requests on connections
    that silently died (e.g. after a network change) fail and are retried instead of
    hanging forever."""
    if models._env_proxy_configured():
        pytest.skip("Proxy configured in the system settings (no socket options)")
    server = llm_server(keep_alive=True)

    async def get_socket_options() -> dict:
        """Keepalive options of a connection (those the platform supports)."""
        async with models.create_async_http_client(model) as client:
            response = await client.post(
                f"{server.base_url}/chat/completions",
                json={"messages": [{"role": "user", "content": "Hi"}]},
            )
            sock = response.extensions["network_stream"].get_extra_info("socket")
            options = {
                "SO_KEEPALIVE": sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
            }
            for name in EXPECTED_TCP_KEEPALIVE_OPTIONS:
                with contextlib.suppress(AttributeError, OSError):
                    options[name] = sock.getsockopt(
                        socket.IPPROTO_TCP, getattr(socket, name)
                    )
            return options

    options = asyncio.run(get_socket_options())

    assert options.pop("SO_KEEPALIVE")
    assert options == {
        name: value
        for name, value in EXPECTED_TCP_KEEPALIVE_OPTIONS.items()
        if name in options
    }
    if sys.platform == "linux":
        assert options.keys() == {
            "TCP_KEEPIDLE",
            "TCP_KEEPINTVL",
            "TCP_KEEPCNT",
            "TCP_USER_TIMEOUT",
        }
    elif sys.platform == "darwin":
        assert {"TCP_KEEPALIVE", "TCP_KEEPINTVL", "TCP_KEEPCNT"} <= options.keys()


def test_mock_anthropic_api_without_system_prompt():
    """(Test infrastructure) The system prompt of Anthropic requests is optional."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    assert anthropic_to_openai_messages({"messages": messages}) == [
        {"role": "user", "content": "Hi"}
    ]
