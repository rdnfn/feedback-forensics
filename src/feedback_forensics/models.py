"""LLM model access for Feedback Forensics (generation and annotation).

Replaces `inverse_cai.models.get_model`. Model names use a provider prefix:
`openai/<model>`, `anthropic/<model>` or `openrouter/<provider>/<model>`.
"""

import importlib
import os
import socket
import urllib.request
from types import ModuleType
from typing import Any

# Note: API keys can be set in a `secrets.toml` file, which is loaded on
# `import feedback_forensics` (see `feedback_forensics.environment`).

# Custom OpenRouter headers
# https://openrouter.ai/docs/api-reference/overview#headers
OPENROUTER_HEADERS = {
    "X-Title": "Feedback Forensics",
    # HTTP-Referer is required in addition to X-Title for OpenRouter to show an
    # app name in the activity overview.
    "HTTP-Referer": "https://github.com/rdnfn/feedback-forensics",
}
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# Timeouts (seconds) of API requests made with `create_async_http_client`
REQUEST_TIMEOUT_S = 600.0
CONNECT_TIMEOUT_S = 30.0

# TCP keepalive of `create_async_http_client` (as LangChain's default OpenAI
# client): probe idle connections after 60 s, then every 10 s, and drop them
# after 3 unanswered probes or when sent data stays unacknowledged for 120 s.
TCP_KEEPALIVE_OPTIONS = {
    "TCP_KEEPIDLE": 60,
    "TCP_KEEPALIVE": 60,  # macOS name of TCP_KEEPIDLE
    "TCP_KEEPINTVL": 10,
    "TCP_KEEPCNT": 3,
    "TCP_USER_TIMEOUT": 120_000,  # milliseconds
}


def check_model_settings(name: str, reasoning_effort: str | None = None) -> None:
    """Raise a ValueError if `get_model` does not support a model name or setting.

    Allows checking settings before any work is done (e.g. output files written).
    """
    if not name.startswith(("openai/", "anthropic/", "openrouter/")):
        raise ValueError(
            f"Unsupported model name '{name}'. Use one of the prefixes "
            "'openai/', 'anthropic/' or 'openrouter/'."
        )
    if name.startswith("anthropic/") and reasoning_effort is not None:
        raise ValueError(
            "reasoning_effort is not supported for 'anthropic/' models, "
            "use the 'openrouter/anthropic/...' model name instead."
        )
    if name.startswith("openrouter/") and len(name.split("/", 2)) < 3:
        raise ValueError(
            "OpenRouter model format should be 'openrouter/provider/model'"
        )


def get_model(
    name: str,
    temp: float = 0.0,
    max_tokens: int = 1000,
    reasoning_effort: str | None = None,
    http_async_client: Any = None,
) -> Any:
    """Get a LangChain chat model instance.

    Args:
        name: Model name with provider prefix (e.g. "openai/gpt-4o-2024-05-13")
        temp: Temperature for generation (default: 0.0)
        max_tokens: Maximum tokens to generate, incl. reasoning tokens (default: 1000)
        reasoning_effort: Optional reasoning effort ("minimal", "low", "medium",
            "high") for reasoning models. None: provider default.
        http_async_client: Optional HTTP client for async calls, created with
            `create_async_http_client` inside the running event loop. Needed
            when calling the model from more than one `asyncio.run()` in a
            process (see `create_async_http_client`). Its timeout applies to
            all requests, and it is responsible for proxy settings
            (`OPENAI_PROXY` and `ANTHROPIC_PROXY` are not applied on top).

    Returns:
        LangChain chat model
    """
    check_model_settings(name, reasoning_effort)

    # Imported lazily to keep `import feedback_forensics.models` cheap.
    from langchain_openai import ChatOpenAI

    openai_kwargs = {}
    if http_async_client is not None:
        openai_kwargs.update(
            http_async_client=http_async_client,
            # LangChain's default timeout (None: no timeout) would override the
            # client's timeout.
            timeout=http_async_client.timeout,
            # LangChain refuses a proxy (default: `OPENAI_PROXY`) together with
            # a client; the client applies proxy settings itself.
            openai_proxy=None,
        )
        if "http_socket_options" in ChatOpenAI.model_fields:  # langchain-openai>=1.2
            # No socket options for LangChain's own (sync) client, unused here:
            # else LangChain warns that proxy settings are ignored (by that client).
            openai_kwargs["http_socket_options"] = ()

    if name.startswith("openai/"):
        if reasoning_effort is not None:
            openai_kwargs["reasoning_effort"] = reasoning_effort
        return ChatOpenAI(
            model=name.split("/", 1)[1],
            max_tokens=max_tokens,
            temperature=temp,
            **openai_kwargs,
        )

    if name.startswith("anthropic/"):
        from langchain_anthropic import ChatAnthropic

        model = ChatAnthropic(
            model=name.split("/", 1)[1],
            max_tokens=max_tokens,
            temperature=temp,
        )
        if http_async_client is not None:
            import anthropic

            # ChatAnthropic has no option for a custom HTTP client: replace its
            # async API client (created lazily, with a process-wide cached HTTP
            # client) by one using the given HTTP client.
            model._async_client = anthropic.AsyncAnthropic(
                **{**model._client_params, "timeout": http_async_client.timeout},
                http_client=http_async_client,
            )
        return model

    if name.startswith("openrouter/"):
        model_id = name.split("/", 1)[1]

        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable must be set for OpenRouter models"
            )

        extra_body = {"usage": {"include": True}}
        if reasoning_effort is not None:
            # https://openrouter.ai/docs/use-cases/reasoning-tokens
            extra_body["reasoning"] = {"effort": reasoning_effort}

        return ChatOpenAI(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temp,
            api_key=openrouter_api_key,
            base_url=OPENROUTER_API_BASE,
            default_headers=OPENROUTER_HEADERS,
            # Ask OpenRouter to report the cost of each request in the usage field
            # (https://openrouter.ai/docs/use-cases/usage-accounting)
            extra_body=extra_body,
            **openai_kwargs,
        )

    raise ValueError(
        f"Unsupported model name '{name}'. Use one of the prefixes "
        "'openai/', 'anthropic/' or 'openrouter/'."
    )


def create_async_http_client(name: str, max_connections: int = 1000) -> Any:
    """Create an HTTP client for async API calls to the provider of model `name`.

    Pass it to `get_model` and close it before the event loop closes (e.g. with
    `async with`). Unlike LangChain's default HTTP clients, which are cached
    process-wide, it is bound to the running event loop: the connections of a
    cached client break ("Event loop is closed") when it is used again from a
    later `asyncio.run()` in the same process.

    Proxies: as with LangChain's default clients, the proxy settings from the
    environment (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`; for `anthropic/` models
    also `ALL_PROXY`) are used, and take precedence over `OPENAI_PROXY`
    (`openai/`, `openrouter/` models) or `ANTHROPIC_PROXY` (`anthropic/` models),
    which, if set, is used for all other requests. Like LangChain's default
    OpenAI client, it enables TCP keepalive (`TCP_KEEPALIVE_OPTIONS`) if no
    proxy is configured: requests on connections that silently died (e.g. after
    a network change) fail within about two minutes instead of hanging forever.
    Requests time out if the API does not respond within `REQUEST_TIMEOUT_S`
    (LangChain default: no timeout).

    Args:
        name: Model name with provider prefix (see `get_model`).
        max_connections: Maximum number of concurrent connections.
    """
    if name.startswith("anthropic/"):
        import anthropic as sdk

        proxy = os.environ.get("ANTHROPIC_PROXY")
    else:
        import openai as sdk

        proxy = os.environ.get("OPENAI_PROXY")

    # The SDK's own client class, to get its defaults and HTTP library
    httpx = _get_http_library(sdk.DefaultAsyncHttpxClient)
    limits = httpx.Limits(
        max_connections=max_connections, max_keepalive_connections=100
    )
    kwargs = dict(
        timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        limits=limits,
    )
    if proxy and sdk.__name__ == "anthropic":
        # As LangChain's default Anthropic client: the SDK's client mounts the
        # proxy settings from the environment over the proxy
        kwargs["proxy"] = proxy
    elif proxy:
        # As LangChain's default OpenAI client with `OPENAI_PROXY`: the proxy
        # settings from the environment (except `ALL_PROXY`) take precedence
        # over this mount for all URLs
        kwargs["mounts"] = {
            "all://": httpx.AsyncHTTPTransport(
                proxy=httpx.Proxy(proxy),
                socket_options=_get_tcp_keepalive_socket_options(),
                limits=limits,
            )
        }
    elif not _env_proxy_configured():
        # A custom transport (needed for socket options) disables the proxy
        # settings from the environment, hence not used with those.
        kwargs["transport"] = httpx.AsyncHTTPTransport(
            socket_options=_get_tcp_keepalive_socket_options(),
            limits=limits,
        )
    return sdk.DefaultAsyncHttpxClient(**kwargs)


def _get_http_library(client_cls: type) -> ModuleType:
    """The HTTP library of an SDK's HTTP client class: `httpx`, or its fork `httpx2`
    (used by openai>=3 and anthropic>=2; the latter rejects `httpx` objects)."""
    for module_name in ("httpx2", "httpx"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if issubclass(client_cls, module.AsyncClient):
            return module
    raise TypeError(f"{client_cls} is not an httpx or httpx2 client.")


def _env_proxy_configured() -> bool:
    """Whether httpx would use a proxy from the environment or system settings."""
    # Note: getproxies() also lists e.g. OPENAI_PROXY (as scheme "openai")
    return any(
        scheme in ("http", "https", "all") for scheme in urllib.request.getproxies()
    )


def _get_tcp_keepalive_socket_options() -> list[tuple[int, int, int]]:
    """Socket options enabling TCP keepalive, as far as the platform supports them."""
    options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)] + [
        (socket.IPPROTO_TCP, getattr(socket, option), value)
        for option, value in TCP_KEEPALIVE_OPTIONS.items()
        if hasattr(socket, option)
    ]
    # Leave out options the platform rejects (else every connection would fail)
    supported = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        for option in options:
            try:
                probe.setsockopt(*option)
            except OSError:
                continue
            supported.append(option)
    return supported
