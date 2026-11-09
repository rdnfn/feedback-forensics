"""LLM model access for Feedback Forensics (generation and annotation).

Replaces `inverse_cai.models.get_model`. Model names use a provider prefix:
`openai/<model>`, `anthropic/<model>` or `openrouter/<provider>/<model>`.
"""

import os
from typing import Any

from dotenv import load_dotenv

# Load API keys from a `secrets.toml` file in the working directory (if present),
# matching the behaviour documented for `ff-annotate` (previously done by ICAI).
# Existing environment variables take precedence.
load_dotenv("./secrets.toml")

# Custom OpenRouter headers
# https://openrouter.ai/docs/api-reference/overview#headers
OPENROUTER_HEADERS = {
    "X-Title": "Feedback Forensics",
    # HTTP-Referer is required in addition to X-Title for OpenRouter to show an
    # app name in the activity overview.
    "HTTP-Referer": "https://github.com/rdnfn/feedback-forensics",
}
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


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
        http_async_client: Optional `httpx.AsyncClient` for OpenAI-compatible
            models. Pass a client created inside the running event loop when
            calling the model from more than one `asyncio.run()` in a process:
            LangChain otherwise shares a cached connection pool across model
            instances, which breaks ("Event loop is closed") once the loop that
            created it is closed.

    Returns:
        LangChain chat model
    """
    # Imported lazily to keep `import feedback_forensics.models` cheap.
    from langchain_openai import ChatOpenAI

    openai_kwargs = {}
    if http_async_client is not None:
        openai_kwargs["http_async_client"] = http_async_client

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

        if reasoning_effort is not None:
            raise ValueError(
                "reasoning_effort is not supported for 'anthropic/' models, "
                "use the 'openrouter/anthropic/...' model name instead."
            )
        return ChatAnthropic(
            model=name.split("/", 1)[1],
            max_tokens=max_tokens,
            temperature=temp,
        )

    if name.startswith("openrouter/"):
        parts = name.split("/", 2)
        if len(parts) < 3:
            raise ValueError(
                "OpenRouter model format should be 'openrouter/provider/model'"
            )
        model_id = "/".join(parts[1:])

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
