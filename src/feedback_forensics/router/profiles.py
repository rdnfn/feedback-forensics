"""Model profiles: prices and caching behaviour, as data for the planner.

The values below were measured on OpenRouter in November 2026. They are inputs
that go stale: pass your own profiles to `Router`, or build them from
OpenRouter's public model list with `fetch_profiles`.
"""

from dataclasses import replace
from datetime import timedelta
from typing import Iterable, Mapping

from loguru import logger

from feedback_forensics.router.datatypes import ModelProfile

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_PREFIX = "openrouter/"

# What OpenRouter's model list does not say (measured, November 2026)
KNOWN_CACHING = {
    # Explicit breakpoints work: up to four, prefixes of 512 tokens or more,
    # 5-minute lifetime by default. About 2.8 characters per token for the
    # annotation prompt. OpenRouter's caching guide says that breakpoints work
    # on Anthropic batch endpoints too; how often batch requests then read a
    # prefix written before is not known (`observed_hit_rate`).
    "anthropic/claude-haiku-5.5": dict(
        min_cacheable_tokens=512,
        max_breakpoints=4,
        chars_per_token=2.8,
        cache_write_multiplier_1h=2.0,
        cache_in_batch=True,
    ),
    # Four consecutive calls sharing a prefix of over 3,000 tokens, with
    # breakpoints, reported zero cached tokens: no breakpoints are sent.
    "google/gemini-3.5-flash-lite": dict(max_breakpoints=0, cache_in_batch=False),
}

# List prices (USD per million tokens): input 0.10, output 0.50, cache read
# 0.01, cache write 0.125 (5 minutes) or 0.20 (1 hour). The batch variant lists
# exactly half of each.
HAIKU_5_5 = ModelProfile(
    model="anthropic/claude-haiku-5.5",
    input_price=0.10,
    output_price=0.50,
    cache_read_multiplier=0.1,
    cache_write_multiplier_5m=1.25,
    cache_lifetime=timedelta(minutes=5),
    batch_model="anthropic/claude-haiku-5.5:batch",
    batch_price_multiplier=0.5,
    **KNOWN_CACHING["anthropic/claude-haiku-5.5"],
)

DEFAULT_PROFILES = {HAIKU_5_5.model: HAIKU_5_5}


def model_id(model: str) -> str:
    """OpenRouter id of a model name ('openrouter/provider/model' -> 'provider/model')."""
    return model.removeprefix(OPENROUTER_PREFIX)


def get_profile(profiles: Mapping[str, ModelProfile], model: str) -> ModelProfile:
    """Profile of a model, looked up by its name or OpenRouter id."""
    profile = profiles.get(model) or profiles.get(model_id(model))
    if profile is None:
        raise KeyError(
            f"No model profile for '{model}' (known: {sorted(profiles)}). Pass "
            "`profiles` to the router: create a `ModelProfile`, or use "
            "`feedback_forensics.router.profiles.fetch_profiles`."
        )
    return profile


def as_profile_map(profiles) -> dict[str, ModelProfile]:
    """Profiles (a mapping or a list of them) by model id."""
    if profiles is None:
        return dict(DEFAULT_PROFILES)
    values = profiles.values() if isinstance(profiles, Mapping) else profiles
    return {model_id(p.model): p for p in values}


def profiles_from_listing(models: Iterable[str], listing: list[dict]) -> dict:
    """Profiles from OpenRouter's model list (`data` of the response).

    Prices, cache prices and the batch variant are taken from the list. What it
    does not say (minimum cacheable length, breakpoints, characters per token)
    is taken from `KNOWN_CACHING`; models not in there get no breakpoints.
    Models missing from the list are left out.
    """
    by_id = {entry.get("id"): entry for entry in listing}
    profiles = {}
    for model in models:
        name = model_id(model)
        entry = by_id.get(name)
        if entry is None:
            logger.warning(f"Model '{name}' not found in OpenRouter's model list.")
            continue
        pricing = entry.get("pricing") or {}
        token_price = float(pricing["prompt"])
        fields = dict(
            model=name,
            input_price=token_price * 1e6,
            output_price=float(pricing["completion"]) * 1e6,
        )

        def multiplier(field):
            value = pricing.get(field)
            if value is None or not token_price:
                return None
            return float(value) / token_price

        read, write = multiplier("input_cache_read"), multiplier("input_cache_write")
        if read is not None:
            fields["cache_read_multiplier"] = read
        if write is not None:
            fields["cache_write_multiplier_5m"] = write
            # Replaced below if known. Otherwise assumed: 1-hour writes cost
            # 1.6 times the 5-minute ones (as for claude-haiku-5.5).
            fields["cache_write_multiplier_1h"] = write * 1.6
        batch = by_id.get(name + ":batch")
        if batch is not None:
            fields["batch_model"] = batch["id"]
            batch_price = float((batch.get("pricing") or {}).get("prompt") or 0)
            if batch_price and token_price:
                fields["batch_price_multiplier"] = batch_price / token_price
        fields.update(KNOWN_CACHING.get(name, {}))
        profiles[name] = ModelProfile(**fields)
    return profiles


def fetch_profiles(models: Iterable[str]) -> dict[str, ModelProfile]:
    """Profiles of the given models from OpenRouter's public model list (one
    request, no API key needed). See `profiles_from_listing`."""
    import requests

    response = requests.get(OPENROUTER_MODELS_URL, timeout=30)
    response.raise_for_status()
    return profiles_from_listing(models, response.json()["data"])


def with_observation(profile: ModelProfile, hit_rate: float) -> ModelProfile:
    """The profile with the batch cache hit rate observed in a run."""
    return replace(profile, observed_hit_rate=hit_rate, cache_in_batch=hit_rate > 0)
