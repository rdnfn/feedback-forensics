"""Tests of model profiles."""

import pytest

from feedback_forensics.router.profiles import (
    DEFAULT_PROFILES,
    HAIKU_5_5,
    as_profile_map,
    get_profile,
    profiles_from_listing,
)

LISTING = [
    {
        "id": "anthropic/claude-haiku-5.5",
        "pricing": {
            "prompt": "0.0000001",
            "completion": "0.0000005",
            "input_cache_read": "0.00000001",
            "input_cache_write": "0.000000125",
        },
    },
    {
        "id": "anthropic/claude-haiku-5.5:batch",
        "pricing": {"prompt": "0.00000005", "completion": "0.00000025"},
    },
    {
        "id": "google/gemini-3.5-flash-lite",
        "pricing": {
            "prompt": "0.0000002",
            "completion": "0.0000008",
            "input_cache_read": "0.00000002",
            "input_cache_write": "0.0000002",
        },
    },
    {"id": "x/plain", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
]


def test_default_profile_holds_the_measured_values():
    profile = DEFAULT_PROFILES["anthropic/claude-haiku-5.5"]
    assert (profile.input_price, profile.output_price) == (0.10, 0.50)
    # Cache read $0.01, write $0.125 (5 minutes) or $0.20 (1 hour) per million
    assert profile.input_price * profile.cache_read_multiplier == pytest.approx(0.01)
    assert profile.input_price * profile.cache_write_multiplier("5m") == pytest.approx(
        0.125
    )
    assert profile.input_price * profile.cache_write_multiplier("1h") == pytest.approx(
        0.20
    )
    assert (profile.min_cacheable_tokens, profile.max_breakpoints) == (512, 4)
    assert profile.cache_lifetime.total_seconds() == 300
    assert profile.chars_per_token == 2.8
    assert profile.batch_model == "anthropic/claude-haiku-5.5:batch"
    assert profile.batch_price_multiplier == 0.5
    assert profile.observed_hit_rate is None


def test_profiles_from_openrouter_listing():
    models = [
        "openrouter/anthropic/claude-haiku-5.5",
        "google/gemini-3.5-flash-lite",
        "x/plain",
        "x/missing",
    ]
    profiles = profiles_from_listing(models, LISTING)
    assert set(profiles) == {
        "anthropic/claude-haiku-5.5",
        "google/gemini-3.5-flash-lite",
        "x/plain",
    }
    haiku = profiles["anthropic/claude-haiku-5.5"]
    for field in (
        "input_price",
        "output_price",
        "cache_read_multiplier",
        "cache_write_multiplier_5m",
        "cache_write_multiplier_1h",
        "batch_price_multiplier",
    ):
        assert getattr(haiku, field) == pytest.approx(getattr(HAIKU_5_5, field))
    assert haiku.batch_model == HAIKU_5_5.batch_model
    assert haiku.max_breakpoints == 4 and haiku.min_cacheable_tokens == 512
    # Breakpoints gave no cached tokens for this model: none are planned
    gemini = profiles["google/gemini-3.5-flash-lite"]
    assert gemini.max_breakpoints == 0 and gemini.batch_model is None
    # Unknown caching behaviour: no breakpoints either
    plain = profiles["x/plain"]
    assert plain.max_breakpoints == 0 and plain.cache_read_multiplier == 1.0
    assert plain.input_price == pytest.approx(1.0)


def test_profile_lookup():
    profiles = as_profile_map([HAIKU_5_5])
    assert get_profile(profiles, "openrouter/anthropic/claude-haiku-5.5") is HAIKU_5_5
    assert get_profile(profiles, "anthropic/claude-haiku-5.5") is HAIKU_5_5
    assert as_profile_map(None) == DEFAULT_PROFILES
    with pytest.raises(KeyError, match="No model profile"):
        get_profile(profiles, "openrouter/x/unknown")
