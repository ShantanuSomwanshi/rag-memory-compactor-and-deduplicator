"""Token accounting.

Uses tiktoken when it is installed so the benchmark numbers are real, and falls
back to a characters/4 estimate otherwise. The fallback is flagged on the
result so a report never silently presents an estimate as a measurement.
"""

from __future__ import annotations

from functools import lru_cache

_FALLBACK_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=8)
def _encoder(model: str):
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - depends on optional extra
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover - tiktoken present but unusable
            return None


def tokenizer_is_exact(model: str = "gpt-4o-mini") -> bool:
    """True when real tokenization is available (tiktoken installed)."""
    return _encoder(model) is not None


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in ``text``."""
    if not text:
        return 0
    enc = _encoder(model)
    if enc is None:
        return max(1, len(text) // _FALLBACK_CHARS_PER_TOKEN)
    return len(enc.encode(text))


def count_many(texts: list[str], model: str = "gpt-4o-mini") -> int:
    return sum(count_tokens(t, model) for t in texts)
