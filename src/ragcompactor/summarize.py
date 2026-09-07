"""Summarization backends.

The compactor depends only on the ``Summarizer`` protocol, and LiteLLM does the
provider routing underneath, so the same code path serves a hosted API
(``gpt-4o-mini``, ``claude-3-5-haiku-20241022``) and a local model
(``ollama/llama3.2:3b``) - the model string is the only thing that changes.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from ragcompactor.models import SummaryResult
from ragcompactor.tokens import count_tokens

SYSTEM_PROMPT = (
    "You compact a retrieval corpus. You are given several chunks that have been "
    "measured as near-duplicates of each other. Rewrite them as ONE self-contained "
    "chunk that preserves every distinct fact, number, name and qualifier found in "
    "any of them. Do not add information that is not present. Do not editorialize, "
    "and do not refer to the chunks or to the merging process. Return only the "
    "merged text."
)


def build_prompt(texts: list[str]) -> str:
    parts = [f"--- chunk {i + 1} ---\n{t.strip()}" for i, t in enumerate(texts)]
    return (
        "Merge the following near-duplicate chunks into a single chunk.\n\n"
        + "\n\n".join(parts)
    )


@runtime_checkable
class Summarizer(Protocol):
    model: str

    def summarize(self, texts: list[str]) -> SummaryResult: ...


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-z0-9']+", a.lower()))
    wb = set(re.findall(r"[a-z0-9']+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class StubSummarizer:
    """Deterministic, offline, zero-cost extractive summarizer.

    Takes the longest chunk as the spine and adds a sentence from the other
    chunks only when it is not a near-repeat of something already kept
    (word-level Jaccard below ``sentence_threshold``). That makes it genuinely
    compressive rather than a concatenation, so offline demos and tests show the
    same *shape* of result a real model would - just cruder. It is not a
    substitute for an LLM: it can only select sentences, never rewrite them.
    """

    model = "stub"

    def __init__(self, sentence_threshold: float = 0.6) -> None:
        self.sentence_threshold = sentence_threshold

    def summarize(self, texts: list[str]) -> SummaryResult:
        if not texts:
            return SummaryResult(text="", model=self.model)

        ordered = sorted(texts, key=len, reverse=True)
        kept: list[str] = list(_sentences(ordered[0]))

        for other in ordered[1:]:
            for sentence in _sentences(other):
                if len(sentence) < 15:
                    continue
                if any(_jaccard(sentence, k) >= self.sentence_threshold for k in kept):
                    continue
                kept.append(sentence)

        return SummaryResult(text=" ".join(kept).strip(), model=self.model)


class LiteLLMSummarizer:
    """Routes the summarization call through LiteLLM to any supported provider."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def summarize(self, texts: list[str]) -> SummaryResult:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The litellm summarizer needs the 'llm' extra: pip install 'ragcompactor[llm]'"
            ) from exc

        prompt = build_prompt(texts)
        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        text = (response.choices[0].message.content or "").strip()

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if not prompt_tokens:  # provider did not report usage - estimate locally
            prompt_tokens = count_tokens(SYSTEM_PROMPT + prompt)
        if not completion_tokens:
            completion_tokens = count_tokens(text)

        return SummaryResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self.model,
        )


def get_summarizer(config) -> Summarizer:
    backend = (config.llm_backend or "").lower()
    if backend in {"stub", "none", "offline"}:
        return StubSummarizer()
    if backend in {"litellm", "api", "llm"}:
        return LiteLLMSummarizer(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.max_summary_tokens,
            timeout=config.llm_timeout,
        )
    raise ValueError(f"unknown llm_backend: {config.llm_backend!r}")
