"""Minimal OpenAI-compatible chat-completions client.

Honors HTTP 429 (rate-limit) responses with bounded retry: parses the standard
`Retry-After` header (or falls back to exponential backoff) and waits before
re-issuing the same request. This keeps long benchmark runs against rate-limited
hosted endpoints (OpenAI Tier 1, Anthropic Build tiers) from aborting on
transient throttling, while still surfacing persistent failures.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import httpx


# Bounded retry on HTTP 429. Tuned conservatively:
#   - DEFAULT_RETRY_AFTER_SECONDS: fallback when no `Retry-After` header is sent
#   - MAX_RETRIES_ON_429: hard cap so a structurally-too-large request fails fast
#     instead of looping forever (e.g. Anthropic Tier 1 = 30k ITPM, single
#     request of 80k tokens will always be rejected)
DEFAULT_RETRY_AFTER_SECONDS = 60.0
MAX_RETRIES_ON_429 = 3


@dataclass
class ClientConfig:
    base_url: str
    model: str
    api_key: str = "not-needed"
    temperature: float = 0.0
    max_tokens: int = 6000  # generous — reasoning models can spend most of this on CoT
    timeout: float = 600.0
    # Reasoning-suppression techniques. Each works on a different subset of
    # reasoning models — see configs/CONFIG_README.md for the matrix. Combine
    # freely; harmless flags are just ignored by models that don't recognize them.
    reasoning_effort: str | None = None    # sends `reasoning_effort: <value>` in request body (e.g. "none", "low")
    prefill_no_think: bool = False         # appends an assistant message containing `<think>\n</think>\n\n`
    stop: list[str] | None = None          # stop sequences sent to the server; useful for models that parrot the prompt back (Gemma 4)
    use_max_completion_tokens: bool = False  # send `max_completion_tokens` instead of `max_tokens` (required by OpenAI GPT-5 family)
    omit_temperature: bool = False         # skip sending `temperature` (Claude Opus 4.7+ deprecates the field and rejects requests that include it)


def _parse_retry_after(headers: httpx.Headers, attempt: int) -> float:
    """Return seconds to wait before retrying a 429.

    Honors the standard HTTP `Retry-After` header (integer seconds) when
    present. Falls back to exponential backoff anchored at
    DEFAULT_RETRY_AFTER_SECONDS so we don't hammer the endpoint when the
    server didn't tell us how long to wait.
    """
    header_value = headers.get("retry-after")
    if header_value:
        try:
            return max(float(header_value), 1.0)
        except ValueError:
            pass
    return DEFAULT_RETRY_AFTER_SECONDS * (2 ** attempt)


def chat_complete(cfg: ClientConfig, system: str | None, user: str) -> str:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if cfg.prefill_no_think:
        # Pre-filling the assistant turn with an empty think block is the only
        # technique that reliably skips CoT on Qwen3.5/3.6 — the model sees
        # </think> and continues from there with the actual answer.
        messages.append({"role": "assistant", "content": "<think>\n</think>\n\n"})

    payload = {
        "model": cfg.model,
        "messages": messages,
        "stream": False,
    }
    if not cfg.omit_temperature:
        payload["temperature"] = cfg.temperature
    if cfg.use_max_completion_tokens:
        payload["max_completion_tokens"] = cfg.max_tokens
    else:
        payload["max_tokens"] = cfg.max_tokens
    if cfg.reasoning_effort is not None:
        payload["reasoning_effort"] = cfg.reasoning_effort
    if cfg.stop:
        payload["stop"] = cfg.stop

    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"

    attempt = 0
    while True:
        with httpx.Client(timeout=cfg.timeout) as client:
            r = client.post(url, json=payload, headers=headers)
        if r.status_code == 429 and attempt < MAX_RETRIES_ON_429:
            wait_s = _parse_retry_after(r.headers, attempt)
            attempt += 1
            print(
                f"  ⏸ HTTP 429 — sleeping {wait_s:.1f}s before retry {attempt}/{MAX_RETRIES_ON_429}",
                file=sys.stderr,
            )
            time.sleep(wait_s)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        return data["choices"][0]["message"]["content"]
