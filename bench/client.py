"""Minimal OpenAI-compatible chat-completions client."""
from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass
class ClientConfig:
    base_url: str
    model: str
    api_key: str = "not-needed"
    temperature: float = 0.0
    max_tokens: int = 6000  # generous — reasoning models (qwen3.6, o1-style) can't reliably be told to skip CoT, so the budget must cover reasoning + ~20 answer lines
    timeout: float = 600.0


def chat_complete(cfg: ClientConfig, system: str | None, user: str) -> str:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
    with httpx.Client(timeout=cfg.timeout) as client:
        r = client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
    return data["choices"][0]["message"]["content"]
