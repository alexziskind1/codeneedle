"""End-to-end smoke test for hosted-endpoint resilience without a live server.

Uses unittest.mock to simulate HTTP responses and verifies the new
client behaviors introduced by the cloud-retry patch:

  - 429 with Retry-After is retried after sleeping the indicated duration
  - 429 without Retry-After falls back to exponential backoff
  - Repeated 429s eventually surface as RuntimeError (bounded retry)
  - Non-429 4xx/5xx errors are NOT retried — they raise immediately
  - omit_temperature drops the `temperature` field from the request payload
  - Default behavior includes the `temperature` field

Exits non-zero on failure.
"""
from __future__ import annotations

import sys
from unittest import mock

import httpx

from bench.client import ClientConfig, MAX_RETRIES_ON_429, chat_complete


def _ok(content: str = "ok") -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"message": {"content": content}}]}
    )


def _err(status: int, body: str = "x", headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, text=body, headers=headers or {})


def _capture_payloads_and_post(responses: list[httpx.Response]) -> tuple[mock.MagicMock, list[dict]]:
    """Build a mock httpx.Client whose .post() returns the given responses in order
    and captures every JSON payload it was sent. Also no-ops time.sleep so retry
    delays don't slow the test down.
    """
    sent_payloads: list[dict] = []

    def fake_post(url: str, *, json: dict, headers: dict) -> httpx.Response:
        sent_payloads.append(json)
        if not responses:
            raise AssertionError("more requests than mocked responses")
        return responses.pop(0)

    instance = mock.MagicMock()
    instance.post = mock.Mock(side_effect=fake_post)

    client_ctx = mock.MagicMock()
    client_ctx.__enter__ = mock.Mock(return_value=instance)
    client_ctx.__exit__ = mock.Mock(return_value=False)

    return client_ctx, sent_payloads


def _run(cfg: ClientConfig, responses: list[httpx.Response]) -> tuple[str | None, list[dict], list[float], Exception | None]:
    """Run chat_complete with mocked httpx + time.sleep. Returns (result, payloads, sleeps, error)."""
    client_ctx, payloads = _capture_payloads_and_post(responses)
    sleeps: list[float] = []
    err: Exception | None = None
    result: str | None = None
    with mock.patch("bench.client.httpx.Client", return_value=client_ctx), \
         mock.patch("bench.client.time.sleep", side_effect=sleeps.append):
        try:
            result = chat_complete(cfg, system=None, user="hi")
        except Exception as e:  # noqa: BLE001
            err = e
    return result, payloads, sleeps, err


def case_happy_path() -> None:
    cfg = ClientConfig(base_url="http://test", model="m", api_key="k")
    result, payloads, sleeps, err = _run(cfg, [_ok("hello")])
    assert err is None, f"unexpected error: {err}"
    assert result == "hello"
    assert len(payloads) == 1
    assert sleeps == []  # no retry, no sleep
    assert payloads[0]["temperature"] == 0.0
    assert payloads[0]["max_tokens"] == 6000
    print("✓ happy path: single 200, returns content, sends temperature")


def case_retry_after_header() -> None:
    cfg = ClientConfig(base_url="http://test", model="m", api_key="k")
    responses = [
        _err(429, "rate", headers={"retry-after": "3"}),
        _ok("after-retry"),
    ]
    result, payloads, sleeps, err = _run(cfg, responses)
    assert err is None
    assert result == "after-retry"
    assert len(payloads) == 2
    assert sleeps == [3.0], f"expected sleep=[3.0], got {sleeps}"
    print("✓ 429 with Retry-After: sleeps the indicated duration, retries once")


def case_retry_exponential_backoff() -> None:
    """Without Retry-After the client falls back to exponential backoff anchored at 60s."""
    cfg = ClientConfig(base_url="http://test", model="m", api_key="k")
    responses = [
        _err(429, "rate"),  # no header → backoff attempt 0
        _err(429, "rate"),  # no header → backoff attempt 1
        _ok("after-2-retries"),
    ]
    result, payloads, sleeps, err = _run(cfg, responses)
    assert err is None
    assert result == "after-2-retries"
    assert len(payloads) == 3
    assert sleeps == [60.0, 120.0], f"expected [60.0, 120.0], got {sleeps}"
    print("✓ 429 without Retry-After: exponential backoff (60, 120)")


def case_bounded_retry_eventually_fails() -> None:
    """After MAX_RETRIES_ON_429 failed attempts, the error is raised."""
    cfg = ClientConfig(base_url="http://test", model="m", api_key="k")
    # initial + MAX_RETRIES_ON_429 retries = MAX+1 total responses, all 429
    responses = [_err(429, "rate") for _ in range(MAX_RETRIES_ON_429 + 1)]
    result, payloads, sleeps, err = _run(cfg, responses)
    assert isinstance(err, RuntimeError), f"expected RuntimeError, got {type(err)}: {err}"
    assert "429" in str(err)
    assert len(payloads) == MAX_RETRIES_ON_429 + 1
    assert len(sleeps) == MAX_RETRIES_ON_429
    print(f"✓ bounded retry: after {MAX_RETRIES_ON_429} retries, raises RuntimeError")


def case_non_429_not_retried() -> None:
    """500 server error must NOT trigger retry — raise immediately."""
    cfg = ClientConfig(base_url="http://test", model="m", api_key="k")
    result, payloads, sleeps, err = _run(cfg, [_err(500, "boom")])
    assert isinstance(err, RuntimeError)
    assert "500" in str(err)
    assert len(payloads) == 1, "should not retry on 5xx"
    assert sleeps == []
    print("✓ non-429 error (500): raises immediately, no retry")


def case_400_not_retried() -> None:
    cfg = ClientConfig(base_url="http://test", model="m", api_key="k")
    result, payloads, sleeps, err = _run(cfg, [_err(400, "bad request")])
    assert isinstance(err, RuntimeError)
    assert "400" in str(err)
    assert len(payloads) == 1
    print("✓ non-429 error (400): raises immediately, no retry")


def case_omit_temperature_drops_field() -> None:
    """Claude Opus 4.7+ rejects requests with `temperature`; omit_temperature must drop it."""
    cfg = ClientConfig(base_url="http://test", model="m", api_key="k", omit_temperature=True)
    result, payloads, sleeps, err = _run(cfg, [_ok("hi")])
    assert err is None
    assert "temperature" not in payloads[0], (
        f"temperature should be omitted, payload keys: {list(payloads[0].keys())}"
    )
    print("✓ omit_temperature=true: temperature absent from payload")


def case_default_includes_temperature() -> None:
    """Sanity: omit_temperature default (false) keeps the field."""
    cfg = ClientConfig(base_url="http://test", model="m", api_key="k")
    assert cfg.omit_temperature is False, "default should be False"
    result, payloads, _, err = _run(cfg, [_ok("hi")])
    assert err is None
    assert "temperature" in payloads[0]
    print("✓ default (omit_temperature=false): temperature present in payload")


def main() -> int:
    case_happy_path()
    case_retry_after_header()
    case_retry_exponential_backoff()
    case_bounded_retry_eventually_fails()
    case_non_429_not_retried()
    case_400_not_retried()
    case_omit_temperature_drops_field()
    case_default_includes_temperature()
    print("\n✅ all client smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
