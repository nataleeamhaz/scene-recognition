import pytest
from fastapi import HTTPException

from rate_limit import RateLimiter


def test_allows_up_to_max_requests():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.check("1.2.3.4", now=1000.0)  # same instant, still within limit


def test_blocks_after_max_requests():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.check("1.2.3.4", now=1000.0)
    with pytest.raises(HTTPException) as exc_info:
        limiter.check("1.2.3.4", now=1000.0)
    assert exc_info.value.status_code == 429


def test_resets_after_window_expires():
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.check("1.2.3.4", now=1000.0)
    limiter.check("1.2.3.4", now=1000.0)
    with pytest.raises(HTTPException):
        limiter.check("1.2.3.4", now=1000.0)

    # past the window — old requests should have aged out
    limiter.check("1.2.3.4", now=1061.0)


def test_tracks_clients_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    limiter.check("1.2.3.4", now=1000.0)
    limiter.check("5.6.7.8", now=1000.0)  # different client, should not raise

    with pytest.raises(HTTPException):
        limiter.check("1.2.3.4", now=1000.0)
