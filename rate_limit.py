"""
In-memory per-client rate limiting.
"""

import time
from collections import defaultdict

from fastapi import HTTPException


class RateLimiter:
    """
    Fixed-window rate limiter keyed by client identifier (e.g. IP address).

    Single-process, in-memory only — fine for a small personal deployment.
    A multi-worker or multi-instance deployment would need a shared store
    (e.g. Redis) instead, since each process would otherwise track its own
    independent counts.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._log: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str, now: float | None = None) -> None:
        """Raise HTTPException(429) if client_id has exceeded the limit."""
        now = now if now is not None else time.time()
        window_start = now - self.window_seconds

        timestamps = self._log[client_id]
        while timestamps and timestamps[0] < window_start:
            timestamps.pop(0)

        if len(timestamps) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

        timestamps.append(now)
