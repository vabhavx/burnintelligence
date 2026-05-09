"""
Exponential-backoff retry helper for ingestor network calls.

Stdlib + aiohttp only — no tenacity. Each retry is logged with attempt
number and computed delay so transient failures are visible in pipeline logs.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable, Tuple, Type

import aiohttp

logger = logging.getLogger("intelligence.ingestors.retry")


async def with_retry(
    coro_factory: Callable[[], Awaitable],
    *,
    retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    retry_on: Tuple[Type[BaseException], ...] = (aiohttp.ClientError, asyncio.TimeoutError),
    name: str = "operation",
):
    """Run coro_factory() with exponential backoff on transient errors.

    coro_factory must be a zero-arg callable returning a fresh coroutine on
    every call (a coroutine object can only be awaited once).

    Returns a tuple: (result, attempts_used, retries_used, last_error_str).
    On final failure, result is None and last_error_str carries the message.
    """
    attempt = 0
    last_error: BaseException | None = None

    while attempt <= retries:
        attempt += 1
        try:
            result = await coro_factory()
            return result, attempt, attempt - 1, None
        except retry_on as e:
            last_error = e
            if attempt > retries:
                logger.error(
                    f"{name}: giving up after {attempt} attempts ({type(e).__name__}: {e})"
                )
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.1)
            logger.warning(
                f"{name}: attempt {attempt}/{retries + 1} failed "
                f"({type(e).__name__}: {e}); retrying in {delay:.2f}s"
            )
            await asyncio.sleep(delay)

    err_str = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown"
    return None, attempt, max(0, attempt - 1), err_str
