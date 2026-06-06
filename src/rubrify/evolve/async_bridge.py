"""Async-to-sync bridge utility for the evolve package.

GEPA's optimization loop is synchronous, but rubrify's Judge and harn_ai's
complete_simple are async.  This module provides a single helper that safely
runs an async coroutine from synchronous code -- even when an event loop is
already running (e.g. inside Jupyter or an async test harness).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TypeVar

T = TypeVar("T")


def run_async(coro: object) -> T:  # noqa: ANN401
    """Run an async coroutine from sync context, handling nested event loops.

    If no event loop is running, simply uses ``asyncio.run``.
    If an event loop *is* already running (nested context), dispatches the
    coroutine to a short-lived thread so a fresh loop can be created without
    colliding with the outer one.

    Parameters
    ----------
    coro:
        An awaitable coroutine object (the result of calling an async function).

    Returns
    -------
    T
        Whatever the coroutine returns.
    """
    try:
        asyncio.get_running_loop()
        # Already inside an async context -- run in a worker thread to
        # avoid nested event-loop issues.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop -- safe to create one directly.
        return asyncio.run(coro)


__all__ = ["run_async"]
