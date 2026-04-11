"""
Dynamic batching for Phase 3 inference serving.

DynamicBatcher collects individual inference requests into batches and
dispatches them to a background worker.  A batch is released when either:

  - the pending count reaches max_batch_size, OR
  - max_wait_s seconds have elapsed since the first request in the batch.

This is a threading-based implementation suitable for the demo.

TODO: For production use, convert to asyncio.Queue so the FastAPI event
loop is not blocked by Future.result() calls.
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class InferenceRequest:
    """A single inference request submitted to the batcher."""
    image_b64: str
    prompt: str
    max_new_tokens: int
    # Timestamp set on construction — used to enforce max_wait_s deadline.
    arrived_at: float = field(default_factory=time.monotonic)
    # Set by DynamicBatcher.submit() before the request enters the queue.
    result_future: Optional["Future"] = field(default=None, init=False)


class DynamicBatcher:
    """
    Collects InferenceRequests into batches and dispatches them to a worker.

    Parameters
    ----------
    process_batch  : Callable[[List[InferenceRequest]], List[str]]
        Takes a batch of requests; returns response strings in the same order.
    max_batch_size : int
        Flush the batch once this many requests are pending.
    max_wait_s     : float
        Flush the batch after this many seconds even if not full.
    """

    def __init__(
        self,
        process_batch: Callable[[List[InferenceRequest]], List[str]],
        max_batch_size: int = 8,
        max_wait_s: float = 0.1,
    ) -> None:
        self._process_batch = process_batch
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_s
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="batcher")
        self._worker.start()

    def submit(self, request: InferenceRequest) -> "Future":
        """
        Submit a request for batched inference.

        Returns a Future whose .result() blocks until the response is ready.
        The caller is responsible for setting a sensible timeout on .result().
        """
        future: Future = Future()
        request.result_future = future
        self._queue.put(request)
        return future

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background loop: collect requests into batches and dispatch them."""
        while True:
            batch: List[InferenceRequest] = []

            # Block until the first request of a new batch arrives
            try:
                first = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            batch.append(first)
            deadline = first.arrived_at + self._max_wait_s

            # Fill the batch up to max_batch_size within the deadline
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = self._queue.get(timeout=remaining)
                    batch.append(req)
                except queue.Empty:
                    break

            self._dispatch(batch)

    def _dispatch(self, batch: List[InferenceRequest]) -> None:
        """Call process_batch and resolve each request's Future."""
        try:
            responses = self._process_batch(batch)
            for req, resp in zip(batch, responses):
                req.result_future.set_result(resp)
        except Exception as exc:
            for req in batch:
                req.result_future.set_exception(exc)


class Future:
    """
    A minimal thread-safe result container.

    Not an asyncio.Future — this is synchronous and meant to be used from
    regular threads.  A single thread calls set_result() or set_exception();
    another thread blocks on result().
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: Optional[Any] = None
        self._exception: Optional[Exception] = None

    def set_result(self, value: Any) -> None:
        self._result = value
        self._event.set()

    def set_exception(self, exc: Exception) -> None:
        self._exception = exc
        self._event.set()

    def result(self, timeout: float = 60.0) -> Any:
        """Block until the result is available, then return it or raise."""
        if not self._event.wait(timeout=timeout):
            raise TimeoutError(
                f"inference request did not complete within {timeout}s"
            )
        if self._exception is not None:
            raise self._exception
        return self._result

    @property
    def done(self) -> bool:
        return self._event.is_set()
