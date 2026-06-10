"""Reference batch_size + timeout policy (grid-searched per scenario)."""
from models import Decision
from .base import BasePolicy


class BaselinePolicy(BasePolicy):
    """
    Close when B requests are queued OR T ms passed since the oldest waiter.
    Works in both simple and advanced (black-box) modes.
    """

    def __init__(self, batch_size: int, timeout_ms: float, max_batch_size: int):
        self.batch_size = min(batch_size, max_batch_size)
        self.timeout_ms = timeout_ms
        self.max_batch_size = max_batch_size

    def name(self) -> str:
        return f"BASELINE(B={self.batch_size},T={self.timeout_ms:.1f})"

    def decide(self, state):
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)
        deadline = state.queue[0].arrival_time + self.timeout_ms
        if len(state.queue) >= self.batch_size or state.now >= deadline:
            return Decision(
                close_batch_at=state.now,
                batch_size=min(len(state.queue), self.batch_size),
            )
        return Decision(close_batch_at=deadline, batch_size=None)
