from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, List, Tuple, Deque

from models import AdvancedState, BatchObservation, Decision


class BaseAdvancedPolicy(ABC):
    @abstractmethod
    def decide(self, state: AdvancedState) -> Decision:
        ...

    def name(self) -> str:
        return self.__class__.__name__


class TotalTimeModel:
    """
    Online least-squares fit of the black-box round-trip time: T_total(b) ≈ A*b + C.
    Sliding window of recent (size, total_latency) with running sums, O(1) per update.
    """

    def __init__(self, window: int = 200):
        self.window = window
        self._obs: Deque[Tuple[float, float]] = deque()
        self._last_id: int = -1
        self._sb = self._st = self._sbb = self._sbt = 0.0

    def _add(self, b: float, t: float) -> None:
        self._obs.append((b, t))
        self._sb += b
        self._st += t
        self._sbb += b * b
        self._sbt += b * t
        if len(self._obs) > self.window:
            ob, ot = self._obs.popleft()
            self._sb -= ob
            self._st -= ot
            self._sbb -= ob * ob
            self._sbt -= ob * ot

    def ingest(self, observations: List[BatchObservation]) -> None:
        for o in observations:
            if o.batch_id > self._last_id:
                self._add(o.size, o.total_latency)
                self._last_id = o.batch_id

    @property
    def n(self) -> int:
        return len(self._obs)

    def predict(self) -> Optional[Tuple[float, float]]:
        """(A, C) for T_total ≈ A*b + C, or None if no data yet."""
        n = len(self._obs)
        if n == 0:
            return None
        if n == 1:
            return (0.0, self._st)
        # least squares: A = (n*Σbt - Σb*Σt) / (n*Σb² - (Σb)²),  C = (Σt - A*Σb)/n
        denom = n * self._sbb - self._sb * self._sb
        if denom <= 1e-9:
            return (0.0, self._st / n)
        a = (n * self._sbt - self._sb * self._st) / denom
        c = (self._st - a * self._sb) / n
        if a < 0.0:
            return (0.0, self._st / n)
        return (a, c)


def _max_b_for_budget(a: float, c: float, budget: float, safety: float,
                      hard_cap: int) -> int:
    """Largest b with safety*(a*b + c) <= budget  →  b = (budget/safety - c)/a."""
    if budget <= 0:
        return 1
    if a <= 1e-9:
        return hard_cap if safety * c <= budget else 1
    b = (budget / safety - c) / a
    if b < 1:
        return 1
    return min(hard_cap, int(b))
