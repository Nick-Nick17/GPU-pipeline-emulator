"""
Task 2 policies — the "black box" pipeline.

Here we are NOT allowed to know the prepare/infer split, whether the infer
slot is busy, or any nominal timing. The only signal is:

    for every batch we sent, when it entered and when it came back.

So a batch of size b is a black box that returns after some total time
T_total(b). Policies below LEARN that function online from observations and
make all decisions from it + the SLA + the visible queue.

Overlap trick without knowing the split:
    In the real system prepare can overlap with the previous infer. A black-box
    policy reproduces that by keeping a small number of batches "in flight"
    (max_in_flight). With max_in_flight=1 the slot idles during every prepare;
    with max_in_flight=2 there is always one batch preparing while another is
    being processed, so the pipeline stays busy — same effect as overlap, but
    discovered purely from entry/exit timing.
"""

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
    Online linear fit of the black-box round-trip time: T_total(b) ~= A*b + C.

    Keeps a sliding window of recent (size, total_latency) observations and the
    running sums needed for least squares, so each update is O(1).
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
        """Return (A, C) for T_total ~= A*b + C, or None if no data yet."""
        n = len(self._obs)
        if n == 0:
            return None
        if n == 1:
            return (0.0, self._st)
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
    """Largest b with safety*(a*b + c) <= budget."""
    if budget <= 0:
        return 1
    if a <= 1e-9:
        return hard_cap if safety * c <= budget else 1
    b = (budget / safety - c) / a
    if b < 1:
        return 1
    return min(hard_cap, int(b))


class BlackBoxSLAOverlapPolicy(BaseAdvancedPolicy):
    """
    Flagship task-2 policy.

    Learns T_total(b) from observed round trips, then:
      - picks the largest batch that still fits the SLA budget (throughput),
      - keeps up to max_in_flight batches in flight to mimic overlap,
      - when the pipeline is empty, may collect for collect_ms before sealing.
    """

    def __init__(self, safety: float = 1.2, max_in_flight: int = 2,
                 collect_ms: float = 0.0, window: int = 200,
                 bootstrap_max: int = 16, min_samples: int = 5,
                 hard_cap: int = 256):
        self.safety = safety
        self.max_in_flight = max_in_flight
        self.collect_ms = collect_ms
        self.bootstrap_max = bootstrap_max
        self.min_samples = min_samples
        self.hard_cap = hard_cap
        self.model = TotalTimeModel(window)

    def name(self) -> str:
        return f"BB-Overlap(s={self.safety},mif={self.max_in_flight},c={self.collect_ms:.0f})"

    def decide(self, state: AdvancedState) -> Decision:
        self.model.ingest(state.observations)
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        est = self.model.predict()
        ready = est is not None and self.model.n >= self.min_samples
        oldest = state.queue[0].arrival_time

        # Size for throughput: the largest batch where a *fresh* request still
        # fits the SLA. We deliberately do NOT shrink by the oldest request's
        # remaining budget — under a backlog that would collapse to size 1 and
        # kill throughput (death spiral). The oldest only affects collect timing.
        if est is None:
            b_cap = self.bootstrap_max
            b_target = min(len(state.queue), self.bootstrap_max)
        else:
            a, c = est
            b_cap = _max_b_for_budget(a, c, state.sla_ms, self.safety, self.hard_cap)
            b_target = min(len(state.queue), b_cap)
            if not ready:
                b_target = min(b_target, self.bootstrap_max)
        b_target = max(1, b_target)

        # Pacing: never exceed max_in_flight. If saturated, hold — we will be
        # re-consulted when a batch returns or a new request arrives.
        if state.in_flight >= self.max_in_flight:
            return Decision(close_batch_at=None, batch_size=None)

        # Pipeline empty and queue not full yet: optionally collect a bit more,
        # but never past the point where the oldest request would miss the SLA.
        if (state.in_flight == 0 and self.collect_ms > 0.0
                and len(state.queue) < b_cap):
            collect_until = oldest + self.collect_ms
            if est is not None:
                a, c = est
                must_close_by = oldest + state.sla_ms - self.safety * (a * b_target + c)
                wait_until = min(collect_until, must_close_by)
            else:
                wait_until = collect_until
            if state.now < wait_until:
                return Decision(close_batch_at=wait_until, batch_size=None)

        return Decision(close_batch_at=state.now, batch_size=b_target)


class LatencyFeedbackPolicy(BaseAdvancedPolicy):
    """
    Pure reactive controller, no timing model at all.

    Watches the observed round-trip latency of finished batches and nudges a
    target batch size (AIMD style): grow when we are comfortably under SLA,
    shrink hard when we get close to it.
    """

    def __init__(self, max_in_flight: int = 2, low: float = 0.55,
                 high: float = 0.85, inc: float = 2.0, dec: float = 0.7,
                 max_wait_frac: float = 0.4, init_b: float = 8.0,
                 hard_cap: int = 256):
        self.max_in_flight = max_in_flight
        self.low = low
        self.high = high
        self.inc = inc
        self.dec = dec
        self.max_wait_frac = max_wait_frac
        self.hard_cap = hard_cap
        self.b_target = init_b
        self._last_id = -1

    def name(self) -> str:
        return f"BB-Feedback(lo={self.low},hi={self.high})"

    def _ingest(self, state: AdvancedState) -> None:
        for o in state.observations:
            if o.batch_id <= self._last_id:
                continue
            self._last_id = o.batch_id
            if o.total_latency > state.sla_ms * self.high:
                self.b_target = max(1.0, self.b_target * self.dec)
            elif o.total_latency < state.sla_ms * self.low:
                self.b_target = min(self.hard_cap, self.b_target + self.inc)

    def decide(self, state: AdvancedState) -> Decision:
        self._ingest(state)
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)
        if state.in_flight >= self.max_in_flight:
            return Decision(close_batch_at=None, batch_size=None)

        oldest = state.queue[0].arrival_time
        waited = state.now - oldest
        target = max(1, int(round(self.b_target)))
        deadline = oldest + state.sla_ms * self.max_wait_frac

        if len(state.queue) >= target or waited >= state.sla_ms * self.max_wait_frac:
            return Decision(close_batch_at=state.now,
                            batch_size=min(len(state.queue), target))
        return Decision(close_batch_at=deadline, batch_size=None)


class ThroughputMatchPolicy(BaseAdvancedPolicy):
    """
    Predictive black-box policy.

    Estimates arrival rate (from when requests show up) and the batch departure
    interval (from when batches come back), then sizes each batch so departures
    keep up with arrivals: b ~= lambda * departure_interval * margin.
    A learned T_total(b) caps the size so the SLA still holds.
    """

    def __init__(self, alpha: float = 0.2, margin: float = 1.15,
                 safety: float = 1.2, max_in_flight: int = 2,
                 window: int = 200, min_samples: int = 5, hard_cap: int = 256):
        self.alpha = alpha
        self.margin = margin
        self.safety = safety
        self.max_in_flight = max_in_flight
        self.min_samples = min_samples
        self.hard_cap = hard_cap
        self.model = TotalTimeModel(window)

        self._ewma_interval: Optional[float] = None   # ms between arrivals
        self._prev_arrival: Optional[float] = None
        self._last_arr_id: int = -1

        self._ewma_depart: Optional[float] = None      # ms between batch returns
        self._prev_return: Optional[float] = None
        self._last_obs_id: int = -1

    def name(self) -> str:
        return f"BB-Throughput(a={self.alpha},m={self.margin})"

    def _update_arrivals(self, queue) -> None:
        new = []
        for req in reversed(queue):
            if req.request_id <= self._last_arr_id:
                break
            new.append(req)
        for req in reversed(new):
            if self._prev_arrival is not None:
                interval = req.arrival_time - self._prev_arrival
                if interval > 0:
                    if self._ewma_interval is None:
                        self._ewma_interval = interval
                    else:
                        self._ewma_interval = (
                            self.alpha * interval
                            + (1.0 - self.alpha) * self._ewma_interval
                        )
            self._prev_arrival = req.arrival_time
            self._last_arr_id = req.request_id

    def _update_departures(self, observations) -> None:
        for o in observations:
            if o.batch_id <= self._last_obs_id:
                continue
            self._last_obs_id = o.batch_id
            self.model._add(o.size, o.total_latency)
            self.model._last_id = o.batch_id
            if self._prev_return is not None:
                gap = o.return_time - self._prev_return
                if gap > 0:
                    if self._ewma_depart is None:
                        self._ewma_depart = gap
                    else:
                        self._ewma_depart = (
                            self.alpha * gap
                            + (1.0 - self.alpha) * self._ewma_depart
                        )
            self._prev_return = o.return_time

    def decide(self, state: AdvancedState) -> Decision:
        self._update_arrivals(state.queue)
        self._update_departures(state.observations)
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)
        if state.in_flight >= self.max_in_flight:
            return Decision(close_batch_at=None, batch_size=None)

        est = self.model.predict()

        if est is None or self.model.n < self.min_samples:
            b_target = min(len(state.queue), 16)
        else:
            a, c = est
            b_sla = _max_b_for_budget(a, c, state.sla_ms, self.safety, self.hard_cap)
            if (self._ewma_interval and self._ewma_interval > 0
                    and self._ewma_depart and self._ewma_depart > 0):
                lam = 1.0 / self._ewma_interval                # req per ms
                need = lam * self._ewma_depart * self.margin   # req per departure
                b_rate = max(1, int(need) + 1)
            else:
                b_rate = b_sla
            b_target = min(len(state.queue), b_sla, b_rate)
        b_target = max(1, b_target)

        return Decision(close_batch_at=state.now, batch_size=b_target)
