"""
Task 2 policies — the "black box" pipeline.

The prepare/infer split, infer_busy and nominal timings are NOT visible. The only
signal: for every batch we sent, when it entered and when it came back. So a batch
of size b is a black box returning after T_total(b); policies learn that online.

Overlap without knowing the split: keep up to max_in_flight batches in flight.
  max_in_flight=1 → the slot idles during every prepare.
  max_in_flight=2 → one batch prepares while another runs → pipeline stays busy.
"""

from typing import Optional

from models import AdvancedState, Decision
from .model import BaseAdvancedPolicy, TotalTimeModel, _max_b_for_budget


class BlackBoxSLAOverlapPolicy(BaseAdvancedPolicy):
    """Learn T_total(b), pick the largest batch that fits SLA, keep max_in_flight in flight."""

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

        # Size by a *fresh* request's SLA budget, not the oldest's remaining: under a
        # backlog the latter collapses to 1 and kills throughput (death spiral).
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

        if state.in_flight >= self.max_in_flight:
            return Decision(close_batch_at=None, batch_size=None)

        if (state.in_flight == 0 and self.collect_ms > 0.0
                and len(state.queue) < b_cap):
            collect_until = oldest + self.collect_ms
            if est is not None:
                a, c = est
                # don't wait past oldest's safe deadline: oldest + SLA - safety*(A*b + C)
                must_close_by = oldest + state.sla_ms - self.safety * (a * b_target + c)
                wait_until = min(collect_until, must_close_by)
            else:
                wait_until = collect_until
            if state.now < wait_until:
                return Decision(close_batch_at=wait_until, batch_size=None)

        return Decision(close_batch_at=state.now, batch_size=b_target)


class LatencyFeedbackPolicy(BaseAdvancedPolicy):
    """AIMD controller, no model: grow target batch when far under SLA, shrink hard when close."""

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
    Match departures to arrivals: b ≈ λ * departure_interval * margin, capped by SLA.
    λ from EWMA of inter-arrivals, departure_interval from EWMA of gaps between returns.
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
