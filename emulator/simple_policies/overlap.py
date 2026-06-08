from typing import Optional, List
from models import SystemState, Decision
from .base import BasePolicy


def _infer_free(state: SystemState, worst: float) -> float:
    base = state.infer_end_time if state.infer_busy else state.now
    return base + worst * state.committed_infer_nominal


def _finalize(state: SystemState, b_cap: int, worst: float) -> Decision:
    """
    Close timing shared by the overlap policies. Two deadlines:
      overlap:  ideal_close = infer_free - worst*T_prepare(b)   (prepare ends as infer frees)
      SLA:      must_close_by = oldest + SLA - worst*(T_prepare(b) + T_infer(b))
    Take the earlier one; if it already passed, close now.
    """
    p = state.params
    b_used = max(1, min(len(state.queue), b_cap))
    oldest = state.queue[0].arrival_time
    deadline = oldest + state.sla_ms
    busy_until = _infer_free(state, worst)

    ideal_close = busy_until - worst * p.t_prepare_nominal(b_used)
    must_close_by = deadline - worst * (
        p.t_prepare_nominal(b_used) + p.t_infer_nominal(b_used)
    )
    close_at = min(ideal_close, must_close_by)

    if state.now >= close_at:
        return Decision(close_batch_at=state.now, batch_size=b_used)
    return Decision(close_batch_at=close_at, batch_size=None)


class HybridSLAOverlapPolicy(BasePolicy):
    """Max batch within SLA budget + overlap timing. Optional idle collection window."""

    def __init__(self, safety: float = 1.0, max_batch_size: Optional[int] = None,
                 collect_ms: float = 0.0):
        self.safety = safety
        self.max_batch_size = max_batch_size
        self.collect_ms = collect_ms

    def name(self) -> str:
        return f"HybridSLAOverlap(s={self.safety},c={self.collect_ms:.0f})"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        p = state.params
        worst = self.safety * (1.0 + p.variance)
        cap = self._cap(state, self.max_batch_size)

        idle = not state.infer_busy and state.committed_count == 0
        if idle and self.collect_ms > 0.0 and len(state.queue) < cap:
            oldest = state.queue[0].arrival_time
            collect_until = oldest + self.collect_ms
            if state.now < collect_until:
                b_used = max(1, min(len(state.queue), cap))
                must_close_by = (oldest + state.sla_ms) - worst * (
                    p.t_prepare_nominal(b_used) + p.t_infer_nominal(b_used)
                )
                return Decision(close_batch_at=min(collect_until, must_close_by),
                                batch_size=None)

        return _finalize(state, cap, worst)


class PredictiveOverlapPolicy(BasePolicy):
    """Estimate arrival rate λ (EWMA of inter-arrivals), size batch to keep capacity ≥ λ."""

    def __init__(self, alpha: float = 0.3, margin: float = 1.3,
                 safety: float = 1.0, max_batch_size: Optional[int] = None):
        self.alpha = alpha
        self.margin = margin
        self.safety = safety
        self.max_batch_size = max_batch_size
        self._ewma_interval: Optional[float] = None
        self._prev_arrival: Optional[float] = None
        self._last_id: int = -1

    def name(self) -> str:
        return f"PredictiveOverlap(a={self.alpha},m={self.margin},s={self.safety})"

    def _update_rate(self, queue: List) -> float:
        new_reqs = []
        for req in reversed(queue):
            if req.request_id <= self._last_id:
                break
            new_reqs.append(req)
        for req in reversed(new_reqs):
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
            self._last_id = req.request_id

        if self._ewma_interval and self._ewma_interval > 0:
            return 1.0 / self._ewma_interval
        return 0.0

    def _throughput_floor(self, params, lam: float) -> int:
        # need = λ*margin req/ms; with T_infer = a2*b + c2, capacity b/T_infer ≥ need
        # → b ≥ need*c2 / (1 - need*a2)
        need = lam * self.margin
        if need <= 0:
            return 1
        denom = 1.0 - need * params.a2
        if denom <= 0:
            return 10 ** 9
        return max(1, int(need * params.c2 / denom) + 1)

    def decide(self, state: SystemState) -> Decision:
        lam = self._update_rate(state.queue)
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        p = state.params
        worst = self.safety * (1.0 + p.variance)
        cap = self._cap(state, self.max_batch_size)

        if lam <= 0.0:
            b_cap = cap
        else:
            b_cap = max(1, min(cap, self._throughput_floor(p, lam)))

        return _finalize(state, b_cap, worst)


class QueueFeedbackPolicy(BasePolicy):
    """Proportional controller: b = clamp(k*len(queue), b_min, cap)."""

    def __init__(self, k: float = 1.0, b_min: int = 1,
                 safety: float = 1.0, max_batch_size: Optional[int] = None):
        self.k = k
        self.b_min = b_min
        self.safety = safety
        self.max_batch_size = max_batch_size

    def name(self) -> str:
        return f"QueueFeedback(k={self.k},bmin={self.b_min})"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        p = state.params
        worst = self.safety * (1.0 + p.variance)
        cap = self._cap(state, self.max_batch_size)

        b_cap = int(round(self.k * len(state.queue)))
        b_cap = max(self.b_min, b_cap)
        b_cap = max(1, min(cap, b_cap))

        return _finalize(state, b_cap, worst)


class OptimalOverlapPolicy(BasePolicy):
    """
    Theoretical optimum for task 1: throughput(b)=b/T_infer(b) grows with b → always b_max.
    Close at min of two deadlines:
      overlap:  infer_free - safety*T_prepare(b)            (prepare ends as infer frees)
      SLA:      oldest + SLA - worst*(T_prepare(b)+T_infer(b))
    If the system is idle there is no infer to overlap with → use the SLA deadline only.
    """

    def __init__(self, safety: float = 1.0, max_batch_size: Optional[int] = None):
        self.safety = safety
        self.max_batch_size = max_batch_size

    def name(self) -> str:
        return f"Optimal(safety={self.safety})"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        p = state.params
        worst = self.safety * (1.0 + p.variance)
        cap = self._cap(state, self.max_batch_size)
        b = min(len(state.queue), cap)

        infer_free = self._infer_free_at(state)
        overlap_close = infer_free - self.safety * p.t_prepare_nominal(b)

        oldest_arrival = state.queue[0].arrival_time
        worst_processing = worst * (p.t_prepare_nominal(b) + p.t_infer_nominal(b))
        sla_close = oldest_arrival + state.sla_ms - worst_processing

        close_at = min(overlap_close, sla_close)

        system_idle = not state.infer_busy and state.committed_count == 0
        if system_idle:
            close_at = sla_close

        if close_at <= state.now:
            return Decision(close_batch_at=state.now, batch_size=b)
        return Decision(close_batch_at=close_at, batch_size=None)
