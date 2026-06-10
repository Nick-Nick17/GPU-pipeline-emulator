from typing import Optional

from models import SystemState, Decision
from .base import BasePolicy
from .overlap_core import (
    finalize,
    infer_free,
    infer_pipeline_batch_size,
    is_pipeline_overloaded,
    must_close_by_oldest,
    must_close_by_salvageable,
    pipeline_max_committed,
)


def _worst(safety: float, state: SystemState) -> float:
    return safety * (1.0 + state.params.variance)


def _with_load_control(d: Decision, worst: float, cap: int,
                       shed_hopeless: bool, admit_infer: bool,
                       max_committed: Optional[int]) -> Decision:
    if not shed_hopeless and not admit_infer and max_committed is None:
        return d
    return Decision(
        close_batch_at=d.close_batch_at,
        batch_size=d.batch_size,
        shed_hopeless=shed_hopeless or admit_infer,
        admit_infer=admit_infer,
        max_committed=max_committed,
        shed_worst=worst,
        shed_b=cap,
    )


def _collect_while_idle(state: SystemState, cap: int, worst: float,
                        collect_ms: float, sla_mode: str) -> Optional[Decision]:
    idle = not state.infer_busy and state.committed_count == 0
    if not (idle and collect_ms > 0.0 and len(state.queue) < cap):
        return None

    oldest = state.queue[0].arrival_time
    collect_until = oldest + collect_ms
    if state.now >= collect_until:
        return None

    b_used = max(1, min(len(state.queue), cap))
    if sla_mode == "oldest":
        must_close_by = must_close_by_oldest(state, b_used, worst)
        return Decision(close_batch_at=min(collect_until, must_close_by), batch_size=None)

    must_close_by = must_close_by_salvageable(state, b_used, worst)
    if must_close_by is None:
        return Decision(close_batch_at=state.now,
                        batch_size=min(len(state.queue), cap))
    return Decision(close_batch_at=min(collect_until, must_close_by), batch_size=None)


class HybridSLAOverlapPolicy(BasePolicy):
    def __init__(self, safety: float = 1.0, max_batch_size: Optional[int] = None,
                 collect_ms: float = 0.0, early_drain: bool = True,
                 shed_hopeless: bool = False, admit_infer: bool = False,
                 max_committed_batches: Optional[int] = None,
                 max_committed_auto: bool = False,
                 overlap_optimistic: bool = False):
        self.safety = safety
        self.max_batch_size = max_batch_size
        self.collect_ms = collect_ms
        self.early_drain = early_drain
        self.shed_hopeless = shed_hopeless
        self.admit_infer = admit_infer
        self.max_committed_auto = max_committed_auto
        self.overlap_optimistic = overlap_optimistic
        if max_committed_auto:
            self.max_committed_batches = None
        elif max_committed_batches is not None:
            self.max_committed_batches = max_committed_batches
        elif admit_infer:
            self.max_committed_batches = 1
        else:
            self.max_committed_batches = None

    def name(self) -> str:
        tags = []
        if self.shed_hopeless:
            tags.append("shed")
        if self.admit_infer:
            tags.append("admit")
        if self.max_committed_auto:
            tags.append("mco" if self.overlap_optimistic else "mcn")
        elif self.max_committed_batches is not None:
            tags.append(f"mc{self.max_committed_batches}")
        tag = f"+{'+'.join(tags)}" if tags else ""
        return f"HybridSLAOverlap(s={self.safety},c={self.collect_ms:.0f}){tag}"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        worst = _worst(self.safety, state)
        cap = self._cap(state, self.max_batch_size)

        d = _collect_while_idle(state, cap, worst, self.collect_ms, "salvageable")
        if d is None:
            d = finalize(state, cap, worst, early_drain=self.early_drain,
                         overlap_optimistic=self.overlap_optimistic)

        mc = self.max_committed_batches
        if self.max_committed_auto:
            b_prep = max(1, d.batch_size or min(len(state.queue), cap))
            b_inf = infer_pipeline_batch_size(state, cap)
            overloaded = is_pipeline_overloaded(state, cap)
            mc = pipeline_max_committed(
                state.params, b_prep, b_inf, overloaded, self.overlap_optimistic,
            )

        return _with_load_control(
            d, worst, cap, self.shed_hopeless, self.admit_infer, mc,
        )


class HybridSLAOverlapLegacyPolicy(BasePolicy):
    def __init__(self, safety: float = 1.0, max_batch_size: Optional[int] = None,
                 collect_ms: float = 0.0, early_drain: bool = True):
        self.safety = safety
        self.max_batch_size = max_batch_size
        self.collect_ms = collect_ms
        self.early_drain = early_drain

    def name(self) -> str:
        return f"HybridSLAOverlap-oldest(s={self.safety},c={self.collect_ms:.0f})"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        worst = _worst(self.safety, state)
        cap = self._cap(state, self.max_batch_size)

        d = _collect_while_idle(state, cap, worst, self.collect_ms, "oldest")
        if d is None:
            d = finalize(state, cap, worst, sla_mode="oldest",
                         early_drain=self.early_drain)
        return d


class PredictiveOverlapPolicy(BasePolicy):
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

    def _update_rate(self, queue) -> float:
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

        worst = _worst(self.safety, state)
        cap = self._cap(state, self.max_batch_size)
        b_cap = cap if lam <= 0.0 else max(1, min(cap, self._throughput_floor(state.params, lam)))
        return finalize(state, b_cap, worst)


class QueueFeedbackPolicy(BasePolicy):
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

        worst = _worst(self.safety, state)
        cap = self._cap(state, self.max_batch_size)
        b_cap = max(self.b_min, int(round(self.k * len(state.queue))))
        b_cap = max(1, min(cap, b_cap))
        return finalize(state, b_cap, worst, drain_cap=cap)


class OptimalOverlapPolicy(BasePolicy):
    def __init__(self, safety: float = 1.0, max_batch_size: Optional[int] = None):
        self.safety = safety
        self.max_batch_size = max_batch_size

    def name(self) -> str:
        return f"Optimal(safety={self.safety})"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        p = state.params
        worst = _worst(self.safety, state)
        cap = self._cap(state, self.max_batch_size)
        if len(state.queue) >= cap:
            return Decision(close_batch_at=state.now, batch_size=cap)

        b = min(len(state.queue), cap)
        overlap_close = infer_free(state, worst) - self.safety * p.t_prepare_nominal(b)
        sla_close = must_close_by_salvageable(state, b, worst)
        if sla_close is None:
            return Decision(close_batch_at=state.now, batch_size=b)

        close_at = sla_close if (not state.infer_busy and state.committed_count == 0) \
            else min(overlap_close, sla_close)

        if close_at <= state.now:
            return Decision(close_batch_at=state.now, batch_size=b)
        return Decision(close_batch_at=close_at, batch_size=None)
