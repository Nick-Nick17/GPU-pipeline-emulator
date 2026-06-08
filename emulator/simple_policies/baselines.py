from typing import Optional
from models import SystemState, Decision
from .base import BasePolicy


class TimeoutBatchPolicy(BasePolicy):
    def __init__(self, timeout_ms: float, max_batch_size: Optional[int] = None):
        self.timeout_ms = timeout_ms
        self.max_batch_size = max_batch_size

    def name(self) -> str:
        return f"Timeout(t={self.timeout_ms:.0f})"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        cap = self._cap(state, self.max_batch_size)
        deadline = state.queue[0].arrival_time + self.timeout_ms

        if len(state.queue) >= cap or state.now >= deadline:
            return Decision(close_batch_at=state.now, batch_size=min(len(state.queue), cap))

        return Decision(close_batch_at=deadline, batch_size=None)


class FixedSizePolicy(BasePolicy):
    def __init__(self, target_size: int, max_wait_ms: float = 1e9,
                 max_batch_size: Optional[int] = None):
        self.target_size = target_size
        self.max_wait_ms = max_wait_ms
        self.max_batch_size = max_batch_size

    def name(self) -> str:
        return f"FixedSize(b={self.target_size})"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        cap = self._cap(state, self.max_batch_size)
        target = min(self.target_size, cap)
        deadline = state.queue[0].arrival_time + self.max_wait_ms

        if len(state.queue) >= target or state.now >= deadline:
            return Decision(close_batch_at=state.now, batch_size=min(len(state.queue), cap))

        return Decision(close_batch_at=deadline, batch_size=None)


class DeadlineOverlapPolicy(BasePolicy):
    def __init__(self, alpha: float = 1.2, max_batch_size: Optional[int] = None,
                 collect_ms: float = 0.0):
        self.alpha = alpha
        self.max_batch_size = max_batch_size
        self.collect_ms = collect_ms

    def name(self) -> str:
        return f"Overlap(a={self.alpha}, c={self.collect_ms:.0f})"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        params = state.params
        cap = self._cap(state, self.max_batch_size)
        b_est = min(len(state.queue), cap)

        # start prepare so it ends right when infer frees up: t_start = busy_until - alpha*T_prepare(b)
        busy_until = self._infer_free_at(state)
        ideal_start = busy_until - self.alpha * params.t_prepare_nominal(b_est)

        if state.now < ideal_start:
            return Decision(close_batch_at=ideal_start, batch_size=None)

        idle = not state.infer_busy and state.committed_count == 0
        if idle and len(state.queue) < cap:
            oldest = state.queue[0].arrival_time
            if state.now - oldest < self.collect_ms:
                return Decision(close_batch_at=oldest + self.collect_ms, batch_size=None)

        return Decision(close_batch_at=state.now, batch_size=min(len(state.queue), cap))


class SLABudgetPolicy(BasePolicy):
    def __init__(self, safety: float = 1.0, max_batch_size: Optional[int] = None):
        self.safety = safety
        self.max_batch_size = max_batch_size

    def name(self) -> str:
        return f"SLABudget(s={self.safety})"

    def decide(self, state: SystemState) -> Decision:
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)

        params = state.params
        cap = self._cap(state, self.max_batch_size)
        if len(state.queue) >= cap:
            return Decision(close_batch_at=state.now, batch_size=cap)

        worst = self.safety * (1.0 + params.variance)
        b_est = min(len(state.queue), cap)
        tp = worst * params.t_prepare_nominal(b_est)
        ti = worst * params.t_infer_nominal(b_est)

        deadline = state.queue[0].arrival_time + state.sla_ms
        busy_until = self._infer_free_at(state, worst)
        must_start_by = deadline - ti

        if busy_until >= must_start_by:
            if len(state.queue) >= cap:
                return Decision(close_batch_at=state.now, batch_size=cap)
            return Decision(close_batch_at=None, batch_size=None)

        latest_close = must_start_by - tp
        if state.now >= latest_close:
            return Decision(close_batch_at=state.now, batch_size=min(len(state.queue), cap))

        return Decision(close_batch_at=latest_close, batch_size=None)
