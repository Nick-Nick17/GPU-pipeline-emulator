"""Shared overlap + SLA batch-close timing (used by Hybrid, Optimal, etc.)."""
from itertools import islice
from typing import Optional

from models import SystemState, Decision


def infer_free(state: SystemState, worst: float) -> float:
    base = state.infer_end_time if state.infer_busy else state.now
    return base + worst * state.committed_infer_nominal


def worst_processing(p, b: int, worst: float) -> float:
    return worst * (p.t_prepare_nominal(b) + p.t_infer_nominal(b))


def must_close_by_oldest(state: SystemState, b: int, worst: float) -> float:
    proc = worst_processing(state.params, b, worst)
    return state.queue[0].arrival_time + state.sla_ms - proc


def must_close_by_salvageable(state: SystemState, b: int, worst: float) -> Optional[float]:
    p = state.params
    proc = worst_processing(p, b, worst)
    threshold = state.now + proc - state.sla_ms
    window = min(b, len(state.queue))
    if window == 0:
        return None

    oldest = state.queue[0]
    if oldest.arrival_time > threshold:
        return oldest.arrival_time + state.sla_ms - proc

    for req in islice(state.queue, 1, window):
        if req.arrival_time > threshold:
            return req.arrival_time + state.sla_ms - proc
    return None


def finalize(state: SystemState, b_cap: int, worst: float,
             drain_cap: Optional[int] = None,
             early_drain: bool = True,
             sla_mode: str = "salvageable") -> Decision:
    """
    Two deadlines: overlap (ideal_close) and SLA (must_close_by).
    early_drain: close immediately when len(queue) >= drain threshold.
    """
    p = state.params
    if early_drain:
        drain = drain_cap if drain_cap is not None else b_cap
        if len(state.queue) >= drain:
            return Decision(close_batch_at=state.now,
                            batch_size=min(len(state.queue), drain))

    b_used = max(1, min(len(state.queue), b_cap))
    busy_until = infer_free(state, worst)
    ideal_close = busy_until - worst * p.t_prepare_nominal(b_used)

    if sla_mode == "oldest":
        must_close_by = must_close_by_oldest(state, b_used, worst)
    else:
        must_close_by = must_close_by_salvageable(state, b_used, worst)
        if must_close_by is None:
            return Decision(close_batch_at=state.now, batch_size=b_used)

    close_at = min(ideal_close, must_close_by)
    if state.now >= close_at:
        return Decision(close_batch_at=state.now, batch_size=b_used)
    return Decision(close_batch_at=close_at, batch_size=None)
