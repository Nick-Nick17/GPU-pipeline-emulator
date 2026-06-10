"""Shared overlap + SLA batch-close timing (used by Hybrid, Optimal, etc.)."""
import math
from itertools import islice
from typing import Optional

from models import SystemState, Decision, PipelineParams

# Optimistic prepare budget when infer is busy and queue is backing up.
OVERLOAD_PREPARE_FACTOR = 0.9


def is_pipeline_overloaded(state: SystemState, cap: int) -> bool:
    """Infer/commit path active and waiting queue is building."""
    if not state.queue:
        return False
    busy = state.infer_busy or state.committed_count > 0
    return busy and len(state.queue) >= max(1, cap // 2)


def effective_prepare_ms(params: PipelineParams, b_prepare: int,
                       overloaded: bool, optimistic: bool) -> float:
    t = params.t_prepare_nominal(b_prepare)
    if optimistic and overloaded:
        return t * OVERLOAD_PREPARE_FACTOR
    return t


def infer_pipeline_batch_size(state: SystemState, cap: int) -> int:
    """Nominal batch size on infer slot / in committed backlog (for slack vs next prepare)."""
    p = state.params
    if state.committed_count > 0:
        avg_infer = state.committed_infer_nominal / state.committed_count
        return max(1, min(cap, round((avg_infer - p.c2) / p.a2)))
    if state.infer_busy and state.infer_end_time is not None:
        remaining = state.infer_end_time - state.now
        if remaining > p.c2:
            return max(1, min(cap, round((remaining - p.c2) / p.a2)))
    return cap


def pipeline_slack(params: PipelineParams, b_infer: int, b_prepare: int,
                   overloaded: bool, optimistic: bool) -> float:
    """Infer time on GPU batch minus (effective) prepare time for the next batch."""
    return (params.t_infer_nominal(b_infer)
            - effective_prepare_ms(params, b_prepare, overloaded, optimistic))


def pipeline_max_committed(params: PipelineParams, b_prepare: int, b_infer: int,
                           overloaded: bool, optimistic: bool) -> int:
    """
    Committed depth from slack between infer-stage and next prepare batch.

    optimistic + overloaded: T_prepare_eff = 0.8 * nominal; else nominal.
    """
    t_prep = effective_prepare_ms(params, b_prepare, overloaded, optimistic)
    t_infer = params.t_infer_nominal(b_infer)
    if t_infer - t_prep >= 0:
        return 1
    return max(1, math.ceil(t_prep / t_infer))


def overlap_close_at(state: SystemState, b_prepare: int, worst: float,
                     cap: int, optimistic: bool = False) -> float:
    """Close so prepare of b_prepare ends when infer frees."""
    overloaded = is_pipeline_overloaded(state, cap)
    prep = effective_prepare_ms(state.params, b_prepare, overloaded, optimistic)
    return infer_free(state, worst) - prep


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
             sla_mode: str = "salvageable",
             overlap_optimistic: bool = False) -> Decision:
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
    ideal_close = overlap_close_at(state, b_used, worst, b_cap, overlap_optimistic)

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
