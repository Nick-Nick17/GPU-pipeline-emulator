"""Time-based prepare pipeline load (remaining prepare work + ready slots)."""
from typing import Iterable, TYPE_CHECKING

from models import PipelineParams

if TYPE_CHECKING:
    from models import Batch


def remaining_prepare_ms(batch: "Batch", params: PipelineParams, now: float) -> float:
    """Nominal prepare time left for a batch still in prepare."""
    end = batch.close_time + params.t_prepare_nominal(batch.size)
    return max(0.0, end - now)


def ready_queue_cost_ms(ready: Iterable["Batch"], params: PipelineParams) -> float:
    """Prepare-slot cost for batches done preparing but not yet on infer."""
    return sum(params.t_prepare_nominal(b.size) for b in ready)


def pipeline_prepare_cost_ms(preparing: Iterable["Batch"], ready: Iterable["Batch"],
                             params: PipelineParams, now: float) -> float:
    """
    Total prepare-pipeline load.

    In prepare: cost decays with elapsed time (remaining nominal prepare).
    In ready: full prepare slot still reserved until infer starts (matches mc).
    """
    return (prepare_queue_cost_ms(preparing, params, now)
            + ready_queue_cost_ms(ready, params))


def prepare_queue_cost_ms(preparing: Iterable["Batch"], params: PipelineParams,
                          now: float) -> float:
    """Sum of remaining prepare work across in-flight prepare stages."""
    return sum(remaining_prepare_ms(b, params, now) for b in preparing)


def prepare_cost_admits(cost_ms: float, add_ms: float, budget_ms: float) -> bool:
    return cost_ms + add_ms <= budget_ms + 1e-9
