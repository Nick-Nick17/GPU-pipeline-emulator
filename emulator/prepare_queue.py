"""Time-based prepare pipeline load (remaining nominal prepare work)."""
from typing import Iterable, TYPE_CHECKING

from models import PipelineParams

if TYPE_CHECKING:
    from models import Batch


def remaining_prepare_ms(batch: "Batch", params: PipelineParams, now: float) -> float:
    """Nominal prepare time left for a batch still in prepare."""
    end = batch.close_time + params.t_prepare_nominal(batch.size)
    return max(0.0, end - now)


def prepare_queue_cost_ms(preparing: Iterable["Batch"], params: PipelineParams,
                          now: float) -> float:
    """Sum of remaining prepare work across in-flight prepare stages."""
    return sum(remaining_prepare_ms(b, params, now) for b in preparing)


def prepare_cost_admits(cost_ms: float, add_ms: float, budget_ms: float) -> bool:
    return cost_ms + add_ms <= budget_ms + 1e-9
