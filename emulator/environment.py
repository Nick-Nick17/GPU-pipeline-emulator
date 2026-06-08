"""
Environment: physical reality of the pipeline.
Knows actual (noisy) execution times. Makes no decisions.
"""
import random
from typing import Optional
from models import PipelineParams, Batch


class PipelineEnvironment:
    """
    Simulates actual execution times with uniform variance.

    This is the only place where randomness lives.
    Scheduler and Policy never see real times — only nominal ones.
    """

    def __init__(self, params: PipelineParams, seed: Optional[int] = None,
                 worst_case: bool = False):
        self.params = params
        self.rng = random.Random(seed)
        self.worst_case = worst_case

    def _sample_multiplier(self) -> float:
        v = self.params.variance
        if self.worst_case:
            return 1.0 + v
        return self.rng.uniform(1 - v, 1 + v)

    def actual_prepare_time(self, batch_size: int) -> float:
        """Real prepare duration (ms). Unknown to scheduler in advance."""
        nominal = self.params.t_prepare_nominal(batch_size)
        return nominal * self._sample_multiplier()

    def actual_infer_time(self, batch_size: int) -> float:
        """Real infer duration (ms). Unknown to scheduler in advance."""
        nominal = self.params.t_infer_nominal(batch_size)
        return nominal * self._sample_multiplier()
