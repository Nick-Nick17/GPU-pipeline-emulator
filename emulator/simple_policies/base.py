from abc import ABC, abstractmethod
from typing import Optional
from models import SystemState, Decision


class BasePolicy(ABC):
    @abstractmethod
    def decide(self, state: SystemState) -> Decision:
        ...

    def name(self) -> str:
        return self.__class__.__name__

    def _cap(self, state: SystemState, max_batch_size: Optional[int]) -> int:
        cap = state.params.b_max_safe(state.sla_ms)
        if max_batch_size is not None:
            cap = min(cap, max_batch_size)
        return max(1, cap)

    def _infer_free_at(self, state: SystemState, worst: float = 1.0) -> float:
        base = state.infer_end_time if state.infer_busy else state.now
        return base + worst * state.committed_infer_nominal
