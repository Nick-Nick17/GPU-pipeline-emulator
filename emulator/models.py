"""
Core data structures shared across all modules.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum, auto


@dataclass
class PipelineParams:
    """Nominal (known) pipeline parameters."""
    a1: float  # prepare time per request (ms)
    c1: float  # prepare fixed overhead (ms)
    a2: float  # infer time per request (ms)
    c2: float  # infer fixed overhead (ms)
    variance: float = 0.2  # ±20% uniform variance

    def t_prepare_nominal(self, batch_size: int) -> float:
        return self.a1 * batch_size + self.c1

    def t_infer_nominal(self, batch_size: int) -> float:
        return self.a2 * batch_size + self.c2

    def t_total_nominal(self, batch_size: int) -> float:
        return self.t_prepare_nominal(batch_size) + self.t_infer_nominal(batch_size)

    def b_max_safe(self, sla_ms: float) -> int:
        """
        Largest batch size where T_total_worst <= sla_ms.
        Worst case: both prepare and infer at (1 + variance).
        """
        worst = 1 + self.variance
        a = (self.a1 + self.a2) * worst
        c = (self.c1 + self.c2) * worst
        return max(1, int((sla_ms - c) / a))


@dataclass
class Request:
    """A single request flowing through the system."""
    request_id: int
    arrival_time: float  # ms

    # filled in as request moves through pipeline
    batch_id: Optional[int] = None
    batch_close_time: Optional[float] = None   # when batch was sealed
    prepare_start_time: Optional[float] = None
    prepare_end_time: Optional[float] = None
    infer_start_time: Optional[float] = None
    infer_end_time: Optional[float] = None     # when response is returned
    dropped_at: Optional[float] = None         # shed load: rejected without processing

    @property
    def latency(self) -> Optional[float]:
        if self.infer_end_time is None:
            return None
        return self.infer_end_time - self.arrival_time

    @property
    def wait_in_queue(self) -> Optional[float]:
        if self.batch_close_time is None:
            return None
        return self.batch_close_time - self.arrival_time

    @property
    def pipeline_idle(self) -> Optional[float]:
        """Time prepare was done but infer slot wasn't ready yet."""
        if self.infer_start_time is None or self.prepare_end_time is None:
            return None
        return max(0.0, self.infer_start_time - self.prepare_end_time)


@dataclass
class Batch:
    """A sealed group of requests going through prepare → infer."""
    batch_id: int
    requests: List[Request]
    close_time: float  # when batch was sealed, prepare starts

    @property
    def size(self) -> int:
        return len(self.requests)


class EventType(Enum):
    REQUEST_ARRIVAL = auto()
    PREPARE_DONE    = auto()
    INFER_DONE      = auto()
    SCHEDULE_TICK   = auto()


@dataclass(order=True)
class Event:
    time: float
    event_type: EventType = field(compare=False)
    payload: object = field(default=None, compare=False)


@dataclass
class SystemState:
    """
    Snapshot of system visible to the Policy.
    Policy must NOT mutate this — read only.
    """
    now: float
    queue: List[Request]           # requests waiting to be batched
    infer_busy: bool               # is inference slot occupied?
    infer_end_time: Optional[float]  # when current infer will finish (None if idle)
    committed_count: int           # batches closed but not yet started inferring
    committed_infer_nominal: float  # sum of nominal infer times of committed batches
    prepare_queue_cost_ms: float   # remaining nominal prepare work (decays with time)
    preparing_count: int           # batches currently in prepare
    ready_count: int                 # prepare done, waiting for infer
    params: PipelineParams
    sla_ms: float
    batch_history: List[int]       # recent batch sizes (for adaptive policies)


@dataclass
class Decision:
    """
    What the Policy tells the Scheduler to do right now.
    All times are absolute simulation time (ms).
    """
    close_batch_at: Optional[float] = None   # None = don't close yet, wait for next event
    batch_size: Optional[int] = None         # how many to take from queue front (None = all)
    shed_hopeless: bool = False              # drop hopeless prefix before batch close
    drop_expired: bool = False               # drop queue prefix already past SLA deadline
    admit_infer: bool = False                # infer-backlog admission + shed ready batches
    max_committed: Optional[int] = None      # legacy batch-count cap (mc1 compare)
    max_prepare_cost_ms: Optional[float] = None  # time-based prepare load budget
    prepare_add_cost_ms: Optional[float] = None  # nominal/effective cost of planned close
    shed_worst: float = 1.0                 # worst-case factor for hopeless check
    shed_b: int = 1                          # batch size assumed for hopeless check


@dataclass
class BatchObservation:
    """
    Black-box record of one finished batch (task 2).
    The only thing we are allowed to know: when the batch entered the
    pipeline (was sealed) and when it came back. No prepare/infer split.
    """
    batch_id: int
    size: int
    close_time: float    # batch entered the pipeline (prepare started)
    return_time: float   # batch exited (all responses returned)

    @property
    def total_latency(self) -> float:
        return self.return_time - self.close_time


@dataclass
class AdvancedState:
    """
    Restricted snapshot for task 2 policies.

    Unlike SystemState, this DOES NOT expose internal pipeline structure
    (no params split, no infer_busy / infer_end_time, no committed work).
    Decisions may only rely on:
      - the current waiting queue (arrival times are observable)
      - the SLA budget
      - how many batches are currently in flight (entered but not returned)
      - the history of finished batches as (size -> total round-trip time)
    Policy must NOT mutate this — read only.
    """
    now: float
    queue: List[Request]              # requests waiting to be batched
    sla_ms: float
    in_flight: int                    # batches sealed but not yet returned
    observations: List[BatchObservation]  # finished batches (black-box timings)
