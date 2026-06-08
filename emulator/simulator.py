"""
Event-driven simulator.

Time unit: milliseconds.
prepare stages may run in parallel; infer runs one batch at a time.

Flow:
  REQUEST_ARRIVAL  → add to queue, consult policy
  SCHEDULE_TICK    → consult policy at a requested time
  PREPARE_DONE     → batch becomes ready; dispatch to infer if slot free
  INFER_DONE       → record latencies, dispatch next ready batch
"""

import heapq
import random
from collections import deque
from typing import List, Optional, Deque
from models import (
    PipelineParams, Request, Batch, Event, EventType,
    SystemState, Decision
)
from environment import PipelineEnvironment
from policies import BasePolicy


class Simulator:

    def __init__(
        self,
        params: PipelineParams,
        policy: BasePolicy,
        rps: float,
        sla_ms: float,
        sim_duration_ms: float,
        seed: Optional[int] = 42,
        arrival_jitter: float = 0.25,
        worst_case: bool = False,
    ):
        self.params = params
        self.policy = policy
        self.rps = rps
        self.sla_ms = sla_ms
        self.sim_duration_ms = sim_duration_ms
        self.arrival_jitter = arrival_jitter

        self.env = PipelineEnvironment(params, seed=seed, worst_case=worst_case)
        self.rng = random.Random(seed)

        # State
        self._queue: Deque[Request] = deque()
        self._events: List[Event] = []
        self._now: float = 0.0
        self._batch_counter: int = 0
        self._request_counter: int = 0

        # Pipeline state
        self._infer_busy: bool = False
        self._infer_end_time: Optional[float] = None
        self._ready_batches: Deque[Batch] = deque()
        self._committed_count: int = 0
        self._committed_infer_nominal: float = 0.0

        # Results
        self.all_requests: List[Request] = []
        self.completed_requests: List[Request] = []
        self.batch_sizes: List[int] = []

        # Scheduled ticks (to avoid duplicate SCHEDULE_TICK events)
        self._scheduled_tick_at: Optional[float] = None

        self._arrival_clock: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def total_generated(self) -> int:
        return self._request_counter

    def run(self) -> List[Request]:
        """Run simulation and return all completed requests."""
        self._schedule_next_arrival()
        self._push_event(Event(0.0, EventType.SCHEDULE_TICK))

        while self._events:
            event = heapq.heappop(self._events)
            if event.time > self.sim_duration_ms:
                break
            self._now = event.time
            self._handle(event)

        return self.completed_requests

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------

    def _schedule_next_arrival(self):
        rate = self.rps(self._arrival_clock) if callable(self.rps) else self.rps
        period = 1000.0 / rate
        delta = self.rng.uniform(-self.arrival_jitter * period, self.arrival_jitter * period)
        self._arrival_clock += period + delta
        if self._arrival_clock >= self.sim_duration_ms:
            return
        req = Request(request_id=self._request_counter, arrival_time=self._arrival_clock)
        self._request_counter += 1
        self.all_requests.append(req)
        self._push_event(Event(self._arrival_clock, EventType.REQUEST_ARRIVAL, req))

    def _push_event(self, event: Event):
        heapq.heappush(self._events, event)

    def _schedule_tick_at(self, t: float):
        """Ask scheduler to wake up at time t (deduplicated)."""
        if t <= self._now:
            return
        if self._scheduled_tick_at is not None and self._scheduled_tick_at <= t:
            return  # already have an earlier tick
        self._scheduled_tick_at = t
        self._push_event(Event(t, EventType.SCHEDULE_TICK))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle(self, event: Event):
        if event.event_type == EventType.REQUEST_ARRIVAL:
            self._on_request_arrival(event.payload)

        elif event.event_type == EventType.SCHEDULE_TICK:
            if self._scheduled_tick_at == event.time:
                self._scheduled_tick_at = None
            self._consult_policy()

        elif event.event_type == EventType.PREPARE_DONE:
            self._on_prepare_done(event.payload)

        elif event.event_type == EventType.INFER_DONE:
            self._on_infer_done(event.payload)

    def _on_request_arrival(self, req: Request):
        self._schedule_next_arrival()
        self._queue.append(req)
        self._consult_policy()

    def _on_batch_close(self, size: int):
        if not self._queue:
            return

        actual_size = min(size, len(self._queue))
        requests = [self._queue.popleft() for _ in range(actual_size)]

        self._batch_counter += 1
        batch = Batch(
            batch_id=self._batch_counter,
            requests=requests,
            close_time=self._now,
        )
        self.batch_sizes.append(batch.size)

        for req in requests:
            req.batch_id = batch.batch_id
            req.batch_close_time = self._now
            req.prepare_start_time = self._now

        self._committed_count += 1
        self._committed_infer_nominal += self.params.t_infer_nominal(batch.size)

        actual_prepare = self.env.actual_prepare_time(batch.size)
        prepare_end = self._now + actual_prepare

        for req in requests:
            req.prepare_end_time = prepare_end

        self._push_event(Event(prepare_end, EventType.PREPARE_DONE, batch))

    def _on_prepare_done(self, batch: Batch):
        self._ready_batches.append(batch)
        self._dispatch_infer()
        self._consult_policy()

    def _on_infer_done(self, batch: Batch):
        self._infer_busy = False
        self._infer_end_time = None

        for req in batch.requests:
            self.completed_requests.append(req)

        self._dispatch_infer()
        self._consult_policy()

    def _dispatch_infer(self):
        if self._infer_busy or not self._ready_batches:
            return
        self._start_infer(self._ready_batches.popleft())

    def _start_infer(self, batch: Batch):
        self._committed_count -= 1
        self._committed_infer_nominal -= self.params.t_infer_nominal(batch.size)

        actual_infer = self.env.actual_infer_time(batch.size)
        infer_end = self._now + actual_infer

        self._infer_busy = True
        self._infer_end_time = infer_end

        for req in batch.requests:
            req.infer_start_time = self._now
            req.infer_end_time = infer_end

        self._push_event(Event(infer_end, EventType.INFER_DONE, batch))

    # ------------------------------------------------------------------
    # Policy consultation
    # ------------------------------------------------------------------

    def _consult_policy(self):
        state = SystemState(
            now=self._now,
            queue=self._queue,
            infer_busy=self._infer_busy,
            infer_end_time=self._infer_end_time,
            committed_count=self._committed_count,
            committed_infer_nominal=self._committed_infer_nominal,
            params=self.params,
            sla_ms=self.sla_ms,
            batch_history=list(self.batch_sizes[-20:]),
        )

        decision = self.policy.decide(state)

        if decision.close_batch_at is None:
            return

        if decision.close_batch_at <= self._now:
            self._on_batch_close(decision.batch_size or len(self._queue))
        else:
            self._schedule_tick_at(decision.close_batch_at)
