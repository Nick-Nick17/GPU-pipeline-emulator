"""Queue / ready-batch shedding and infer-backlog admission."""
from typing import TYPE_CHECKING

from models import Decision, Request

if TYPE_CHECKING:
    from simulator import Simulator


def apply(decision: Decision, sim: "Simulator") -> None:
    if decision.drop_expired:
        drop_expired_queue(sim)
    if decision.admit_infer:
        shed_queue_prefix(sim, decision.shed_worst, decision.shed_b,
                          use_infer_backlog=True)
        shed_ready_batches(sim, decision.shed_worst)
    elif decision.shed_hopeless:
        shed_queue_prefix(sim, decision.shed_worst, decision.shed_b)


def admit_batch_close(sim: "Simulator", size: int, worst: float) -> bool:
    if not sim._queue:
        return False
    b = max(1, min(size, len(sim._queue)))
    if _salvageable_at_infer(sim, sim._queue[0], worst, b):
        return True
    shed_queue_prefix(sim, worst, b, use_infer_backlog=True)
    if not sim._queue:
        return False
    b = max(1, min(size, len(sim._queue)))
    return _salvageable_at_infer(sim, sim._queue[0], worst, b)


def drop_expired_queue(sim: "Simulator") -> None:
    """Drop waiting requests whose SLA deadline has already passed."""
    while sim._queue and sim._now > sim._queue[0].arrival_time + sim.sla_ms:
        _drop(sim, sim._queue.popleft())


def shed_queue_prefix(sim: "Simulator", worst: float, shed_b: int,
                      use_infer_backlog: bool = False) -> None:
    while sim._queue:
        b = max(1, min(len(sim._queue), shed_b))
        req = sim._queue[0]
        deadline = (_est_infer_complete(sim, worst, b) if use_infer_backlog
                    else sim._now + worst_processing(sim, b, worst))
        if _salvageable(req, sim.sla_ms, deadline):
            break
        sim._queue.popleft()
        _drop(sim, req)


def shed_ready_batches(sim: "Simulator", worst: float) -> None:
    p = sim.params
    slot = sim._infer_end_time if sim._infer_busy else sim._now
    while sim._ready_batches:
        batch = sim._ready_batches[0]
        infer_end = slot + worst * p.t_infer_nominal(batch.size)
        if _salvageable(batch.requests[0], sim.sla_ms, infer_end):
            break
        sim._ready_batches.popleft()
        sim._committed_count -= 1
        sim._committed_infer_nominal -= p.t_infer_nominal(batch.size)
        sim._in_flight -= 1
        for req in batch.requests:
            _drop(sim, req)
        slot = infer_end


def _salvageable(req: Request, sla_ms: float, completion_est: float) -> bool:
    return req.arrival_time + sla_ms > completion_est


def _salvageable_at_infer(sim: "Simulator", req: Request, worst: float, b: int) -> bool:
    return _salvageable(req, sim.sla_ms, _est_infer_complete(sim, worst, b))


def _infer_free(sim: "Simulator", worst: float) -> float:
    base = sim._infer_end_time if sim._infer_busy else sim._now
    return base + worst * sim._committed_infer_nominal


def _est_infer_complete(sim: "Simulator", worst: float, batch_size: int) -> float:
    return _infer_free(sim, worst) + worst * sim.params.t_infer_nominal(batch_size)


def worst_processing(sim: "Simulator", b: int, worst: float) -> float:
    p = sim.params
    return worst * (p.t_prepare_nominal(b) + p.t_infer_nominal(b))


def _drop(sim: "Simulator", req: Request) -> None:
    req.dropped_at = sim._now
    sim.dropped_requests.append(req)
