"""
Metrics: compute and display results from completed requests.
"""
import statistics
from dataclasses import dataclass
from typing import List, Optional
from models import Request


@dataclass
class SimMetrics:
    policy_name: str
    generated: int
    counted: int
    completed: int
    excluded: int
    late_unserved: int
    dropped: int
    success_count: int
    success_rate: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    avg_batch_size: float
    avg_idle_ms: float
    sla_ms: float

    def __str__(self) -> str:
        return (
            f"\n{'─' * 55}\n"
            f"  Policy : {self.policy_name}\n"
            f"  SLA    : {self.sla_ms:.0f} ms\n"
            f"{'─' * 55}\n"
            f"  Requests  generated={self.generated}  counted={self.counted}  "
            f"completed={self.completed}  excluded(tail)={self.excluded}\n"
            f"  Success   {self.success_count} ({self.success_rate:.1%} of counted)  "
            f"late_unserved={self.late_unserved}  dropped={self.dropped}\n"
            f"  Latency   p50={self.p50_ms:.0f}  p90={self.p90_ms:.0f}  "
            f"p99={self.p99_ms:.0f}  mean={self.mean_ms:.0f}  ms\n"
            f"  Avg batch size : {self.avg_batch_size:.1f}\n"
            f"  Avg idle (prepare→infer wait) : {self.avg_idle_ms:.1f} ms\n"
            f"{'─' * 55}"
        )


def sla_rps(m: SimMetrics, duration_s: float) -> float:
    return m.success_count / duration_s


def avg_batch_weighted(batch_sizes: List[int]) -> float:
    """Mean batch size for a random request (weights large batches correctly)."""
    total = sum(batch_sizes)
    if not total:
        return 0.0
    return sum(s * s for s in batch_sizes) / total


def percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def compute_metrics(
    policy_name: str,
    all_requests: List[Request],
    sla_ms: float,
    batch_sizes: List[int],
    sim_end_ms: float,
    dropped_requests: Optional[List[Request]] = None,
) -> SimMetrics:
    latencies = []
    idles = []
    success = 0
    completed = 0
    late_unserved = 0
    excluded = 0

    dropped = len(dropped_requests) if dropped_requests else sum(
        1 for r in all_requests if r.dropped_at is not None)

    for r in all_requests:
        if r.dropped_at is not None:
            continue
        if r.latency is not None:
            completed += 1
            latencies.append(r.latency)
            if r.pipeline_idle is not None:
                idles.append(r.pipeline_idle)
            if r.latency <= sla_ms:
                success += 1
        elif r.arrival_time + sla_ms <= sim_end_ms:
            late_unserved += 1
        else:
            excluded += 1

    counted = completed + late_unserved + dropped

    return SimMetrics(
        policy_name=policy_name,
        generated=len(all_requests),
        counted=counted,
        completed=completed,
        excluded=excluded,
        late_unserved=late_unserved,
        dropped=dropped,
        success_count=success,
        success_rate=success / counted if counted else 0.0,
        p50_ms=percentile(latencies, 50),
        p90_ms=percentile(latencies, 90),
        p99_ms=percentile(latencies, 99),
        mean_ms=statistics.mean(latencies) if latencies else 0.0,
        avg_batch_size=avg_batch_weighted(batch_sizes),
        avg_idle_ms=statistics.mean(idles) if idles else 0.0,
        sla_ms=sla_ms,
    )
