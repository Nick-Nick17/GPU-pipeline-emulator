import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from models import PipelineParams, Decision
from simulator import Simulator
from simple_policies import (
    TimeoutBatchPolicy,
    FixedSizePolicy,
    SLABudgetPolicy,
    HybridSLAOverlapPolicy,
    PredictiveOverlapPolicy,
    QueueFeedbackPolicy,
    OptimalOverlapPolicy,
)
from advanced_policies import (
    BlackBoxSLAOverlapPolicy,
    LatencyFeedbackPolicy,
    ThroughputMatchPolicy,
)
from metrics import compute_metrics

# Model in microseconds, converted to ms (our time unit):
#   prepare = PrepareCpuBeta*b + PrepareCpuAlpha = 0.149*b + 600 us
#   infer   = InferenceBeta*b   + InferenceAlpha = 0.2640*b + 404.52 us
#   MaxExecutionUs = 30000 us = 30 ms,  MaxBatchWeight = 4096
PARAMS = PipelineParams(
    a1=0.149 / 1000.0, c1=600.0 / 1000.0,
    a2=0.2640 / 1000.0, c2=404.52 / 1000.0,
    variance=0.2,
)

SEED = 42
SLA_MS = 30000.0 / 1000.0       # 30 ms
MAX_BATCH_WEIGHT = 4096

# Capacity is huge here (fixed overhead dominates), so RPS is in the millions and
# durations are short. Each scenario carries its own duration to keep stats decent
# at low load and runtime bounded at high load.


def burst(base, peak, base_s, peak_s):
    period_ms = (base_s + peak_s) * 1000.0
    hi_after_ms = base_s * 1000.0

    def rate(t_ms):
        return peak if (t_ms % period_ms) >= hi_after_ms else base

    return rate


SCENARIOS = [
    # name, rps, worst_case, duration_s
    ("light",      50_000,                              False, 1.0),
    ("loaded",     1_000_000,                           False, 0.3),
    ("bursty",     burst(400_000, 2_000_000, 0.03, 0.015), False, 0.3),
    ("overloaded", 4_500_000,                           False, 0.15),
]

# Grid for the reference "batch_size + timeout" policy (full Cartesian product).
BATCH_GRID = [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
TIMEOUT_GRID_MS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

# Short, heavily-overloaded probe used to rank baseline configs by max sustainable RPS.
BASELINE_PROBE_RPS = 4_500_000
BASELINE_PROBE_DURATION_S = 0.05

# Hardcoded reference found by the search (see --baseline). Filled in below.
BASELINE_BATCH = 4096
BASELINE_TIMEOUT_MS = 1.0


class BaselinePolicy:
    """
    The reference "batch_size + timeout": close the batch when B requests are
    queued OR T ms passed since the oldest waiting request. Uses only the queue
    and the clock, so it works in both simple and advanced (black-box) modes.
    """

    def __init__(self, batch_size, timeout_ms, max_batch_size):
        self.batch_size = min(batch_size, max_batch_size)
        self.timeout_ms = timeout_ms

    def name(self):
        return "BASELINE"

    def decide(self, state):
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)
        deadline = state.queue[0].arrival_time + self.timeout_ms
        if len(state.queue) >= self.batch_size or state.now >= deadline:
            return Decision(close_batch_at=state.now,
                            batch_size=min(len(state.queue), self.batch_size))
        return Decision(close_batch_at=deadline, batch_size=None)


def build_simple_policies(b_max):
    return [
        BaselinePolicy(BASELINE_BATCH, BASELINE_TIMEOUT_MS, max_batch_size=b_max),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=2.0, max_batch_size=b_max),
        HybridSLAOverlapPolicy(safety=1.2, collect_ms=0.0, max_batch_size=b_max),
        PredictiveOverlapPolicy(alpha=0.2, margin=1.1, safety=1.0, max_batch_size=b_max),
        QueueFeedbackPolicy(k=0.5, b_min=1, max_batch_size=b_max),
        SLABudgetPolicy(safety=1.0, max_batch_size=b_max),
        FixedSizePolicy(target_size=512, max_wait_ms=5.0, max_batch_size=b_max),
        TimeoutBatchPolicy(timeout_ms=5.0, max_batch_size=b_max),
        OptimalOverlapPolicy(safety=1.0, max_batch_size=b_max),
        OptimalOverlapPolicy(safety=1.2, max_batch_size=b_max),
    ]


def build_advanced_policies(b_max):
    return [
        BaselinePolicy(BASELINE_BATCH, BASELINE_TIMEOUT_MS, max_batch_size=b_max),
        BlackBoxSLAOverlapPolicy(safety=1.2, max_in_flight=2, collect_ms=0.0,
                                 bootstrap_max=64, hard_cap=b_max),
        BlackBoxSLAOverlapPolicy(safety=1.2, max_in_flight=2, collect_ms=2.0,
                                 bootstrap_max=64, hard_cap=b_max),
        BlackBoxSLAOverlapPolicy(safety=1.2, max_in_flight=1, collect_ms=0.0,
                                 bootstrap_max=64, hard_cap=b_max),
        LatencyFeedbackPolicy(max_in_flight=2, low=0.55, high=0.85,
                              init_b=64.0, hard_cap=b_max),
        ThroughputMatchPolicy(alpha=0.2, margin=1.15, safety=1.2, max_in_flight=2,
                              hard_cap=b_max),
    ]


def run_scenario(name, rps, worst_case, duration_s, mode):
    sim_ms = duration_s * 1000
    sla_ms = SLA_MS
    b_max = min(PARAMS.b_max_safe(sla_ms), MAX_BATCH_WEIGHT)
    mult = f"x{1 + PARAMS.variance:.1f} fixed" if worst_case else "U(0.8,1.2)"
    rps_label = "burst" if callable(rps) else f"{rps}"

    build = build_advanced_policies if mode == "advanced" else build_simple_policies

    print(f"\n\n{'=' * 86}")
    print(f"  SCENARIO {name}   RPS={rps_label}  dur={duration_s}s  SLA={sla_ms:.0f}ms  "
          f"time={mult}  b_max={b_max}  mode={mode}")
    print(f"{'=' * 86}")
    print(f"  {'Policy':<38} {'maxRPS':>11} {'Success':>8} {'Rate':>7} {'Late':>6} {'batch':>6} "
          f"{'p50':>9} {'p90':>9} {'p99':>9} {'idle':>6}")
    print(f"  {'-' * 38} {'-' * 11} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 6} "
          f"{'-' * 9} {'-' * 9} {'-' * 9} {'-' * 6}")

    results = []
    for policy in build(b_max):
        sim = Simulator(
            params=PARAMS,
            policy=policy,
            rps=rps,
            sla_ms=sla_ms,
            sim_duration_ms=sim_ms,
            seed=SEED,
            worst_case=worst_case,
            mode=mode,
        )
        sim.run()
        m = compute_metrics(
            policy_name=policy.name(),
            all_requests=sim.all_requests,
            sla_ms=sla_ms,
            batch_sizes=sim.batch_sizes,
            sim_end_ms=sim_ms,
        )
        results.append(m)

    def max_rps(m):
        # throughput we can actually process (capacity), independent of SLA backlog
        return m.completed / duration_s

    # BASELINE pinned first; the rest ranked by max sustainable RPS, then p99.
    baseline = [m for m in results if m.policy_name == "BASELINE"]
    rest = sorted((m for m in results if m.policy_name != "BASELINE"),
                  key=lambda x: (-max_rps(x), x.p99_ms))
    for m in baseline + rest:
        print(
            f"  {m.policy_name:<38} "
            f"{max_rps(m):>11,.0f} "
            f"{m.success_count:>8} "
            f"{m.success_rate:>6.1%} "
            f"{m.late_unserved:>6} "
            f"{m.avg_batch_size:>6.1f} "
            f"{m.p50_ms:>9.3f} "
            f"{m.p90_ms:>9.3f} "
            f"{m.p99_ms:>9.3f} "
            f"{m.avg_idle_ms:>6.1f}"
        )


def run_baseline_search():
    """
    Grid-search the best batch_size + timeout reference over the full (B, T) grid.
    Objective: maximum sustainable RPS = successful-within-SLA requests per second,
    measured under a short heavy-overload probe. Tie-break by lower p99.
    """
    sim_ms = BASELINE_PROBE_DURATION_S * 1000
    sla_ms = SLA_MS
    b_max = min(PARAMS.b_max_safe(sla_ms), MAX_BATCH_WEIGHT)
    batch_grid = [b for b in BATCH_GRID if b <= b_max] or [b_max]

    def max_rps(m):
        # capacity = requests actually processed per second (not SLA-limited)
        return m.completed / BASELINE_PROBE_DURATION_S

    print(f"\n\n{'=' * 86}")
    print(f"  BASELINE SEARCH (batch_size + timeout)   probe RPS={BASELINE_PROBE_RPS}  "
          f"dur={BASELINE_PROBE_DURATION_S}s  SLA={sla_ms:.0f}ms  b_max={b_max}")
    print(f"  ranked by max sustainable RPS (throughput = processed / sec)   "
          f"grid: {len(batch_grid)}x{len(TIMEOUT_GRID_MS)} = "
          f"{len(batch_grid) * len(TIMEOUT_GRID_MS)} runs")
    print(f"{'=' * 86}")
    print(f"  {'batch':>6} {'timeout':>8} {'maxRPS':>12} {'Rate':>7} {'avgB':>7} "
          f"{'p50':>9} {'p90':>9} {'p99':>9}")
    print(f"  {'-' * 6} {'-' * 8} {'-' * 12} {'-' * 7} {'-' * 7} "
          f"{'-' * 9} {'-' * 9} {'-' * 9}")

    results = []
    for b in batch_grid:
        for t in TIMEOUT_GRID_MS:
            policy = BaselinePolicy(b, t, max_batch_size=b_max)
            sim = Simulator(
                params=PARAMS, policy=policy, rps=BASELINE_PROBE_RPS, sla_ms=sla_ms,
                sim_duration_ms=sim_ms, seed=SEED, worst_case=False, mode="simple",
            )
            sim.run()
            m = compute_metrics(
                policy_name=f"B={b},T={t}", all_requests=sim.all_requests,
                sla_ms=sla_ms, batch_sizes=sim.batch_sizes, sim_end_ms=sim_ms,
            )
            results.append((b, t, m))

    results.sort(key=lambda x: (-max_rps(x[2]), x[2].p99_ms))
    for b, t, m in results:
        print(
            f"  {b:>6} {t:>8.1f} {max_rps(m):>12,.0f} {m.success_rate:>6.1%} "
            f"{m.avg_batch_size:>7.1f} "
            f"{m.p50_ms:>9.3f} {m.p90_ms:>9.3f} {m.p99_ms:>9.3f}"
        )

    bb, bt, bm = results[0]
    print(f"\n  >>> BEST reference: batch_size={bb}, timeout={bt:.1f}ms  ->  "
          f"maxRPS={max_rps(bm):,.0f}  (set BASELINE_BATCH / BASELINE_TIMEOUT_MS)")
    return bb, bt, bm


def main():
    parser = argparse.ArgumentParser(description="GPU pipeline batching emulator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--simple", action="store_const", dest="mode", const="simple",
                       help="task 1: policies know prepare/infer split (default)")
    group.add_argument("--advanced", action="store_const", dest="mode", const="advanced",
                       help="task 2: black-box, only batch entry/exit timing is known")
    parser.add_argument("--baseline", action="store_true",
                        help="phase 0: grid-search the best batch_size+timeout reference")
    parser.set_defaults(mode="simple")
    args = parser.parse_args()

    if args.baseline:
        print(f"Pipeline: prepare = {PARAMS.a1}*b + {PARAMS.c1}   "
              f"infer = {PARAMS.a2}*b + {PARAMS.c2}   variance=±{PARAMS.variance:.0%}")
        run_baseline_search()
        print()
        return

    label = "TASK 2 (black box: only batch in/out times)" if args.mode == "advanced" \
        else "TASK 1 (known prepare/infer split)"
    print(f"Mode: {args.mode}  ->  {label}")
    print(f"Pipeline: prepare = {PARAMS.a1}*b + {PARAMS.c1}   "
          f"infer = {PARAMS.a2}*b + {PARAMS.c2}   variance=±{PARAMS.variance:.0%}")
    for name, rps, worst_case, duration_s in SCENARIOS:
        run_scenario(name, rps, worst_case, duration_s, args.mode)
    print()


if __name__ == "__main__":
    main()
