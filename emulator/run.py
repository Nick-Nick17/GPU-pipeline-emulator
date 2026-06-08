import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from models import PipelineParams
from simulator import Simulator
from policies import (
    TimeoutBatchPolicy,
    FixedSizePolicy,
    SLABudgetPolicy,
)
from optimal_policy import OptimalOverlapPolicy

from simple_policies import (
    HybridSLAOverlapPolicy,
    PredictiveOverlapPolicy,
    QueueFeedbackPolicy,
)
from advanced_policies import (
    BlackBoxSLAOverlapPolicy,
    LatencyFeedbackPolicy,
    ThroughputMatchPolicy,
)
from metrics import compute_metrics

PARAMS = PipelineParams(a1=1.0, c1=10.0, a2=2.0, c2=80.0, variance=0.2)

SIM_DURATION_S = 300
SEED = 42

SLA_MS = 1500.0


def burst(base, peak, base_s, peak_s):
    period_ms = (base_s + peak_s) * 1000.0
    hi_after_ms = base_s * 1000.0

    def rate(t_ms):
        return peak if (t_ms % period_ms) >= hi_after_ms else base

    return rate


SCENARIOS = [
    ("light",      30,                      False),
    ("loaded",     250,                     False),
    ("bursty",     burst(80, 240, 6, 2),    False),
    ("overloaded", 350,                     True),
]


def build_simple_policies(b_max):
    return [
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=150.0, max_batch_size=b_max),
        HybridSLAOverlapPolicy(safety=1.2, collect_ms=0.0, max_batch_size=b_max),
        PredictiveOverlapPolicy(alpha=0.2, margin=1.1, safety=1.0, max_batch_size=b_max),
        QueueFeedbackPolicy(k=0.5, b_min=1, max_batch_size=b_max),
        SLABudgetPolicy(safety=1.0, max_batch_size=b_max),
        FixedSizePolicy(target_size=32, max_wait_ms=500, max_batch_size=b_max),
        TimeoutBatchPolicy(timeout_ms=200, max_batch_size=b_max),
        OptimalOverlapPolicy(safety=1.0, max_batch_size=b_max),
        OptimalOverlapPolicy(safety=1.2, max_batch_size=b_max),
    ]


def build_advanced_policies(b_max):
    return [
        BlackBoxSLAOverlapPolicy(safety=1.2, max_in_flight=2, collect_ms=0.0),
        BlackBoxSLAOverlapPolicy(safety=1.2, max_in_flight=2, collect_ms=150.0),
        BlackBoxSLAOverlapPolicy(safety=1.2, max_in_flight=1, collect_ms=0.0),
        LatencyFeedbackPolicy(max_in_flight=2, low=0.55, high=0.85),
        ThroughputMatchPolicy(alpha=0.2, margin=1.15, safety=1.2, max_in_flight=2),
    ]


def run_scenario(name, rps, worst_case, mode):
    sim_ms = SIM_DURATION_S * 1000
    sla_ms = SLA_MS
    b_max = PARAMS.b_max_safe(sla_ms)
    mult = f"x{1 + PARAMS.variance:.1f} fixed" if worst_case else "U(0.8,1.2)"
    rps_label = "burst" if callable(rps) else f"{rps}"

    build = build_advanced_policies if mode == "advanced" else build_simple_policies

    print(f"\n\n{'=' * 86}")
    print(f"  SCENARIO {name}   RPS={rps_label}  SLA={sla_ms:.0f}ms  "
          f"time={mult}  b_max={b_max}  mode={mode}")
    print(f"{'=' * 86}")
    print(f"  {'Policy':<38} {'Success':>8} {'Rate':>7} {'Late':>6} {'batch':>6} "
          f"{'p50':>6} {'p90':>6} {'p99':>6} {'idle':>6}")
    print(f"  {'-' * 38} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 6} "
          f"{'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}")

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

    for m in sorted(results, key=lambda x: (-x.success_count, x.p99_ms)):
        print(
            f"  {m.policy_name:<38} "
            f"{m.success_count:>8} "
            f"{m.success_rate:>6.1%} "
            f"{m.late_unserved:>6} "
            f"{m.avg_batch_size:>6.1f} "
            f"{m.p50_ms:>6.0f} "
            f"{m.p90_ms:>6.0f} "
            f"{m.p99_ms:>6.0f} "
            f"{m.avg_idle_ms:>6.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description="GPU pipeline batching emulator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--simple", action="store_const", dest="mode", const="simple",
                       help="task 1: policies know prepare/infer split (default)")
    group.add_argument("--advanced", action="store_const", dest="mode", const="advanced",
                       help="task 2: black-box, only batch entry/exit timing is known")
    parser.set_defaults(mode="simple")
    args = parser.parse_args()

    label = "TASK 2 (black box: only batch in/out times)" if args.mode == "advanced" \
        else "TASK 1 (known prepare/infer split)"
    print(f"Mode: {args.mode}  ->  {label}")
    print(f"Pipeline: prepare = {PARAMS.a1}*b + {PARAMS.c1}   "
          f"infer = {PARAMS.a2}*b + {PARAMS.c2}   variance=±{PARAMS.variance:.0%}")
    for name, rps, worst_case in SCENARIOS:
        run_scenario(name, rps, worst_case, args.mode)
    print()


if __name__ == "__main__":
    main()
