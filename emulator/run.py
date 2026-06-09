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
    HybridSLAOverlapLegacyPolicy,
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
    # name, rps, worst_case, duration_s [, opts]
    # opts may include sla_ms, b_max, hybrid_early_drain, baseline=(batch, timeout_ms).
  # If baseline is omitted, SCENARIO_BASELINES[name] is used (from --baseline search).
    ("light",      50_000,                              False, 1.0),
    ("loaded",     3_000_000,                           False, 0.3),
    ("bursty",     burst(400_000, 2_000_000, 0.03, 0.015), False, 0.3),
    ("overloaded", 4_000_000,                           False, 0.15),
    ("tight_sla",  2_850_000,                           False, 0.3, {
        "sla_ms": 6.0,
        "hybrid_early_drain": False,
    }),
]

# Per-scenario reference (batch_size, timeout_ms), ranked by slaRPS on that scenario.
# Re-tune with: python run.py --baseline
SCENARIO_BASELINES = {
    "light":      (32,   0.5),
    "loaded":     (4096, 2.0),
    "bursty":     (2048, 1.0),
    "overloaded": (4096, 1.0),
    "tight_sla":  (4096, 2.0),
}

# Grid for the reference "batch_size + timeout" policy (full Cartesian product).
BATCH_GRID = [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
TIMEOUT_GRID_MS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

# Coarse grid for a quick per-scenario search during --baseline (full grid is slow).
BASELINE_SEARCH_BATCH = [32, 128, 512, 1024, 2048, 4096]
BASELINE_SEARCH_TIMEOUT_MS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


class BaselinePolicy:
    """
    The reference "batch_size + timeout": close the batch when B requests are
    queued OR T ms passed since the oldest waiting request. Uses only the queue
    and the clock, so it works in both simple and advanced (black-box) modes.
    """

    def __init__(self, batch_size, timeout_ms, max_batch_size):
        self.batch_size = min(batch_size, max_batch_size)
        self.timeout_ms = timeout_ms
        self.max_batch_size = max_batch_size

    def name(self):
        return f"BASELINE(B={self.batch_size},T={self.timeout_ms:.1f})"

    def decide(self, state):
        if not state.queue:
            return Decision(close_batch_at=None, batch_size=None)
        deadline = state.queue[0].arrival_time + self.timeout_ms
        if len(state.queue) >= self.batch_size or state.now >= deadline:
            return Decision(close_batch_at=state.now,
                            batch_size=min(len(state.queue), self.batch_size))
        return Decision(close_batch_at=deadline, batch_size=None)


def build_simple_policies(b_max, baseline_batch, baseline_timeout_ms,
                          hybrid_early_drain=True):
    hybrid_kw = {"early_drain": hybrid_early_drain}
    return [
        BaselinePolicy(baseline_batch, baseline_timeout_ms, max_batch_size=b_max),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max,
                               **hybrid_kw),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=2.0, max_batch_size=b_max,
                               **hybrid_kw),
        HybridSLAOverlapPolicy(safety=1.2, collect_ms=0.0, max_batch_size=b_max,
                               **hybrid_kw),
        HybridSLAOverlapLegacyPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max,
                                     **hybrid_kw),
        HybridSLAOverlapLegacyPolicy(safety=1.0, collect_ms=2.0, max_batch_size=b_max,
                                     **hybrid_kw),
        PredictiveOverlapPolicy(alpha=0.2, margin=1.1, safety=1.0, max_batch_size=b_max),
        QueueFeedbackPolicy(k=0.5, b_min=1, max_batch_size=b_max),
        SLABudgetPolicy(safety=1.0, max_batch_size=b_max),
        FixedSizePolicy(target_size=512, max_wait_ms=5.0, max_batch_size=b_max),
        TimeoutBatchPolicy(timeout_ms=5.0, max_batch_size=b_max),
        OptimalOverlapPolicy(safety=1.0, max_batch_size=b_max),
        OptimalOverlapPolicy(safety=1.2, max_batch_size=b_max),
    ]


def build_advanced_policies(b_max, baseline_batch, baseline_timeout_ms):
    return [
        BaselinePolicy(baseline_batch, baseline_timeout_ms, max_batch_size=b_max),
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


def _probe_rps(rps):
    """For bursty traffic, tune baseline against the peak rate."""
    return 2_000_000 if callable(rps) else rps


def search_baseline(name, rps, worst_case, duration_s, sla_ms, b_max, mode,
                    batch_grid=None, timeout_grid=None):
    """Grid-search batch_size + timeout for one scenario; rank by slaRPS."""
    batch_grid = batch_grid or [b for b in BASELINE_SEARCH_BATCH if b <= b_max]
    timeout_grid = timeout_grid or BASELINE_SEARCH_TIMEOUT_MS
    sim_ms = duration_s * 1000
    probe = _probe_rps(rps)

    def sla_rps(m):
        return m.success_count / duration_s

    best = None
    for b in batch_grid:
        for t in timeout_grid:
            policy = BaselinePolicy(b, t, max_batch_size=b_max)
            sim = Simulator(
                params=PARAMS, policy=policy, rps=probe if callable(rps) else rps,
                sla_ms=sla_ms, sim_duration_ms=sim_ms, seed=SEED,
                worst_case=worst_case, mode=mode,
            )
            sim.run()
            m = compute_metrics(
                policy_name=policy.name(), all_requests=sim.all_requests,
                sla_ms=sla_ms, batch_sizes=sim.batch_sizes, sim_end_ms=sim_ms,
            )
            key = (sla_rps(m), m.success_rate, -m.p99_ms)
            if best is None or key > best[0]:
                best = (key, b, t, m)
    _, bb, bt, bm = best
    return bb, bt, bm


def resolve_baseline(name, rps, worst_case, duration_s, sla_ms, b_max, mode, opts):
    if "baseline" in opts:
        return opts["baseline"]
    if name in SCENARIO_BASELINES:
        return SCENARIO_BASELINES[name]
    print(f"  (no cached baseline for {name}, searching...)", flush=True)
    bb, bt, _ = search_baseline(name, rps, worst_case, duration_s, sla_ms, b_max, mode)
    return bb, bt


def run_scenario(name, rps, worst_case, duration_s, mode, opts=None):
    opts = opts or {}
    sim_ms = duration_s * 1000
    sla_ms = opts.get("sla_ms", SLA_MS)
    b_max = opts.get("b_max", min(PARAMS.b_max_safe(sla_ms), MAX_BATCH_WEIGHT))
    hybrid_early_drain = opts.get("hybrid_early_drain", True)
    baseline_batch, baseline_timeout_ms = resolve_baseline(
        name, rps, worst_case, duration_s, sla_ms, b_max, mode, opts)
    mult = f"x{1 + PARAMS.variance:.1f} fixed" if worst_case else "U(0.8,1.2)"
    rps_label = "burst" if callable(rps) else f"{rps}"

    if mode == "advanced":
        build = lambda cap: build_advanced_policies(
            cap, baseline_batch, baseline_timeout_ms)
    else:
        build = lambda cap: build_simple_policies(
            cap, baseline_batch, baseline_timeout_ms,
            hybrid_early_drain=hybrid_early_drain)

    extra = ""
    if not hybrid_early_drain:
        extra = "  hybrid_drain=off"

    print(f"\n\n{'=' * 86}")
    print(f"  SCENARIO {name}   RPS={rps_label}  dur={duration_s}s  SLA={sla_ms:.0f}ms  "
          f"time={mult}  b_max={b_max}  mode={mode}{extra}")
    print(f"  reference: BASELINE(B={baseline_batch}, T={baseline_timeout_ms:.1f}ms)")
    print(f"{'=' * 86}")
    print(f"  {'Policy':<38} {'slaRPS':>11} {'Success':>8} {'Rate':>7} {'Late':>6} {'batch':>6} "
          f"{'p50':>9} {'p90':>9} {'p99':>9} {'idle':>6}")
    print(f"  {'-' * 38} {'-' * 11} {'-' * 8} {'-' * 7} {'-' * 6} {'-' * 6} "
          f"{'-' * 9} {'-' * 9} {'-' * 9} {'-' * 6}")

    def sla_rps(m):
        return m.success_count / duration_s

    def completed_rps(m):
        return m.completed / duration_s

    def print_row(m):
        print(
            f"  {m.policy_name:<38} "
            f"{sla_rps(m):>11,.0f} "
            f"{m.success_count:>8} "
            f"{m.success_rate:>6.1%} "
            f"{m.late_unserved:>6} "
            f"{m.avg_batch_size:>6.1f} "
            f"{m.p50_ms:>9.3f} "
            f"{m.p90_ms:>9.3f} "
            f"{m.p99_ms:>9.3f} "
            f"{m.avg_idle_ms:>6.1f}",
            flush=True,
        )

    results = []
    for policy in build(b_max):
        print(f"  · {policy.name()} ...", end="", flush=True)
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
        print(f" done (slaRPS={sla_rps(m):,.0f})", flush=True)

    # BASELINE pinned first; the rest ranked by SLA throughput, then p99.
    baseline = [m for m in results if m.policy_name.startswith("BASELINE")]
    rest = sorted((m for m in results if not m.policy_name.startswith("BASELINE")),
                  key=lambda x: (-sla_rps(x), x.p99_ms))
    for m in baseline + rest:
        print_row(m)


def run_baseline_search(mode="simple"):
    """Grid-search the best batch_size + timeout per scenario; update SCENARIO_BASELINES."""
    print(f"\n\n{'=' * 86}")
    print(f"  BASELINE SEARCH — one reference per scenario (ranked by slaRPS)")
    print(f"  coarse grid: {BASELINE_SEARCH_BATCH} x {BASELINE_SEARCH_TIMEOUT_MS}")
    print(f"{'=' * 86}")

    found = {}
    for scenario in SCENARIOS:
        name, rps, worst_case, duration_s, *rest = scenario
        opts = rest[0] if rest else {}
        sla_ms = opts.get("sla_ms", SLA_MS)
        b_max = opts.get("b_max", min(PARAMS.b_max_safe(sla_ms), MAX_BATCH_WEIGHT))
        rps_label = "burst(peak=2M)" if callable(rps) else str(rps)

        print(f"\n  --- {name}  RPS={rps_label}  dur={duration_s}s  SLA={sla_ms:.0f}ms ---")
        print(f"  {'batch':>6} {'timeout':>8} {'slaRPS':>12} {'Rate':>7} {'p99':>9}")

        batch_grid = [b for b in BASELINE_SEARCH_BATCH if b <= b_max]
        sim_ms = duration_s * 1000
        probe = _probe_rps(rps)
        rows = []
        for b in batch_grid:
            for t in BASELINE_SEARCH_TIMEOUT_MS:
                policy = BaselinePolicy(b, t, max_batch_size=b_max)
                sim = Simulator(
                    params=PARAMS, policy=policy,
                    rps=probe if callable(rps) else rps,
                    sla_ms=sla_ms, sim_duration_ms=sim_ms, seed=SEED,
                    worst_case=worst_case, mode=mode,
                )
                sim.run()
                m = compute_metrics(
                    policy_name=f"B={b},T={t}", all_requests=sim.all_requests,
                    sla_ms=sla_ms, batch_sizes=sim.batch_sizes, sim_end_ms=sim_ms,
                )
                rows.append((b, t, m))

        def sla_rps(m):
            return m.success_count / duration_s

        rows.sort(key=lambda x: (-sla_rps(x[2]), x[2].p99_ms))
        for b, t, m in rows[:8]:
            print(
                f"  {b:>6} {t:>8.1f} {sla_rps(m):>12,.0f} {m.success_rate:>6.1%} "
                f"{m.p99_ms:>9.3f}"
            )
        bb, bt, bm = rows[0][0], rows[0][1], rows[0][2]
        found[name] = (bb, bt)
        print(f"  >>> BEST: ({bb}, {bt})  slaRPS={sla_rps(bm):,.0f}  rate={bm.success_rate:.1%}")

    print(f"\n  Copy into SCENARIO_BASELINES:")
    print("  SCENARIO_BASELINES = {")
    for name, (bb, bt) in found.items():
        print(f'      "{name}": ({bb}, {bt}),')
    print("  }")
    return found


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
        run_baseline_search(mode=args.mode)
        print()
        return

    label = "TASK 2 (black box: only batch in/out times)" if args.mode == "advanced" \
        else "TASK 1 (known prepare/infer split)"
    print(f"Mode: {args.mode}  ->  {label}")
    print(f"Pipeline: prepare = {PARAMS.a1}*b + {PARAMS.c1}   "
          f"infer = {PARAMS.a2}*b + {PARAMS.c2}   variance=±{PARAMS.variance:.0%}")
    for scenario in SCENARIOS:
        name, rps, worst_case, duration_s, *rest = scenario
        opts = rest[0] if rest else {}
        run_scenario(name, rps, worst_case, duration_s, args.mode, opts=opts)
    print()


if __name__ == "__main__":
    main()
