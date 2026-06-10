import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    PARAMS, SEED, SLA_MS, MAX_BATCH_WEIGHT, format_duration_s,
    filter_scenarios, add_scenario_args, scenario_names_from_args,
)
from baseline_search import resolve_baseline, run_baseline_search
from metrics import compute_metrics, sla_rps
from simulator import Simulator
from simple_policies.registry import build_simple, build_advanced


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
    extra = "  hybrid_drain=off" if not hybrid_early_drain else ""

    print(f"\n\n{'=' * 86}")
    print(f"  SCENARIO {name}   RPS={rps_label}  dur={format_duration_s(duration_s)}s  "
          f"SLA={sla_ms:.0f}ms  "
          f"time={mult}  b_max={b_max}  mode={mode}{extra}")
    print(f"  reference: BASELINE(B={baseline_batch}, T={baseline_timeout_ms:.1f}ms)")
    print(f"{'=' * 86}")

    if mode == "advanced":
        policies = build_advanced(b_max, baseline_batch, baseline_timeout_ms)
    else:
        policies = build_simple(b_max, baseline_batch, baseline_timeout_ms,
                                hybrid_early_drain=hybrid_early_drain)

    results = []
    for policy in policies:
        print(f"  · {policy.name()} ...", end="", flush=True)
        sim = Simulator(
            params=PARAMS, policy=policy, rps=rps, sla_ms=sla_ms,
            sim_duration_ms=sim_ms, seed=SEED, worst_case=worst_case, mode=mode,
        )
        sim.run()
        m = compute_metrics(
            policy.name(), sim.all_requests, sla_ms,
            sim.batch_sizes, sim_ms, dropped_requests=sim.dropped_requests,
        )
        results.append(m)
        print(f" done (slaRPS={sla_rps(m, duration_s):,.0f})", flush=True)

    baseline = [m for m in results if m.policy_name.startswith("BASELINE")]
    rest = sorted((m for m in results if not m.policy_name.startswith("BASELINE")),
                  key=lambda x: (-sla_rps(x, duration_s), x.p99_ms))

    print(f"  {'Policy':<38} {'slaRPS':>11} {'Success':>8} {'Rate':>7} {'Late':>6} "
          f"{'Drop':>6} {'batch':>6} {'p50':>9} {'p90':>9} {'p99':>9} {'idle':>6}")
    print(f"  {'-' * 38} {'-' * 11} {'-' * 8} {'-' * 7} {'-' * 6} "
          f"{'-' * 6} {'-' * 6} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 6}")
    for m in baseline + rest:
        print(
            f"  {m.policy_name:<38} "
            f"{sla_rps(m, duration_s):>11,.0f} "
            f"{m.success_count:>8} "
            f"{m.success_rate:>6.1%} "
            f"{m.late_unserved:>6} "
            f"{m.dropped:>6} "
            f"{m.avg_batch_size:>6.1f} "
            f"{m.p50_ms:>9.3f} "
            f"{m.p90_ms:>9.3f} "
            f"{m.p99_ms:>9.3f} "
            f"{m.avg_idle_ms:>6.1f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="GPU pipeline batching emulator")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--simple", action="store_const", dest="mode", const="simple",
                       help="task 1: policies know prepare/infer split (default)")
    group.add_argument("--advanced", action="store_const", dest="mode", const="advanced",
                       help="task 2: black-box, only batch entry/exit timing is known")
    parser.add_argument("--baseline", action="store_true",
                        help="phase 0: grid-search the best batch_size+timeout reference")
    add_scenario_args(parser)
    parser.set_defaults(mode="simple")
    args = parser.parse_args()
    scenarios = filter_scenarios(scenario_names_from_args(args))

    pipeline = (f"Pipeline: prepare = {PARAMS.a1}*b + {PARAMS.c1}   "
                f"infer = {PARAMS.a2}*b + {PARAMS.c2}   variance=±{PARAMS.variance:.0%}")

    if args.baseline:
        print(pipeline)
        run_baseline_search(mode=args.mode, scenario_names=scenario_names_from_args(args))
        print()
        return

    label = ("TASK 2 (black box: only batch in/out times)" if args.mode == "advanced"
             else "TASK 1 (known prepare/infer split)")
    print(f"Mode: {args.mode}  ->  {label}")
    print(pipeline)
    for scenario in scenarios:
        name, rps, worst_case, duration_s, *rest = scenario
        opts = rest[0] if rest else {}
        run_scenario(name, rps, worst_case, duration_s, args.mode, opts=opts)
    print()


if __name__ == "__main__":
    main()
