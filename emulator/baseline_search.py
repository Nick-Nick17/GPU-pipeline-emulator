"""Grid-search for the reference batch_size + timeout policy."""
from config import (
    PARAMS, SEED, SLA_MS, MAX_BATCH_WEIGHT, SCENARIOS, SCENARIO_BASELINES,
    BASELINE_SEARCH_BATCH, BASELINE_SEARCH_TIMEOUT_MS,
)
from metrics import compute_metrics, sla_rps
from simulator import Simulator
from simple_policies.reference import BaselinePolicy


def probe_rps(rps):
    return 2_000_000 if callable(rps) else rps


def resolve_baseline(name, rps, worst_case, duration_s, sla_ms, b_max, mode, opts):
    if "baseline" in opts:
        return opts["baseline"]
    if name in SCENARIO_BASELINES:
        return SCENARIO_BASELINES[name]
    print(f"  (no cached baseline for {name}, searching...)", flush=True)
    bb, bt, _ = search_baseline(name, rps, worst_case, duration_s, sla_ms, b_max, mode)
    return bb, bt


def search_baseline(name, rps, worst_case, duration_s, sla_ms, b_max, mode,
                    batch_grid=None, timeout_grid=None):
    batch_grid = batch_grid or [b for b in BASELINE_SEARCH_BATCH if b <= b_max]
    timeout_grid = timeout_grid or BASELINE_SEARCH_TIMEOUT_MS
    sim_ms = duration_s * 1000
    probe = probe_rps(rps)
    best = None

    for b in batch_grid:
        for t in timeout_grid:
            policy = BaselinePolicy(b, t, max_batch_size=b_max)
            sim = Simulator(
                params=PARAMS, policy=policy,
                rps=probe if callable(rps) else rps,
                sla_ms=sla_ms, sim_duration_ms=sim_ms, seed=SEED,
                worst_case=worst_case, mode=mode,
            )
            sim.run()
            m = compute_metrics(
                policy.name(), sim.all_requests, sla_ms,
                sim.batch_sizes, sim_ms,
            )
            key = (sla_rps(m, duration_s), m.success_rate, -m.p99_ms)
            if best is None or key > best[0]:
                best = (key, b, t, m)
    _, bb, bt, bm = best
    return bb, bt, bm


def run_baseline_search(mode="simple"):
    print(f"\n\n{'=' * 86}")
    print("  BASELINE SEARCH — one reference per scenario (ranked by slaRPS)")
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
        probe = probe_rps(rps)
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
                    f"B={b},T={t}", sim.all_requests, sla_ms,
                    sim.batch_sizes, sim_ms,
                )
                rows.append((b, t, m))

        rows.sort(key=lambda x: (-sla_rps(x[2], duration_s), x[2].p99_ms))
        for b, t, m in rows[:8]:
            print(
                f"  {b:>6} {t:>8.1f} {sla_rps(m, duration_s):>12,.0f} "
                f"{m.success_rate:>6.1%} {m.p99_ms:>9.3f}"
            )
        bb, bt, bm = rows[0][0], rows[0][1], rows[0][2]
        found[name] = (bb, bt)
        print(f"  >>> BEST: ({bb}, {bt})  slaRPS={sla_rps(bm, duration_s):,.0f}  "
              f"rate={bm.success_rate:.1%}")

    print("\n  Copy into SCENARIO_BASELINES:")
    print("  SCENARIO_BASELINES = {")
    for name, (bb, bt) in found.items():
        print(f'      "{name}": ({bb}, {bt}),')
    print("  }")
    return found
