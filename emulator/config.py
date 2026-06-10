"""Simulation parameters, scenarios, and baseline reference table."""
from models import PipelineParams

# Model in microseconds, converted to ms:
#   prepare = 0.149*b + 600 us,  infer = 0.2640*b + 404.52 us
PARAMS = PipelineParams(
    a1=0.149 / 1000.0, c1=600.0 / 1000.0,
    a2=0.2640 / 1000.0, c2=404.52 / 1000.0,
    variance=0.2,
)

SEED = 42
SLA_MS = 30000.0 / 1000.0
MAX_BATCH_WEIGHT = 4096

# Shorter sim wall time; RPS unchanged → same load/capacity ratio, ~1/SCALE fewer events.
SCENARIO_TIME_SCALE = 1.0 / 3.0


def format_duration_s(duration_s: float) -> str:
    """Human-readable sim duration (avoids 0.09999999999999999 artifacts)."""
    return f"{round(duration_s, 4):g}"

BATCH_GRID = [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
TIMEOUT_GRID_MS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
BASELINE_SEARCH_BATCH = [32, 128, 512, 1024, 2048, 4096]
BASELINE_SEARCH_TIMEOUT_MS = [
    0.5,
    0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
    2.0, 2.25, 2.5, 2.75,
    3.0, 3.25, 3.5, 3.75,
    4.0,
    5.0, 10.0, 20.0,
]

# Per-scenario reference (batch_size, timeout_ms). Re-tune: python run.py --baseline
SCENARIO_BASELINES = {
    "light":      (32,   0.5),
    "loaded":     (4096, 1.4),
    "bursty":     (2048, 1.0),
    "overloaded": (4096, 1.0),
    "tight_sla":  (4096, 1.5),
}


def burst(base, peak, base_s, peak_s):
    period_ms = (base_s + peak_s) * 1000.0
    hi_after_ms = base_s * 1000.0

    def rate(t_ms):
        return peak if (t_ms % period_ms) >= hi_after_ms else base

    return rate


_T = SCENARIO_TIME_SCALE

# success_rate excludes arrivals in the last SLA window; overloaded needs ≥5×SLA wall time.
_OVERLOADED_DUR_S = max(0.15 * _T, 5.0 * SLA_MS / 1000.0)

SCENARIOS = [
    ("light",      50_000,    False, 1.0 * _T),
    ("loaded",     3_000_000, False, 0.3 * _T),
    # Burst phase timing unscaled: only sim duration shrinks, not per-cycle shape.
    ("bursty",     burst(400_000, 2_000_000, 0.03, 0.015), False, 0.3 * _T),
    ("overloaded", 4_000_000, False, _OVERLOADED_DUR_S),
    ("tight_sla",  2_850_000, False, 0.3 * _T, {
        "sla_ms": 6.0,
        "hybrid_early_drain": False,
    }),
]

DEFAULT_SCENARIO_NAMES = ("loaded", "overloaded", "tight_sla")


def filter_scenarios(names=None):
    """None or empty → default set (no light/bursty). Unknown names ignored."""
    chosen = names if names else DEFAULT_SCENARIO_NAMES
    allow = set(chosen)
    return [s for s in SCENARIOS if s[0] in allow]


def add_scenario_args(parser):
    parser.add_argument("--light", action="store_true", help="run light scenario")
    parser.add_argument("--loaded", action="store_true", help="run loaded scenario")
    parser.add_argument("--bursty", action="store_true", help="run bursty scenario")
    parser.add_argument("--overloaded", action="store_true", help="run overloaded scenario")
    parser.add_argument("--tight-sla", dest="tight_sla", action="store_true",
                        help="run tight_sla scenario")


def scenario_names_from_args(args):
    flags = {
        "light": args.light,
        "loaded": args.loaded,
        "bursty": args.bursty,
        "overloaded": args.overloaded,
        "tight_sla": args.tight_sla,
    }
    picked = [name for name, on in flags.items() if on]
    return picked if picked else None
