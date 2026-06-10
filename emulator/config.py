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

BATCH_GRID = [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
TIMEOUT_GRID_MS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
BASELINE_SEARCH_BATCH = [32, 128, 512, 1024, 2048, 4096]
BASELINE_SEARCH_TIMEOUT_MS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Per-scenario reference (batch_size, timeout_ms). Re-tune: python run.py --baseline
SCENARIO_BASELINES = {
    "light":      (32,   0.5),
    "loaded":     (4096, 2.0),
    "bursty":     (2048, 1.0),
    "overloaded": (4096, 1.0),
    "tight_sla":  (4096, 2.0),
}


def burst(base, peak, base_s, peak_s):
    period_ms = (base_s + peak_s) * 1000.0
    hi_after_ms = base_s * 1000.0

    def rate(t_ms):
        return peak if (t_ms % period_ms) >= hi_after_ms else base

    return rate


SCENARIOS = [
    ("light",      50_000,                              False, 1.0),
    ("loaded",     3_000_000,                           False, 0.3),
    ("bursty",     burst(400_000, 2_000_000, 0.03, 0.015), False, 0.3),
    ("overloaded", 4_000_000,                           False, 0.15),
    ("tight_sla",  2_850_000,                           False, 0.3, {
        "sla_ms": 6.0,
        "hybrid_early_drain": False,
    }),
]
