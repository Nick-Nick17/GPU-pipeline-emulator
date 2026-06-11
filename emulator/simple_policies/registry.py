"""Policy lists used in benchmark runs."""
from advanced_policies import (
    BlackBoxSLAOverlapPolicy,
    LatencyFeedbackPolicy,
    ThroughputMatchPolicy,
)
from .reference import BaselinePolicy
from .baselines import SLABudgetPolicy, FixedSizePolicy, TimeoutBatchPolicy
from .overlap import (
    HybridSLAOverlapPolicy,
    HybridSLAOverlapLegacyPolicy,
    PredictiveOverlapPolicy,
    QueueFeedbackPolicy,
    OptimalOverlapPolicy,
)


def build_simple(b_max, baseline_batch, baseline_timeout_ms, hybrid_early_drain=True):
    hybrid_kw = {"early_drain": hybrid_early_drain}
    return [
        BaselinePolicy(baseline_batch, baseline_timeout_ms, max_batch_size=b_max),
        BaselinePolicy(baseline_batch, baseline_timeout_ms, max_batch_size=b_max,
                       drop_expired=True),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max, **hybrid_kw),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max,
                               shed_hopeless=True, **hybrid_kw),
        # count-based admission (legacy)
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max,
                               shed_hopeless=True, admit_infer=True,
                               max_committed_auto=True, overlap_optimistic=True,
                               **hybrid_kw),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max,
                               shed_hopeless=True, admit_infer=True,
                               max_committed_auto=True, overlap_optimistic=False,
                               **hybrid_kw),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max,
                               shed_hopeless=True, admit_infer=True,
                               max_committed_batches=1, **hybrid_kw),
        # prepare-queue time admission (dynamic cost)
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max,
                               shed_hopeless=True, admit_infer=True,
                               max_committed_auto=True, overlap_optimistic=True,
                               prepare_queue_admit=True, **hybrid_kw),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max,
                               shed_hopeless=True, admit_infer=True,
                               max_committed_auto=True, overlap_optimistic=False,
                               prepare_queue_admit=True, **hybrid_kw),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max,
                               shed_hopeless=True, admit_infer=True,
                               max_committed_batches=1, prepare_queue_admit=True,
                               **hybrid_kw),
        HybridSLAOverlapPolicy(safety=1.0, collect_ms=2.0, max_batch_size=b_max, **hybrid_kw),
        HybridSLAOverlapPolicy(safety=1.2, collect_ms=0.0, max_batch_size=b_max, **hybrid_kw),
        HybridSLAOverlapLegacyPolicy(safety=1.0, collect_ms=0.0, max_batch_size=b_max, **hybrid_kw),
        HybridSLAOverlapLegacyPolicy(safety=1.0, collect_ms=2.0, max_batch_size=b_max, **hybrid_kw),
        PredictiveOverlapPolicy(alpha=0.2, margin=1.1, safety=1.0, max_batch_size=b_max),
        QueueFeedbackPolicy(k=0.5, b_min=1, max_batch_size=b_max),
        SLABudgetPolicy(safety=1.0, max_batch_size=b_max),
        FixedSizePolicy(target_size=512, max_wait_ms=5.0, max_batch_size=b_max),
        TimeoutBatchPolicy(timeout_ms=5.0, max_batch_size=b_max),
        OptimalOverlapPolicy(safety=1.0, max_batch_size=b_max),
        OptimalOverlapPolicy(safety=1.2, max_batch_size=b_max),
    ]


def build_advanced(b_max, baseline_batch, baseline_timeout_ms):
    return [
        BaselinePolicy(baseline_batch, baseline_timeout_ms, max_batch_size=b_max),
        BaselinePolicy(baseline_batch, baseline_timeout_ms, max_batch_size=b_max,
                       drop_expired=True),
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
