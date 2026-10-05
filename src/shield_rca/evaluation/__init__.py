"""CyberHeal AI / TRINETRA Evaluation Package.

Multi-class metrics, SIH classification benchmarks, baseline models,
and unseen attack generalization frameworks.
"""

from .dataset_adapters import (
    DataQualityReport,
    DatasetAdapter,
    DatasetProvenance,
    CicIds2018Adapter,
    Ctu13Adapter,
    get_dataset_adapter,
)
from .logistic_baseline import (
    ClassificationMetricsReport,
    LogisticBaselineConfig,
    LogisticRegressionBaseline,
    generate_synthetic_classification_dataset,
)
from .metrics import compute_multiclass_metrics
from .real_data_benchmark import (
    RealDataBenchmarkRunner,
    discover_dataset,
)
from .unseen_attack import (
    UnseenAttackConfig,
    UnseenAttackDataset,
    UnseenAttackEvaluator,
    UnseenExperimentResult,
    generate_synthetic_unseen_attack_dataset,
)

__all__ = [
    "compute_multiclass_metrics",
    "LogisticBaselineConfig",
    "ClassificationMetricsReport",
    "LogisticRegressionBaseline",
    "generate_synthetic_classification_dataset",
    "UnseenAttackConfig",
    "UnseenAttackDataset",
    "UnseenAttackEvaluator",
    "UnseenExperimentResult",
    "generate_synthetic_unseen_attack_dataset",
    # Real-Data Benchmark (Phase 9)
    "DatasetAdapter",
    "CicIds2018Adapter",
    "Ctu13Adapter",
    "get_dataset_adapter",
    "DataQualityReport",
    "DatasetProvenance",
    "RealDataBenchmarkRunner",
    "discover_dataset",
]
