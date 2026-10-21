"""指标计算模块。

指标只负责"算什么差异"：输入 EvaluationBatch，输出可合并状态与 MetricResult。
"""

from datakit.evaluation.metrics.acc import ACC
from datakit.evaluation.metrics.base import Metric, MetricRequirements, MetricState
from datakit.evaluation.metrics.bias import Bias
from datakit.evaluation.metrics.categorical import ContingencyTable, TSScore
from datakit.evaluation.metrics.ensemble import CRPS, SpreadError
from datakit.evaluation.metrics.probabilistic import EnsembleProbabilityScore
from datakit.evaluation.metrics.rmse import RMSE

__all__ = [
    "Metric",
    "MetricRequirements",
    "MetricState",
    "ContingencyTable",
    "RMSE",
    "Bias",
    "ACC",
    "TSScore",
    "EnsembleProbabilityScore",
    "CRPS",
    "SpreadError",
]
