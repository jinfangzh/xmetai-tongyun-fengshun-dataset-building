"""Evaluation pipeline declarations, protocols, and execution.

Runner 通过模块级 ``__getattr__`` 惰性导入，避免导入数据源或匹配模块时
同时加载整套执行层和报告组件。
"""

from pipelines.evaluation.matching import Matcher
from pipelines.evaluation.spec import (
    MetricSpec,
    PipelineSpec,
    SourceSpec,
    TransformSpec,
)

__all__ = ["Matcher", "Runner", "PipelineSpec", "SourceSpec", "TransformSpec", "MetricSpec"]


def __getattr__(name: str):
    if name == "Runner":
        from pipelines.evaluation.runner import Runner

        return Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
