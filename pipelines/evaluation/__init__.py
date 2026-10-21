"""Evaluation pipeline declarations, protocols, and execution.

Runner 走模块级 ``__getattr__`` 惰性导入：执行层（``execution/``）要用
``datakit.evaluation.matching``，而 runner 又要导入执行层，急切导入会形成环。
"""

from datakit.evaluation.matching import Matcher
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
