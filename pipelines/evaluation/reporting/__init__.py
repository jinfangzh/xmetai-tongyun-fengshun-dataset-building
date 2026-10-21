# -*- coding: utf-8 -*-
"""Evaluation reporting: result tables, optional views, and report files.

对外入口：
    build_tables(results, ...) -> ResultTables   把 MetricResult 展平成统一长表
    ReportWriter(...).write(results, ...)        把长表与运行记录落盘
    build_report(output_dir, ...)               按产物类型生成统计报告与图表
"""

from pipelines.evaluation.reporting.writer import ReportWriter, RunContext
from pipelines.evaluation.reporting.report import build_report, detect_capability
from pipelines.evaluation.reporting.tables import (
    COVERAGE_COLUMNS,
    DETAIL_COLUMNS,
    SCORE_COLUMNS,
    ResultTables,
    build_tables,
    categorical_wide,
)

__all__ = [
    "COVERAGE_COLUMNS",
    "DETAIL_COLUMNS",
    "SCORE_COLUMNS",
    "ResultTables",
    "build_tables",
    "build_report",
    "detect_capability",
    "categorical_wide",
    "ReportWriter",
    "RunContext",
]
