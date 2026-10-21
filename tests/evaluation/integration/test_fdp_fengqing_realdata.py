# -*- coding: utf-8 -*-
"""FDP 要素检验（Fengqing × CRA40）真实数据档：与标准脚本产物对表。

跑两条真配置，产物与 ``D:\\xmetai-evalation\\fdp\\`` 下两个标准脚本的输出逐列比：

- ``ensemble_verifier.py``           → ``GOLDEN_ENSEMBLE``
- ``activity_spectrum_verifier.py``  → ``GOLDEN_ACTIVITY`` / ``GOLDEN_SPECTRUM``

标准答案统一保存在 ``tests/evaluation/helpers/fengqing_reference.py``。数值是
从标准脚本落下的三个 CSV 原样抄进来的，没做任何四舍五入；那三份 CSV 还在
``tmp_data/ref_spectrum/``（服务器上是 ``DATA_ROOT/ref_spectrum/``），是出处。

fixture 在 ``DATA_ROOT``（见下），是一份**裁剪过的最小样例**：起报 20260820、
时效 6/12/18；预报 21 成员、实况 3 个时刻、气候态是编的。目录不齐（或没装
netCDF4 / cfgrib）时相关用例跳过；可用 ``pytest -m realdata`` 单独选中。

对表容差 ``REL_TOL``：实测最差 6.8e-6。那点差全出在单位换算——框架
``datakit/io/readers/layouts.py:212`` 的 z500 写 ``scale=1.0/GRAVITY``（乘 float32 倒数），标准脚本
是 ``z / GRAVITY``（除），``float32(1/9.80665)`` 比 ``1/9.80665`` 单向偏低 2.4e-8，
于是整场带一个系统性尺度偏移。布局是 weather 系列共用的，不为一万分之一去动它，
所以容差放到 1e-4（比实测最差宽 15 倍），只拦真回归。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


from tests.evaluation.helpers.fengqing_reference import (
    ENS_COLUMNS,
    GOLDEN_ENSEMBLE,
    GRID_POINTS,
    INIT_ISO,
    LEADS,
    assert_activity,
    assert_relative,
    assert_spectrum,
    score_value,
)
from tests.evaluation.helpers.paths import fixture_directory

pytestmark = [pytest.mark.realdata, pytest.mark.usefixtures("fdp_field_data")]

DATA_ROOT = fixture_directory("fdp_test_data")


def _run(config_name: str, output_dir: Path) -> Path:
    """按真实配置跑一条链，只覆盖产物目录与并发形态。

    数据根覆盖为 ``XMETAI_FIXTURE_DIR`` 下的对应 fixture。
    并发写成 serial + ``resume: False``：测试必须每次从零算，否则会捡到上一次
    落在 ``.states/segNN_*.pkl`` 的状态，而那份状态**不校验配置指纹**，改了指标
    口径也会被静默复用。
    """
    from pipelines.evaluation.cli import run_evaluation
    from pipelines.evaluation.config import load_config

    cfg = load_config(config_name)
    assert not isinstance(cfg, list), f"{config_name} 应导出单个 cfg"
    cfg.forecast_reader = {**cfg.forecast_reader, "root_dir": str(DATA_ROOT / "fengqing")}
    cfg.observation_reader = {**cfg.observation_reader, "root_dir": str(DATA_ROOT / "cra_root")}
    cfg.reference_reader = {**cfg.reference_reader, "root_dir": str(DATA_ROOT / "cli_fake")}
    cfg.output_dir = str(output_dir)
    cfg.execution = {**cfg.execution, "mode": "serial", "n_workers": 1, "resume": False}
    run_evaluation(cfg)
    return output_dir


@pytest.fixture(scope="session")
def ens(tmp_path_factory):
    """ens 配置（field_scores + ens_crps + activity_spectrum 三段）跑一次，整模块共用。"""
    root = _run("fdp_rmse_ens_fengqing", tmp_path_factory.mktemp("ens"))
    return root, pd.read_csv(root / "scores.csv")


@pytest.fixture(scope="session")
def single(tmp_path_factory):
    """single 配置（field_scores + activity_spectrum 两段）跑一次，整模块共用。"""
    root = _run("fdp_rmse_single_fengqing", tmp_path_factory.mktemp("single"))
    return root, pd.read_csv(root / "scores.csv")


def test_ens_error_parity(ens):
    """ens：crps / spread / rmse / spread-error 比，四指标 × 三时效逐格对表。"""
    _, scores = ens
    for lead in LEADS:
        for column in ENS_COLUMNS:
            assert_relative(
                score_value(scores, column, lead),
                GOLDEN_ENSEMBLE[lead][column],
                f"ens {column} @ {lead}h",
            )


def test_ens_activity_parity(ens):
    """ens：活跃度比三列（ratio / forecast / observation）对表。"""
    assert_activity(ens[1])


def test_ens_spectrum_parity(ens):
    """ens：功率谱 P1..P30（预报 + 实况两列）对表。"""
    assert_spectrum(ens[0])


def test_single_parity(single, ens):
    """single：rmse + 活跃度 + 谱对表，且与 ens 的确定性指标逐位相同。

    single 少挂 ``fdp_ens_crps`` 段，crps / spread / spread_error_ratio 本来就没有；
    rmse 与谱是同一批样本上同一套公式，实测与 ens **逐位相同**（``array_equal``），
    这里把这个性质钉住——它保证"加不加 crps 段"不影响确定性指标。
    """
    out, scores = single
    ens_out, ens_scores = ens

    for lead in LEADS:
        ours = score_value(scores, "rmse", lead)
        assert_relative(ours, GOLDEN_ENSEMBLE[lead]["rmse"], f"single rmse @ {lead}h")
        assert ours == score_value(ens_scores, "rmse", lead), (
            f"single/ens 的 rmse @ {lead}h 不逐位相同"
        )

    assert "crps" not in set(scores["metric"]), "single 配置不该出 crps"
    assert_activity(scores)
    assert_spectrum(out)

    ours_spectrum = pd.read_csv(out / "diagnostics" / "spectrum_z500.csv")
    ens_spectrum = pd.read_csv(ens_out / "diagnostics" / "spectrum_z500.csv")
    for column in ("pred_mean", "obs_mean"):
        assert np.array_equal(ours_spectrum[column], ens_spectrum[column]), (
            f"single/ens 的诊断谱 {column} 不逐位相同"
        )


def test_scores_coverage(ens):
    """兜底：样本数、状态、以及 golden 之外那几个指标的完整性。

    golden 里没有 acc / bias / spectrum_power_ratio / activity_bias（标准脚本不算
    这些），对不了表；这里只保证它们出得来、覆盖全网格——漏算或静默出假数的回归
    至少能被这条拦住。
    """
    _, scores = ens
    assert set(scores["variable"]) == {"z500"}, "本配置只评 z500"
    assert scores["region"].isna().all(), "不该出现分区行"
    assert (scores["status"] == "success").all()
    assert set(scores["lead_h"]) == {float(lead) for lead in LEADS}

    # 逐格指标覆盖整张网格；``spectrum_power_ratio`` 是谱的汇总量（group=summary），
    # 一个时效一个样本，别按全网格去要求它。
    per_grid = scores[scores["metric"] != "spectrum_power_ratio"]
    assert (per_grid["n_valid"] == GRID_POINTS).all()
    summary = scores[scores["metric"] == "spectrum_power_ratio"]
    assert not summary.empty, "缺少 spectrum_power_ratio 汇总行"
    assert (summary["group"] == "summary").all() and (summary["n_valid"] == 1).all()

    # 时段与 fixture 对得上：起报 2026082000，有效时刻 0820 的 06/12/18。
    assert set(scores["init_time"]) == {INIT_ISO}
    assert set(scores["valid_time"]) == {f"2026-08-20T{lead:02d}:00:00.000000" for lead in LEADS}

    required = {
        "rmse",
        "bias",
        "acc",
        "crps",
        "spread",
        "spread_error_ratio",
        "activity_ratio",
        "activity_forecast",
        "activity_observation",
        "activity_bias",
    }
    assert required <= set(scores["metric"]), f"缺指标: {sorted(required - set(scores['metric']))}"

    # acc 用的是 FDP/WeatherBench2 的 uncentered 口径（配置里 centered=False），
    # 在编的气候态上会顶到接近 1；只要求它有值、不是常数零场兜底出来的。
    for lead in LEADS:
        assert 0.0 < score_value(scores, "acc", lead) <= 1.0


def main() -> None:
    """直接运行一条配置，方便在评估链中设置断点。"""
    parser = argparse.ArgumentParser(description="调试 Fengqing × CRA40 真实数据评估")
    parser.add_argument("--config", choices=("ens", "single"), default="ens")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_name = f"fdp_rmse_{args.config}_fengqing"
    print(f"准备运行 {config_name}，数据目录: {DATA_ROOT}", flush=True)
    missing = [
        name for name in ("fengqing", "cra_root", "cli_fake") if not (DATA_ROOT / name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"fixture 不存在: {DATA_ROOT}（缺 {', '.join(missing)}/）")

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="fdp_fengqing_debug_"))
    print(f"输出目录: {output_dir}", flush=True)
    _run(config_name, output_dir)


if __name__ == "__main__":
    main()
