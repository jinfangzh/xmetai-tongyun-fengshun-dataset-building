"""Independent Fengqing reference values shared by generic and serial entry tests.

Values come from ensemble_verifier.py and activity_spectrum_verifier.py for
2026-08-20 00 UTC, leads 6/12/18 h, z500. Spectrum values average these leads;
wavenumber zero is excluded. Reference CSVs: fdp_test_data/ref_spectrum/.
REL_TOL=1e-4 accommodates float32 geopotential-to-height scaling and six-digit
spectrum CSV output. No reference is computed with the production metrics.
"""

from pathlib import Path

import pandas as pd

INIT_ISO = "2026-08-20T00:00:00.000000"


LEADS = (6, 12, 18)


GRID_POINTS = 721 * 1440


REL_TOL = 1e-4


ENS_COLUMNS = ("crps", "spread", "rmse", "spread_error_ratio")


ACTIVITY_COLUMNS = {
    "activity_ratio": "activity_ratio",
    "fc_activity": "activity_forecast",
    "obs_activity": "activity_observation",
}


GOLDEN_ENSEMBLE = {
    6: {
        "crps": 3.030698164066889,
        "spread": 2.350909948348999,
        "rmse": 4.886978626251221,
        "spread_error_ratio": 0.4810559096208635,
    },
    12: {
        "crps": 3.144235071187685,
        "spread": 2.8866982460021973,
        "rmse": 5.096440315246582,
        "spread_error_ratio": 0.566414608519266,
    },
    18: {
        "crps": 3.19647544233606,
        "spread": 3.446047067642212,
        "rmse": 5.442727565765381,
        "spread_error_ratio": 0.6331470803936176,
    },
}


GOLDEN_ACTIVITY = {
    6: {
        "activity_ratio": 0.9984731078147888,
        "fc_activity": 265.7129821777344,
        "obs_activity": 266.11932373046875,
    },
    12: {
        "activity_ratio": 0.9968142509460448,
        "fc_activity": 264.0156555175781,
        "obs_activity": 264.85943603515625,
    },
    18: {
        "activity_ratio": 0.9973190426826476,
        "fc_activity": 261.57635498046875,
        "obs_activity": 262.2795104980469,
    },
}


GOLDEN_SPECTRUM = {
    "forecast": (
        602057836131054.9,
        1.2488027576399048e17,
        807979311525485.4,
        1.6420130552952886e16,
        1010865679762588.9,
        2749974009958590.5,
        883489351224695.6,
        2199338762714431.2,
        384668495464970.7,
        1710851339258180.8,
        226638010045345.75,
        807108072970235.0,
        118580330260528.05,
        422046234790953.8,
        82967250180868.1,
        294094593373972.56,
        41559749690681.13,
        219543220651043.66,
        31973249144773.906,
        171883638838190.03,
        24354744430155.02,
        110617312461897.38,
        12245064642216.703,
        109701306054021.64,
        13317854388573.172,
        112036754775453.05,
        7307808410876.227,
        86425389406370.06,
        4899362090951.778,
        68379130231360.77,
    ),
    "observation": (
        590475791453331.9,
        1.256451034130574e17,
        801732460696704.5,
        1.6325055978468918e16,
        995167719105104.0,
        2683966357194221.0,
        880020156681322.6,
        2117717428008450.8,
        381894680673279.8,
        1686591012104204.0,
        220602211506857.66,
        794462269191437.6,
        121316389771244.62,
        417930748933442.2,
        86794081875033.77,
        295278230464185.06,
        41906289281946.836,
        215244837824897.66,
        33155361335004.8,
        167933403979950.34,
        26263735777208.625,
        107676127259620.23,
        12876831557649.65,
        108578799548685.48,
        14338496641093.963,
        110641032963001.56,
        7801803305996.925,
        86926841288547.14,
        5214079539673.633,
        67860116820273.62,
    ),
}


def score_value(scores: pd.DataFrame, metric: str, lead: int) -> float:
    """取 ``(metric, z500, lead)`` 那一格的 value。

    ``region`` 必须为空：档案里混着 tropics 行，不过滤就会取到别处的数。

    ens 配置的 ``rmse`` 会出两行——``fdp_field_scores`` 与 ``fdp_ens_crps`` 两段
    都声明了它，各算各的，实测能差到 3.7e-8（float32 累加次序，不是口径分叉；
    长表里没有哪一列能区分这两行）。所以只要求它们在容差内一致，真分叉了照样拦
    得住，然后取第一行。
    """
    rows = scores[
        (scores["metric"] == metric)
        & (scores["variable"] == "z500")
        & (scores["lead_h"] == float(lead))
        & scores["region"].isna()
    ]
    assert len(rows) >= 1, f"scores.csv 里没有 {metric} @ {lead}h"
    values = rows["value"].to_numpy(dtype=float)
    spread = (values.max() - values.min()) / abs(values[0])
    assert spread <= REL_TOL, (
        f"{metric} @ {lead}h 的两行互相矛盾（相对差 {spread:.3e}）: {values!r}"
    )
    return float(values[0])


def assert_relative(actual: float, expected: float, label: str) -> None:
    """相对差断言；``expected`` 是标准脚本的数。"""
    relative = abs(actual - expected) / abs(expected)
    assert relative <= REL_TOL, (
        f"{label}: 框架 {actual!r} vs 标准脚本 {expected!r}，相对差 {relative:.3e}"
    )


def assert_spectrum(out_dir: Path) -> None:
    """``diagnostics/spectrum_z500.csv`` 的 P1..P30 两列对表。

    框架那份 CSV 只写到 6 位有效数字，比不出比 1e-5 更细的东西——``REL_TOL`` 罩得住。
    """
    ours = pd.read_csv(out_dir / "diagnostics" / "spectrum_z500.csv")
    assert sorted(ours["wavenumber"]) == list(range(1, 31)), "诊断谱的波数不是 1..30"
    for kind, column in (("forecast", "pred_mean"), ("observation", "obs_mean")):
        expected = GOLDEN_SPECTRUM[kind]  # P1 在 [0]
        for k in range(1, 31):
            actual = float(ours.loc[ours["wavenumber"] == k, column].iloc[0])
            assert_relative(actual, expected[k - 1], f"谱 P{k}（{kind}）")


def assert_activity(scores: pd.DataFrame) -> None:
    for lead in LEADS:
        for column, metric in ACTIVITY_COLUMNS.items():
            assert_relative(
                score_value(scores, metric, lead),
                GOLDEN_ACTIVITY[lead][column],
                f"活跃度 {column} @ {lead}h",
            )
