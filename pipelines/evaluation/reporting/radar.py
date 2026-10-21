"""多模型雷达图：共享归一化规则与绘图，缺值轴整根剔除。"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from ._plotting import MODEL_COLORS, configure_plotting, save_figure


def model_colors(names: Sequence[str]) -> dict[str, str]:
    """按模型出现次序取色；超过调色板长度就循环。"""
    return {
        str(name): MODEL_COLORS[index % len(MODEL_COLORS)]
        for index, name in enumerate(names)
    }


def _is_number(value) -> bool:
    """能不能当有限浮点数用（``None`` / ``NaN`` / 非数值串都算不能）。

    雷达图拿它判「这根轴这个模型有没有值」——缺值要整根剔除，
    不能当 0 画（那会把模型画成某一项能力为零，是另一回事）。
    """
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def radar_scores(
    raw: Mapping[str, Mapping[str, float]],
    axes: Sequence[str],
    higher_is_better: Mapping[str, bool],
) -> dict[str, dict[str, float]]:
    """把各模型的**原始**综合评分折算成雷达图用的**批内 min-max** 得分。

    每根轴上 ``1 = 本批最好``、``0 = 本批最差``，方向由 ``higher_is_better``
    定（缺省按"越低越好"，与 RMSE / 偏差这类指标的惯例一致）。

    这是**相对分**，不是绝对能力分：两个模型时必然是 1 和 0。报告里必须写明。

    两个退化情形：

    - 某根轴全批打平（极差为 0）→ 一律给 1.0。除零会炸，而且"大家一样"
      本来就该画在满格上，画成 0 会让多边形凭空凹进去；
    - 某根轴只要有模型缺值 → **整根剔除**（不出现在返回值的键里）。
      雷达图没有断点，少画一个顶点会把多边形拉歪。

    抽成模块级函数是**故意的**：绘图（:func:`plot_capability_radar`）
    与正文的判读句都要用这两个退化规则，各写一份迟早会漂。
    """
    names = [str(name) for name in raw]
    kept = [
        str(axis)
        for axis in axes
        if all(_is_number(raw.get(name, {}).get(axis)) for name in names)
    ]
    scores: dict[str, dict[str, float]] = {name: {} for name in names}
    for axis_name in kept:
        values = {name: float(raw[name][axis_name]) for name in names}
        low, high = min(values.values()), max(values.values())
        span = high - low
        for name, value in values.items():
            if span <= 0:
                scores[name][axis_name] = 1.0
            elif higher_is_better.get(axis_name, False):
                scores[name][axis_name] = (value - low) / span
            else:
                scores[name][axis_name] = (high - value) / span
    return scores


def plot_capability_radar(
    raw: Mapping[str, Mapping[str, float]],
    axes: Sequence[str],
    higher_is_better: Mapping[str, bool],
    *,
    suptitle: str | None = None,
    save_path: Path | None = None,
) -> dict[str, dict[str, float]]:
    """多模型能力雷达图；返回各模型的**归一化**得分。

    ``raw`` 是 ``{模型名: {轴名: 原始值}}``，``axes`` 给轴序。

    归一化是**批内 min-max**：每个模型在每根轴上得 ``1 = 本批最好``、
    ``0 = 本批最差``，方向由 ``higher_is_better`` 定（缺省按"越低越好"，
    与 RMSE / 偏差这类指标的惯例一致）。这是**相对分**——两个模型时
    必然是 1 和 0，不代表绝对能力，报告里必须把这句话写出来。

    两个退化情形（全批打平给满格、缺值整根剔除）见 :func:`radar_scores`，
    归一化只有那一处实现。
    """
    import matplotlib.pyplot as plt

    configure_plotting()
    names = [str(name) for name in raw]
    scores = radar_scores(raw, axes, higher_is_better)
    # 归一化后还留在结果里的，就是真正画出来的轴（缺值的已被剔掉）
    kept = (
        [str(axis) for axis in axes if str(axis) in scores[names[0]]] if names else []
    )

    if not kept:
        return scores

    count = len(kept)
    angles = [index / count * 2 * math.pi for index in range(count)]
    closed = angles + angles[:1]
    colors = model_colors(names)

    figure = plt.figure(figsize=(7.2, 7.2))
    axis = figure.add_subplot(111, polar=True)
    axis.set_theta_offset(math.pi / 2)  # 第一根轴摆在正上方
    axis.set_theta_direction(-1)  # 顺时针排，读起来跟表同序
    axis.set_xticks(angles)
    axis.set_xticklabels(kept, fontsize=9)
    # 轴名往外挪一点：极坐标默认贴着外圈画，五六个字的中文标签会压到圆周上
    axis.tick_params(axis="x", pad=10)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="0.45")
    axis.grid(alpha=0.35, linewidth=0.6)

    for name in names:
        ring = [scores[name][each] for each in kept]
        axis.plot(
            closed, ring + ring[:1], color=colors[name], linewidth=1.8, label=name
        )
        axis.fill(closed, ring + ring[:1], color=colors[name], alpha=0.12)

    axis.legend(
        loc="upper right", bbox_to_anchor=(1.22, 1.08), fontsize=9, frameon=False
    )
    if suptitle:
        # 顶部那根轴的名字就压在 suptitle 正下方，不留出这段空隙会叠在一起
        figure.subplots_adjust(top=0.86)
        figure.suptitle(suptitle, fontsize=13)
    save_figure(figure, save_path)
    # 返回**得分**而不是 figure：调用方要靠它写图注与判读句，
    # 图本身落盘就够了（其余几个 plot_* 返回 figure 但调用方一律不用）。
    return scores
