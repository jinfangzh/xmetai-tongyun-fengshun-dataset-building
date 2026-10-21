"""绘图样式和图片保存；不依赖任何报告类型。"""

from pathlib import Path

_CJK_FONTS = (
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti TC",
    "Arial Unicode MS",
)

MODEL_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b")


def setup_chinese_font() -> str | None:
    """选择可用的中文字体；返回选中的字体名（没有则 None）。"""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    available = {font.name for font in fm.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def configure_plotting(style: str | None = None) -> str | None:
    """设置统一字体、分辨率和裁边，返回选中的中文字体。"""
    import matplotlib.pyplot as plt

    font = setup_chinese_font()
    if style:
        plt.style.use(style)
        setup_chinese_font()
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    return font


def save_figure(fig, save_path: Path | None) -> None:
    """有输出路径时保存图片，并创建所需目录。"""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"已保存: {save_path}")
