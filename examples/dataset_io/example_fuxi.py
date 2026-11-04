"""读取 FuXi 集合预报，选择源变量并导出 NetCDF。"""

from pathlib import Path
from datakit import io


def main() -> None:
    source_path = Path("/data/fuxi")
    output_path = Path("/data/output/fuxi.nc")
    description = io.source_config("forecast_fuxi_ensemble")
    variables = ["z500", "t2m"]
    selection = {
        "init_time": ["2025-01-01"],
        "lead_time": [6, 12],
        "member": [0],
    }

    # FuXi 文件编号对应 6 小时间隔：001.nc、002.nc 分别为 6、12 小时预报。
    source = io.open(
        source_path,
        description,
        selection=selection,
    )
    with source.read() as dataset:
        data = dataset[variables]
        print(data)
        outputs = io.open(output_path).write(data)

    print("Written:", outputs)


if __name__ == "__main__":
    main()
