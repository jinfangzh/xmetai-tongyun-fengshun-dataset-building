"""Explicit selection of supported product definitions, without discovery."""

from .standard_dataset import DatasetProduct
from .cra_c71 import PRODUCT as CRA_C71
from .cra_c87 import PRODUCT as CRA_C87
from .era5_c78 import PRODUCT as ERA5_C78
from .era5_c87 import PRODUCT as ERA5_C87
from .era5_c116 import PRODUCT as ERA5_C116
from .era5_land_c8 import PRODUCT as ERA5_LAND_C8
from .s2s_c76 import PRODUCT as S2S_C76
from .oras5_c88 import PRODUCT as ORAS5_C88, CURRENT_PRODUCT as ORAS5_C88_CURRENT

STANDARD_PRODUCTS: tuple[DatasetProduct, ...] = (
    CRA_C71,
    CRA_C87,
    ERA5_C78,
    ERA5_C87,
    ERA5_C116,
    ERA5_LAND_C8,
    ORAS5_C88,
    S2S_C76,
    ORAS5_C88_CURRENT,
)


def get_dataset_product(product_id: str, version: str = "1.0") -> DatasetProduct:
    for product in STANDARD_PRODUCTS:
        if (product.product_id, product.version) == (product_id, version):
            return product
    raise ValueError(f"Unknown product/version: {product_id}/{version}")
