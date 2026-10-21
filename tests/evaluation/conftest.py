# -*- coding: utf-8 -*-
"""Shared fixture exports for the unit / integration suites.

合成数据工厂按业务放在 ``helpers/typhoon.py``、``helpers/precipitation.py``
和 ``helpers/weather_field.py``，这里将它们暴露成共享 fixture。
"""

import pytest

from tests.evaluation.helpers import precipitation, typhoon


@pytest.fixture(scope="session")
def grid():
    """合成网格 ``(lat, lon)``：0.25°，所有设计位置都落在格点上。"""
    return typhoon.GRID_LAT.copy(), typhoon.GRID_LON.copy()


@pytest.fixture
def make_babj():
    """写 BABJ 报文：``make_babj(path, tcname, tcid, rows)``。

    ``rows``: ``(北京时 datetime, 时效 h, lon, lat, pmin_hpa, vmax_ms)``，
    时效 000 的行才是分析实况。
    """
    return typhoon.write_babj


@pytest.fixture
def make_det_forecast():
    """写确定性预报目录：``make_det_forecast(root, init_utc, centers)``。"""
    return typhoon.write_fuxi_single


@pytest.fixture
def make_ens_forecast():
    """写集合预报目录：``make_ens_forecast(root, init_utc, member_centers)``。"""
    return typhoon.write_fuxi_ens


@pytest.fixture
def make_station_obs():
    """写 Diamond 站点文件：``make_station_obs(path, moment, rows)``。

    ``rows``: ``(站号, 经度, 纬度, 海拔 m, 降水 mm)``；路径用 ``station_file``
    取分层布局（``root/YYYYMMDD/HH.000``）、``flat_station_file`` 取扁平布局。
    """
    return precipitation.write_diamond


@pytest.fixture
def station_file():
    """分层布局路径：``station_file(root, moment) -> root/YYYYMMDD/HH.000``。"""
    return precipitation.station_file


@pytest.fixture
def flat_station_file():
    """扁平布局路径：``flat_station_file(root, moment) -> root/YYYYMMDDHH.000``。"""
    return precipitation.flat_station_file
