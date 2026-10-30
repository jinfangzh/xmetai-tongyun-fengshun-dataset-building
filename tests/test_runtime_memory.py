"""Memory units, platform selection and unavailable-measurement behavior."""

import sys
from types import SimpleNamespace

import pytest

from datakit.runtime import memory


@pytest.mark.parametrize("platform, expected", [("linux", 2048), ("darwin", 2)])
def test_peak_memory_platform_units(monkeypatch, platform, expected):
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=2)),
    )
    assert memory.peak_memory_bytes() == expected


def test_linux_resident_memory_uses_current_pages(monkeypatch):
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(memory.Path, "read_text", lambda _: "100 12 0 0")
    monkeypatch.setattr(memory.os, "sysconf", lambda _: 4096)
    assert memory.resident_memory_bytes() == 12 * 4096


@pytest.mark.parametrize("contents", ["", "100 invalid"])
def test_unavailable_linux_measurement_does_not_use_peak(monkeypatch, contents):
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(memory.Path, "read_text", lambda _: contents)
    assert memory.resident_memory_bytes() is None


def test_windows_current_and_peak_use_distinct_counters(monkeypatch):
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setitem(sys.modules, "resource", None)
    counters = {"WorkingSetSize": 128, "PeakWorkingSetSize": 4096}
    monkeypatch.setattr(memory, "_windows_memory_bytes", counters.get)
    assert memory.resident_memory_bytes() == 128
    assert memory.peak_memory_bytes() == 4096


@pytest.mark.parametrize("peak, expected", [(None, "n/a"), (2 * 1024**3, "2.00 GB")])
def test_peak_log_uses_shared_byte_measurement(monkeypatch, peak, expected):
    monkeypatch.setattr(memory, "peak_memory_bytes", lambda: peak)
    assert memory.peak_rss_text() == expected


def test_peak_log_handles_measurement_errors(monkeypatch):
    def unavailable():
        raise OSError("measurement unavailable")

    monkeypatch.setattr(memory, "peak_memory_bytes", unavailable)
    assert memory.peak_rss_text() == "n/a"
