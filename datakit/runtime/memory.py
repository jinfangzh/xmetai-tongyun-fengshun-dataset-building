"""Current and peak process memory measurements in bytes, plus log formatting."""

import os
import sys
from pathlib import Path


def _windows_memory_bytes(field: str) -> int | None:
    """Read one working-set measurement from the Windows process counters."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        memory_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        memory_info.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        memory_info.restype = wintypes.BOOL
        if memory_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(getattr(counters, field))
    except (AttributeError, OSError):
        pass
    return None


def resident_memory_bytes() -> int | None:
    """Current RSS, independent of allocations released by previous tasks.

    Linux and Windows have direct native readers. Other systems can use psutil
    when installed; unavailable measurements stay unknown, never a lifetime peak.
    """
    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None
    if sys.platform == "win32":
        return _windows_memory_bytes("WorkingSetSize")
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def peak_memory_bytes() -> int | None:
    """Lifetime process peak for diagnostics, not a current-run budget gate."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        return _windows_memory_bytes("PeakWorkingSetSize")


def peak_rss_text() -> str:
    """Format the current process's peak resident memory for logs."""
    try:
        peak = peak_memory_bytes()
    except Exception:
        return "n/a"
    return "n/a" if peak is None else f"{peak / 1024 ** 3:.2f} GB"
