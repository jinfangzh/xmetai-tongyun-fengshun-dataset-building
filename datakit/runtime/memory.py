"""Process memory information for diagnostic logs."""

import sys


def peak_rss_text() -> str:
    """Format the current process's peak resident memory for logs."""
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return "n/a"
    if sys.platform == "darwin":
        return f"{rss / 1024 ** 3:.2f} GB"
    return f"{rss / 1024 ** 2:.2f} GB"
