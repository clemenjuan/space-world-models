"""Compatibility shims for stable_pretraining 0.1.7 in this environment."""


def configure_utf8_stdio():
    """Make Windows console streams tolerate stable_pretraining's Unicode logs."""
    import os
    import sys

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def patch_pyarrow_for_legacy_datasets():
    """Restore the pyarrow alias expected by datasets 2.14.x, if needed."""
    try:
        import pyarrow as pa
    except ImportError:
        return
    if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
        pa.PyExtensionType = pa.ExtensionType


def patch_stable_pretraining_windows_signals():
    """Guard stable_pretraining signal logging on Windows."""
    import signal

    if all(hasattr(signal, name) for name in ("SIGUSR1", "SIGUSR2", "SIGCONT")):
        return

    import stable_pretraining.manager as manager

    def print_signal_info(label="current"):
        manager.log_header(f"SignalHandlers ({label})")
        for name in ("SIGUSR1", "SIGUSR2", "SIGCONT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is None:
                manager.logging.info(f"  {name:<8} -> unavailable on this platform")
            else:
                manager.logging.info(
                    f"  {name:<8} -> {manager._describe_handler(signal.getsignal(sig))}"
                )

    manager.print_signal_info = print_signal_info
