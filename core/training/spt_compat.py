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


def stub_lance_if_no_avx():
    """Register a stub ``lance`` module on CPUs without AVX.

    ``pylance``'s native extension is compiled for a newer x86-64 baseline and
    raises SIGILL (an uncatchable illegal-instruction crash) on import on CPUs
    without AVX -- e.g. the ``qemu64`` virtual CPU on the TUM VM, whose flags
    expose only ``sse4_1``/``sse4_2``. ``lance`` is imported solely by
    ``stable_pretraining.data.video`` (used only for video datasets, which our
    vector-observation tasks never touch), so we satisfy ``import lance`` with a
    stub before stable_pretraining imports it. On AVX-capable CPUs the real
    package loads unchanged. Call before importing ``stable_pretraining``.
    """
    import sys

    if "lance" in sys.modules:
        return
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            cpuinfo = fh.read()
    except OSError:
        return  # not Linux / unreadable: let the real import proceed
    if any(token.startswith("avx") for token in cpuinfo.split()):
        return  # real pylance will load fine on an AVX-capable CPU

    import types

    stub = types.ModuleType("lance")
    stub.__doc__ = (
        "Stub injected by spt_compat: this CPU lacks AVX, so real pylance "
        "cannot be imported. Video datasets are unavailable."
    )

    def _unavailable(*args, **kwargs):
        raise RuntimeError(
            "lance is stubbed because this CPU lacks AVX; video datasets are "
            "unavailable on this machine."
        )

    stub.dataset = _unavailable
    stub.write_dataset = _unavailable
    sys.modules["lance"] = stub


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
