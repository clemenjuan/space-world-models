"""Idempotent, process-global Orekit JVM + data bootstrap."""
import os
import threading
from pathlib import Path

import orekit_jpype

# orekit-data.zip lives at repo root (gitignored, ~70 MB).
_DATA_ZIP = Path(__file__).resolve().parents[1] / "orekit-data.zip"
_lock = threading.Lock()
_ready = False


def ensure_orekit() -> None:
    """Start the JVM and load Orekit physical data exactly once per process."""
    global _ready
    with _lock:
        if _ready:
            return
        orekit_jpype.initVM()
        from orekit_jpype.pyhelpers import (
            download_orekit_data_curdir,
            setup_orekit_curdir,
        )
        if not _DATA_ZIP.exists():
            cwd = os.getcwd()
            os.chdir(_DATA_ZIP.parent)
            try:
                download_orekit_data_curdir()
            finally:
                os.chdir(cwd)
        setup_orekit_curdir(str(_DATA_ZIP))
        _ready = True
