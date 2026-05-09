from __future__ import annotations

import ctypes
import gc
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _malloc_trim() -> Any | None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
    except Exception:
        return None
    return trim


def release_unused_memory() -> None:
    """Collect Python objects and ask glibc to return free heap pages when possible."""

    gc.collect()
    trim = _malloc_trim()
    if trim is None:
        return
    try:
        trim(0)
    except Exception:
        return
