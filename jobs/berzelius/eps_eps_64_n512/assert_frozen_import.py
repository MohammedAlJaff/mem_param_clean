#!/usr/bin/env python3
"""Require Python to import mem_param_src from the frozen batch snapshot."""

import sys
from pathlib import Path

import mem_param_src


def main():
    """Reject a live-checkout or installed-package import before any job work."""
    batch_root = Path(sys.argv[1]).resolve()
    expected = (batch_root / "source/mem_param_src/__init__.py").resolve()
    actual = Path(mem_param_src.__file__).resolve()
    if actual != expected:
        raise RuntimeError(
            f"Expected frozen mem_param_src at {expected}, imported {actual} instead."
        )
    print(f"Frozen source import: {actual}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
