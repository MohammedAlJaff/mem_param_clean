#!/usr/bin/env python3
"""Require workers to import the frozen batch source."""

import sys
from pathlib import Path

import mem_param_src


def main():
    """Reject accidental imports from the live worktree."""
    batch_root = Path(sys.argv[1]).resolve()
    expected = (batch_root / "source" / "mem_param_src").resolve()
    imported = Path(mem_param_src.__file__).resolve()
    if expected not in imported.parents:
        raise RuntimeError(f"Imported {imported}, expected source below {expected}.")
    print(f"Frozen source: {imported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
