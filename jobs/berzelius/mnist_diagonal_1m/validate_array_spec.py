#!/usr/bin/env python3
"""Validate unthrottled array expressions for the MNIST suite."""

import sys


def expand_array_spec(value):
    """Expand comma-separated integers and inclusive ranges."""
    if "%" in value:
        raise ValueError("Concurrency throttles are not accepted for this 18-task array.")
    result = set()
    for part in value.split(","):
        if not part:
            raise ValueError("Array expression contains an empty item.")
        pieces = part.split("-")
        if len(pieces) == 1:
            start = end = int(pieces[0])
        elif len(pieces) == 2:
            start, end = (int(item) for item in pieces)
        else:
            raise ValueError(f"Invalid array item: {part}")
        if start > end:
            raise ValueError(f"Descending array range: {part}")
        result.update(range(start, end + 1))
    return result


def main():
    """Require selected IDs to match the requested GPU class."""
    value = sys.argv[1]
    run_class = sys.argv[2] if len(sys.argv) > 2 else "all"
    allowed = {
        "thin": set(range(18)),
        "fat": set(range(18, 36)),
        "all": set(range(36)),
    }
    if run_class not in allowed:
        raise ValueError("Run class must be thin, fat, or all.")
    requested = expand_array_spec(value)
    if not requested or not requested <= allowed[run_class]:
        raise ValueError(
            f"Task IDs {value!r} do not belong to run class {run_class!r}."
        )
    print(",".join(str(item) for item in sorted(requested)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
