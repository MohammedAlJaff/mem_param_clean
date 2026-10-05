#!/usr/bin/env python3
"""Validate an unthrottled array expression for the 36-task study."""

import sys


def expand_array_spec(value):
    """Expand comma-separated integers and inclusive ranges."""
    if "%" in value:
        raise ValueError("Concurrency throttles are not accepted for this study.")
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
    """Require requested IDs to stay inside the production matrix."""
    value = sys.argv[1]
    requested = expand_array_spec(value)
    allowed = set(range(36))
    if not requested or not requested <= allowed:
        raise ValueError(f"Task IDs must stay inside 0-35, got {value!r}.")
    print(",".join(str(item) for item in sorted(requested)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
