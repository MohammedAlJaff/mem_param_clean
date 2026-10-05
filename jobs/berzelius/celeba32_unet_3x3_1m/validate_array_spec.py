#!/usr/bin/env python3
"""Validate unthrottled array expressions for the 108-run CelebA suite."""

import sys


def expand_array_spec(value):
    """Expand comma-separated integers and inclusive ranges."""
    if "%" in value:
        raise ValueError("This production array must remain unthrottled.")
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
    if not result or not result <= set(range(108)):
        raise ValueError("Task IDs must be a nonempty subset of 0 through 107.")
    return result


def main():
    """Validate one selection, optionally requiring the complete matrix."""
    requested = expand_array_spec(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "--require-full":
        if requested != set(range(108)):
            raise ValueError("Initial production submission must select all 108 tasks.")
    print(",".join(str(item) for item in sorted(requested)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
