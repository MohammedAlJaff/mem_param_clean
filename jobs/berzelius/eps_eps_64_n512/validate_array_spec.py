#!/usr/bin/env python3
"""Validate one unthrottled Slurm array expression against a GPU-class range."""

import sys


def expand_array_spec(value):
    """Expand comma-separated integers and inclusive ranges."""
    if "%" in value:
        raise ValueError("Concurrency throttles are not accepted for this 24-task study")
    result = set()
    for part in value.split(","):
        if not part:
            raise ValueError("Array expression contains an empty item")
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
    """Require every requested ID to belong to thin or fat routing."""
    resource_class, value = sys.argv[1:3]
    allowed = set(range(16)) if resource_class == "thin" else set(range(16, 24))
    requested = expand_array_spec(value)
    if not requested or not requested <= allowed:
        raise ValueError(
            f"{resource_class} task IDs must stay inside "
            f"{min(allowed)}-{max(allowed)}, got {value!r}"
        )
    print(",".join(str(item) for item in sorted(requested)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
