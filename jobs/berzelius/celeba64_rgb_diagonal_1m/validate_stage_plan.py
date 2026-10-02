#!/usr/bin/env python3
"""Validate the five absolute stop steps for CelebA64 exact resumes."""

import sys


EXPECTED_STOPS = (200_000, 400_000, 600_000, 800_000, 1_000_000)


def main():
    """Require five ordered 200k stages ending at one million."""
    observed = tuple(int(value) for value in sys.argv[1:])
    if observed != EXPECTED_STOPS:
        raise ValueError(
            f"Expected stage stops {EXPECTED_STOPS}, got {observed}."
        )
    print(",".join(str(value) for value in observed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
