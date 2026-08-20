"""
Reusable dataset-entry helpers for first-party studies.

This package owns the canonical data-bundle loaders used by the active study
surface. Its public entrypoint is `build_data_bundle`, which dispatches to the
appropriate dataset-specific loader and returns tensors plus the metadata
needed for later replay and evaluation.
"""

from mem_param_src.datasets.common import (
    build_data_bundle,
    validate_data_bundle,
    validate_finite_data_identity,
)

__all__ = [
    "build_data_bundle",
    "validate_data_bundle",
    "validate_finite_data_identity",
]
