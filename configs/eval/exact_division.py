"""Evaluate saved checkpoints with exact sampling-time division."""

from ml_collections import ConfigDict


def get_config():
    """Return the sampling field changed by this evaluation."""
    config = ConfigDict()
    config.sampling = ConfigDict()
    config.sampling.use_safe_div = False
    return config
