"""
Shared time-conditioning utilities for GDFM backbones.

This module owns the pieces that every backbone needs before it can do any
model-specific work:
- validating the shared `[B]` time-vector contract
- turning scalar times into sinusoidal features
- learning a small post-processing map on top of those fixed features

Keeping these helpers together makes the MLP and UNet paths read more
uniformly.
"""

import math

import torch
import torch.nn as nn


def ensure_time_batch(t, ref, name):
    """
    Check the shared time-conditioning contract used by all GDFM backbones.

    Every network in this package expects:
        ref: batch-first input tensor
        t:   one scalar time value per batch element, shape [B]

    Keeping this check in one helper makes the error messages uniform across
    the MLP and UNet paths.
    """
    if t.dim() != 1:
        raise ValueError(f"{name} expects t shape [B], got {tuple(t.shape)}")
    if ref.shape[0] != t.shape[0]:
        raise ValueError(
            f"{name} batch mismatch: input has {ref.shape[0]} samples, "
            f"t has {t.shape[0]} samples"
        )


def sinusoidal_time_embedding(t, time_emb_dim):
    """
    Map scalar times `t in [0, 1]` to a fixed bank of sinusoids.

    The idea is the same as positional encodings in transformers: represent a
    single scalar using multiple frequencies so the downstream network can
    recover both coarse and fine time information.

    Output shape:
        [B, time_emb_dim]
    """
    half = time_emb_dim // 2
    idx = torch.arange(half, device=t.device, dtype=t.dtype)
    freqs = torch.exp(-math.log(10000.0) * idx / max(half - 1, 1))
    args = t[:, None] * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeEmbeddingMLP(nn.Module):
    """
    Learned post-processing on top of the fixed sinusoidal time embedding.

    The sinusoidal features give us a stable, hand-designed representation of
    time. This small MLP then lets each backbone reshape that representation
    into whatever hidden geometry it finds useful.
    """

    def __init__(self, time_emb_dim, out_dim):
        """
        Build a tiny two-layer projection from sinusoidal time features to the
        hidden size expected by the consuming network.
        """
        super().__init__()
        if time_emb_dim % 2 != 0:
            raise ValueError(f"time_emb_dim must be even, got {time_emb_dim}")
        self.time_emb_dim = time_emb_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        """
        Apply the full time-embedding pipeline.

        Steps:
            1. Build the fixed sinusoidal representation of `t`.
            2. Pass it through a small learned MLP.

        This separation is useful pedagogically because it distinguishes the
        hand-designed encoding from the learned adaptation layer.
        """
        emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        return self.net(emb)
