"""
Bonnaire-style UNet adapter for image-space GDFM experiments.

This module presents a stable GDFM-facing constructor and forward contract
while delegating image processing to the local Bonnaire-style `SmallUNet`
implementation in this package.
"""

import torch.nn as nn

from mem_param_src.networks.common import ensure_time_batch
from mem_param_src.networks.small_unet import SmallUNet


# =============================================================================
# Bonnaire-style UNet adapter
# =============================================================================


class BonnaireUNet(nn.Module):
    """
    Thin adapter over the local Bonnaire-style UNet implementation.

    The canonical GDFM-facing surface is:
        image_size
        image_channels
        base_channels
        channel_mult
        attn_resolutions
        num_res_blocks
        dropout
        time_emb_dim

    Internally, this maps to `mem_param_src.networks.small_unet.SmallUNet`.
    The wrapper
    keeps the public constructor stable while the internal architecture stays
    local to `src/`.
    """

    def __init__(
        self,
        image_size,
        image_channels,
        base_channels,
        channel_mult,
        num_res_blocks,
        attn_resolutions,
        dropout,
        time_emb_dim=None,
    ):
        """
        Build the paper-facing UNet wrapper, then call the local
        `SmallUNet` implementation.
        """
        super().__init__()
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if image_channels <= 0:
            raise ValueError(f"image_channels must be positive, got {image_channels}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {num_res_blocks}")

        self.image_size = image_size
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.channel_mult = tuple(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = tuple(attn_resolutions)
        self.dropout = dropout
        self.time_emb_dim = time_emb_dim

        # The wrapped model is the real image backbone. This adapter exists so
        # the rest of GDFM can talk to a stable constructor and forward API.
        self.model = SmallUNet(
            in_channels=image_channels,
            out_channels=image_channels,
            base_channels=base_channels,
            channel_mults=self.channel_mult,
            attention_resolutions=self.attn_resolutions,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_emb_dim,
            dropout=dropout,
        )

    def forward(self, z, t):
        """
        Run one UNet forward pass under the canonical `gdfm` contract.

        This adapter does three jobs:
            1. enforce the expected image shape `[B, C, H, W]`
            2. enforce the shared time shape `[B]`
            3. guarantee that the wrapped model preserves the input image shape

        That last check matters because GDFM backbones are expected to predict
        a tensor living in the same ambient space as the noisy input.
        """
        if z.dim() != 4:
            raise ValueError(
                f"BonnaireUNet expects z shape [B, C, H, W], got {tuple(z.shape)}"
            )
        ensure_time_batch(t, z, "BonnaireUNet")
        if z.shape[1] != self.image_channels:
            raise ValueError(
                f"BonnaireUNet expects {self.image_channels} channels, got {z.shape[1]}"
            )
        if z.shape[2] != self.image_size or z.shape[3] != self.image_size:
            raise ValueError(
                f"BonnaireUNet expects spatial size {self.image_size}x{self.image_size}, "
                f"got {z.shape[2]}x{z.shape[3]}"
            )

        # Delegate the real image processing to the wrapped Bonnaire-style UNet.
        out = self.model(z, t)
        if out.shape != z.shape:
            raise ValueError(
                f"BonnaireUNet returned shape {tuple(out.shape)}, expected {tuple(z.shape)}"
            )
        return out
