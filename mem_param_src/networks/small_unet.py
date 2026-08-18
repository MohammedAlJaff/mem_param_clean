"""
Small UNet used by the canonical Bonnaire-style wrapper.

This file was copied from the archived in-repo natural-images stack and moved
into `mem_param_src/` so the live GDFM path no longer depends on experiments.

The architecture stays the same:
    - residual UNet with adaptive time conditioning
    - self-attention at selected resolutions
    - stride-2 downsampling and nearest-neighbor upsampling

The public image runner usually constructs this model through
`mem_param_src.networks.unet.BonnaireUNet`. Read this file when you want the
layer-level image backbone.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Map `t in [0, 1]` to a bank of sinusoids."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        out = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            out = torch.cat([out, torch.zeros_like(t.unsqueeze(1))], dim=1)
        return out


class TimeMLPEmbedding(nn.Module):
    """Sinusoidal time embedding followed by a small MLP."""

    def __init__(self, time_embed_dim):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

    def forward(self, t):
        if t.dim() == 2:
            t = t.squeeze(1)
        return self.mlp(self.sinusoidal(t))


class ResBlock(nn.Module):
    """
    Residual block with time-conditioned adaptive group normalization.

    Time enters through scale and shift parameters derived from the time
    embedding.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        dropout=0.0,
        num_groups=32,
    ):
        super().__init__()

        ng1 = min(num_groups, in_channels)
        ng2 = min(num_groups, out_channels)

        self.norm1 = nn.GroupNorm(ng1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2)
        self.norm2 = nn.GroupNorm(ng2, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t_proj = self.time_proj(F.silu(t_emb))
        scale, shift = t_proj.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over spatial positions."""

    def __init__(self, channels, num_groups=32):
        super().__init__()
        ng = min(num_groups, channels)
        self.norm = nn.GroupNorm(ng, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.norm(x)

        qkv = self.qkv(h).reshape(batch, 3, channels, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        scale = channels ** -0.5
        attn = torch.bmm(q, k.transpose(1, 2)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(batch, channels, height, width)

        return x + self.proj(out)


class Downsample(nn.Module):
    """Stride-2 convolution downsampling."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsampling followed by a 3x3 convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SmallUNet(nn.Module):
    """
    Bonnaire-style UNet for image-space flow matching.

    The channel schedule is controlled by `base_channels` and `channel_mults`.
    Attention is applied only at the requested spatial resolutions.

    Input and output tensors both have shape `[B, C, H, W]`. GDFM requires this
    shape preservation because the network prediction must live in the same
    ambient image space as the noisy input.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=None,
        base_channels=64,
        channel_mults=(1, 2, 4, 4),
        attention_resolutions=(8,),
        num_res_blocks=2,
        time_embed_dim=None,
        dropout=0.0,
        num_groups=32,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        if time_embed_dim is None:
            time_embed_dim = base_channels * 4

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embed = TimeMLPEmbedding(time_embed_dim)

        channels = [base_channels * multiplier for multiplier in channel_mults]
        num_levels = len(channel_mults)

        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()

        ch_in = channels[0]
        self._encoder_channels = [ch_in]

        for level in range(num_levels):
            ch_out = channels[level]
            blocks_at_level = nn.ModuleList()

            for _ in range(num_res_blocks):
                blocks_at_level.append(
                    ResBlock(ch_in, ch_out, time_embed_dim, dropout, num_groups)
                )
                ch_in = ch_out
                self._encoder_channels.append(ch_in)

            self.encoder_blocks.append(blocks_at_level)
            if level < num_levels - 1:
                self.encoder_attns.append(SelfAttention(ch_out, num_groups))
            else:
                self.encoder_attns.append(nn.Identity())

            if level < num_levels - 1:
                self.encoder_downsamples.append(Downsample(ch_out))
                self._encoder_channels.append(ch_out)
            else:
                self.encoder_downsamples.append(nn.Identity())

        self.middle_block1 = ResBlock(ch_in, ch_in, time_embed_dim, dropout, num_groups)
        self.middle_attn = SelfAttention(ch_in, num_groups)
        self.middle_block2 = ResBlock(ch_in, ch_in, time_embed_dim, dropout, num_groups)

        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        for level in reversed(range(num_levels)):
            ch_out = channels[level]
            blocks_at_level = nn.ModuleList()

            for _ in range(num_res_blocks + 1):
                skip_ch = self._encoder_channels.pop()
                blocks_at_level.append(
                    ResBlock(ch_in + skip_ch, ch_out, time_embed_dim, dropout, num_groups)
                )
                ch_in = ch_out

            self.decoder_blocks.append(blocks_at_level)
            if level > 0:
                self.decoder_upsamples.append(Upsample(ch_out))
            else:
                self.decoder_upsamples.append(nn.Identity())

        ng_out = min(num_groups, ch_in)
        self.output_norm = nn.GroupNorm(ng_out, ch_in)
        self.output_conv = nn.Conv2d(ch_in, out_channels, 3, padding=1)

        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

        self._attention_resolutions = set(attention_resolutions)

    def forward(self, x, t):
        """
        Run the encoder, middle block, and decoder for one image batch.

        Skip tensors are saved during the encoder and concatenated into the
        decoder, which is the usual UNet information path across resolutions.
        """
        if t.dim() == 2:
            t = t.squeeze(1)

        t_emb = self.time_embed(t)
        h = self.input_conv(x)

        skips = [h]
        for level, (blocks, attn, down) in enumerate(
            zip(self.encoder_blocks, self.encoder_attns, self.encoder_downsamples)
        ):
            for block in blocks:
                h = block(h, t_emb)
                if h.shape[-1] in self._attention_resolutions:
                    if not isinstance(attn, nn.Identity):
                        h = attn(h)
                skips.append(h)

            if level < len(self.encoder_blocks) - 1:
                h = down(h)
                skips.append(h)

        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)

        for blocks, up in zip(self.decoder_blocks, self.decoder_upsamples):
            for block in blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
            h = up(h)

        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)
        return h


def make_unet_for_resolution(resolution, in_channels=3, base_channels=64):
    """Build a default UNet depth profile for a common image resolution."""
    configs = {
        32: dict(channel_mults=(1, 2, 4, 4), attention_resolutions=(8,)),
        64: dict(channel_mults=(1, 2, 4, 4), attention_resolutions=(16,)),
        128: dict(channel_mults=(1, 2, 4, 4, 8), attention_resolutions=(16,)),
        256: dict(channel_mults=(1, 2, 4, 4, 8, 8), attention_resolutions=(16,)),
    }

    if resolution not in configs:
        raise ValueError(
            f"No UNet config for resolution {resolution}. "
            f"Available: {list(configs.keys())}"
        )

    return SmallUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        **configs[resolution],
    )
