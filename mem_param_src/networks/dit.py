"""Unconditional Small DiT backbone for image-space GDFM models.

Adapted from ``models.py`` in the official ``willisma/SiT`` repository at
commit ``cbde832a40b153ccc79603412409da9c9b0c568c``:
https://github.com/willisma/SiT/blob/cbde832a40b153ccc79603412409da9c9b0c568c/models.py

The upstream code is MIT licensed. See ``LICENSE_SiT.txt`` beside this file.
Local changes expose the repository's unconditional ``forward(z, t)`` contract,
use one fixed internal class with dropout disabled, and always emit raw
``image_channels`` predictions.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

from mem_param_src.networks.common import ensure_time_batch


def _modulate(x, shift, scale):
    """Apply adaptive layer-normalization shift and scale parameters."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _TimestepEmbedder(nn.Module):
    """Embed scalar, potentially fractional times into hidden vectors."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10_000):
        """Create the upstream sinusoidal embedding without rescaling time."""
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        arguments = t[:, None].float() * frequencies[None]
        embedding = torch.cat(
            [torch.cos(arguments), torch.sin(arguments)], dim=-1
        )
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        """Apply the sinusoidal features and learned time MLP."""
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class _LabelEmbedder(nn.Module):
    """Embed one fixed class for unconditional use, with no class dropout."""

    def __init__(self, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(1, hidden_size)

    def forward(self, batch_size, device):
        """Return the same learned class vector for every sample."""
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        return self.embedding_table(labels)


class _SiTBlock(nn.Module):
    """Standard SiT transformer block with adaLN-Zero conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1.0e-6
        )
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True
        )
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1.0e-6
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approximate_gelu():
            return nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approximate_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, conditioning):
        """Run attention and MLP residual paths under adaLN-Zero control."""
        parameters = self.adaLN_modulation(conditioning).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parameters
        x = x + gate_msa.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class _FinalLayer(nn.Module):
    """Map conditioned transformer tokens back to image patches."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1.0e-6
        )
        self.linear = nn.Linear(
            hidden_size,
            patch_size * patch_size * out_channels,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, conditioning):
        """Apply final adaptive normalization and patch projection."""
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class SmallDiT(nn.Module):
    """Small unconditional DiT with the shared ``forward(z, t)`` interface."""

    def __init__(
        self,
        image_size,
        image_channels,
        *,
        depth=12,
        hidden_size=384,
        num_heads=6,
        mlp_ratio=4.0,
        patch_size=2,
        learn_sigma=False,
    ):
        """Build a configurable SiT-S/2-style image backbone."""
        super().__init__()
        self._validate_dimensions(
            image_size=image_size,
            image_channels=image_channels,
            depth=depth,
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            learn_sigma=learn_sigma,
        )

        self.image_size = image_size
        self.image_channels = image_channels
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.learn_sigma = False
        self.out_channels = image_channels

        self.x_embedder = PatchEmbed(
            image_size,
            patch_size,
            image_channels,
            hidden_size,
            bias=True,
        )
        self.t_embedder = _TimestepEmbedder(hidden_size)
        self.y_embedder = _LabelEmbedder(hidden_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, hidden_size),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                _SiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = _FinalLayer(
            hidden_size,
            patch_size,
            self.out_channels,
        )
        self.initialize_weights()

    @staticmethod
    def _validate_dimensions(
        *,
        image_size,
        image_channels,
        depth,
        hidden_size,
        num_heads,
        mlp_ratio,
        patch_size,
        learn_sigma,
    ):
        """Reject unsupported image and attention dimensions with clear errors."""
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if image_channels <= 0:
            raise ValueError(
                f"image_channels must be positive, got {image_channels}"
            )
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by patch_size {patch_size}"
            )
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        if hidden_size % 4 != 0:
            raise ValueError(
                "hidden_size must be divisible by 4 for two-dimensional "
                f"sin-cos positions, got {hidden_size}"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not isinstance(learn_sigma, bool):
            raise ValueError("learn_sigma must be true or false")
        if learn_sigma:
            raise ValueError(
                "SmallDiT requires learn_sigma=False so it returns one raw "
                "prediction channel per input channel"
            )

    def initialize_weights(self):
        """Apply the upstream Xavier, normal, and adaLN-Zero initialization."""

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        grid_size = self.image_size // self.patch_size
        positional_embedding = _get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], grid_size
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(positional_embedding).float().unsqueeze(0)
        )

        patch_weights = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(patch_weights.view([patch_weights.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _unpatchify(self, tokens):
        """Reassemble patch predictions into a batch of square images."""
        patch_size = self.patch_size
        grid_size = self.image_size // patch_size
        expected_tokens = grid_size * grid_size
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"SmallDiT produced {tokens.shape[1]} patches, expected "
                f"{expected_tokens}"
            )
        images = tokens.reshape(
            tokens.shape[0],
            grid_size,
            grid_size,
            patch_size,
            patch_size,
            self.out_channels,
        )
        images = torch.einsum("nhwpqc->nchpwq", images)
        return images.reshape(
            tokens.shape[0],
            self.out_channels,
            self.image_size,
            self.image_size,
        )

    def forward(self, z, t):
        """Return raw image-shaped predictions at the supplied continuous times."""
        if z.dim() != 4:
            raise ValueError(
                f"SmallDiT expects z shape [B, C, H, W], got {tuple(z.shape)}"
            )
        ensure_time_batch(t, z, "SmallDiT")
        if not torch.is_floating_point(t):
            raise ValueError(
                f"SmallDiT expects floating-point continuous times, got {t.dtype}"
            )
        if z.shape[1] != self.image_channels:
            raise ValueError(
                f"SmallDiT expects {self.image_channels} channels, got {z.shape[1]}"
            )
        if z.shape[2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"SmallDiT expects spatial size {self.image_size}x{self.image_size}, "
                f"got {z.shape[2]}x{z.shape[3]}"
            )

        tokens = self.x_embedder(z) + self.pos_embed
        time_conditioning = self.t_embedder(t)
        class_conditioning = self.y_embedder(z.shape[0], z.device)
        conditioning = time_conditioning + class_conditioning
        for block in self.blocks:
            tokens = block(tokens, conditioning)
        prediction = self._unpatchify(
            self.final_layer(tokens, conditioning)
        )
        if prediction.shape != z.shape:
            raise ValueError(
                f"SmallDiT returned shape {tuple(prediction.shape)}, "
                f"expected {tuple(z.shape)}"
            )
        return prediction


def _get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Create the fixed two-dimensional sine/cosine positional embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Combine one-dimensional embeddings for both image axes."""
    embedding_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    embedding_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([embedding_h, embedding_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, positions):
    """Create one-dimensional sine/cosine features for one position grid."""
    frequencies = np.arange(embed_dim // 2, dtype=np.float64)
    frequencies /= embed_dim / 2.0
    frequencies = 1.0 / 10_000**frequencies
    arguments = np.einsum("m,d->md", positions.reshape(-1), frequencies)
    return np.concatenate([np.sin(arguments), np.cos(arguments)], axis=1)


__all__ = ["SmallDiT"]
