"""
Canonical time-conditioned MLP backbone for synthetic studies.

This module intentionally keeps the architecture simple: one learned time
embedding branch, one plain feed-forward trunk, and a clear `[B, D] -> [B, D]`
contract. It is the easiest backbone in the repo to reason about, which is
why it serves as the default synthetic-data model.
"""

import torch
import torch.nn as nn

from mem_param_src.networks.common import TimeEmbeddingMLP, ensure_time_batch


# =============================================================================
# Canonical synthetic-data backbone
# =============================================================================


class TimeCondMLP(nn.Module):
    """
    Flat-vector MLP with sinusoidal time embedding.

    Input:
        z: [B, D]
        t: [B]

    Output:
        y: [B, D]
    """

    def __init__(self, data_dim, hidden_dim, n_layers, time_emb_dim):
        """
        Build the canonical synthetic-data backbone.

        Structure:
            - one learned time-embedding branch
            - one plain feed-forward trunk on [z, t_emb]

        This is intentionally simple. For low-dimensional synthetic work we
        want the backbone to be easy to reason about before we move to image
        models with more architectural confounders.
        """
        super().__init__()
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim

        self.time_mlp = TimeEmbeddingMLP(time_emb_dim, time_emb_dim)

        # We concatenate the data vector and the learned time embedding once,
        # then use a standard MLP trunk to mix them.
        layers = [nn.Linear(data_dim + time_emb_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, data_dim)]
        self.net = nn.Sequential(*layers)

    def time_embedding(self, t):
        """
        Return the learned time representation consumed by the main MLP trunk.

        We keep this as its own method so notebooks and tests can inspect the
        time branch directly instead of treating it as hidden inside `forward`.
        """
        return self.time_mlp(t)

    def forward(self, z, t):
        """
        Predict one output vector per input vector at time `t`.

        Data flow:
            1. Validate the `[B, D]` data contract and `[B]` time contract.
            2. Embed time into a learned feature vector.
            3. Concatenate `[z, t_emb]`.
            4. Push the result through the feed-forward trunk.
        """
        if z.dim() != 2:
            raise ValueError(
                f"TimeCondMLP expects z shape [B, D], got {tuple(z.shape)}"
            )
        ensure_time_batch(t, z, "TimeCondMLP")

        t_emb = self.time_embedding(t)
        return self.net(torch.cat([z, t_emb], dim=-1))
