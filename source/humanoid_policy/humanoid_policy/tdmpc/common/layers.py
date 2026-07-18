"""NN primitives — adapted from TD-MPC2 (MIT). State-only: rgb/conv/aug and the tensordict `vmap`
Ensemble are dropped. The Q-ensemble is a plain ModuleList (see world_model.py)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """Simplicial normalization (arxiv 2204.00616)."""

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """Linear + LayerNorm + activation (+ optional dropout)."""

    def __init__(self, *args, dropout=0.0, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (f"NormedLinear(in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}{repr_dropout}, act={self.act.__class__.__name__})")


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """MLP with LayerNorm + Mish (last layer plain Linear unless `act` given)."""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + list(mlp_dims) + [out_dim]
    layers = nn.ModuleList()
    for i in range(len(dims) - 2):
        layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def state_encoder(cfg):
    """Encoder for a single state observation of dim `cfg.obs_dim` -> latent (+ SimNorm)."""
    return mlp(cfg.obs_dim, max(cfg.num_enc_layers - 1, 1) * [cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
