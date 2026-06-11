"""Couche ML — architectures de réseaux (MLP, réseau de fonction valeur)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=1, out_dim=4, hidden=64, depth=4,
                 activation="tanh", positive_output=True):
        super().__init__()
        act = {"tanh": nn.Tanh, "swish": nn.SiLU}[activation]
        layers = [nn.Linear(in_dim, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
        self.positive_output = positive_output
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        y = self.net(t)
        return torch.nn.functional.softplus(y) if self.positive_output else y


class ValueNetwork(nn.Module):
    """V(x, t) pour le solveur HJB-PINN : entrée (S,E,I,R,t) -> scalaire >= 0."""

    def __init__(self, hidden=96, depth=5):
        super().__init__()
        self.body = MLP(in_dim=5, out_dim=1, hidden=hidden, depth=depth, positive_output=True)

    def forward(self, x, t):
        return self.body(torch.cat([x, t], dim=-1))
