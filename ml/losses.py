"""Couche ML — termes de perte du PINN (L_data, L_phys, L_IC, L_ctrl) via autograd."""

from __future__ import annotations

import torch


def time_derivative(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    grads = [
        torch.autograd.grad(y[:, k].sum(), t, create_graph=True, retain_graph=True)[0]
        for k in range(y.shape[1])
    ]
    return torch.cat(grads, dim=1)


def seir_residual(t, y, beta, sigma, gamma, N, u1=0.0, u2=0.0):
    S, E, I, R = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    dy = time_derivative(y, t)
    new_inf = beta * (1.0 - u2) * S * I / N
    f = torch.cat(
        [-new_inf - u1 * S, new_inf - sigma * E, sigma * E - gamma * I, gamma * I + u1 * S],
        dim=1,
    )
    return dy - f


def loss_data(y_pred, y_obs):
    return torch.mean((y_pred - y_obs) ** 2)


def loss_physics(residual):
    return torch.mean(residual ** 2)


def loss_initial(y0_pred, y0_true):
    return torch.mean((y0_pred - y0_true) ** 2)
