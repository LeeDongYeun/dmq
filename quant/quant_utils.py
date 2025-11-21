import torch.nn as nn
import torch
from enum import Enum
from typing import List, Union
import numpy as np
import pdb

REDUCTION = Enum('REDUCTION', ('NONE', 'ALL'))

def lp_loss(pred: torch.Tensor, 
            tgt: torch.Tensor, 
            p: int = 2., 
            reduction: REDUCTION = REDUCTION.NONE
            ) -> torch.Tensor:
    if reduction == REDUCTION.NONE:
        return (pred - tgt).abs().pow(p).sum(1).mean()
    elif reduction == REDUCTION.ALL:
        return (pred - tgt).abs().pow(p).mean()
    else:
        raise NotImplementedError


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def minmax(x: torch.Tensor,
            symmetric: bool = False,
            level: int = 256,
            always_zero: bool = False
            ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = min(x.min().item(), 0), max(x.max().item(), 0)
    delta = torch.tensor(float(x_max - x_min) / (level - 1))
    if symmetric:
        x_min, x_max = -max(abs(x_min), x_max), max(abs(x_min), x_max)
        delta = torch.tensor(float(x_max - x_min) / (level - 2))
    if always_zero:
        delta = torch.tensor(float(x_max) / (level - 1))
    if delta < 1e-8:
        delta = 1e-8
    zero_point = torch.round(-x_min / delta) if not (symmetric or always_zero) else 0
    return torch.tensor(delta).type_as(x), zero_point


def mse(x: torch.Tensor,
        symmetric: bool = False,
        level: int = 256,
        always_zero: bool = False,
        return_minmax: bool = False,
        grid=0.01,
        search_range=80,
        ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = x.min().item(), x.max().item()
    delta, zero_point = None, None
    s = 1e+10

    new_min = torch.Tensor([x_min * (1. - (i * grid)) for i in range(search_range)]).cuda()
    new_max = torch.Tensor([x_max * (1. - (i * grid)) for i in range(search_range)]).cuda()
    new_delta = (new_max - new_min) / (level - 1)

    if symmetric: 
        new_min, new_max = -max(abs(new_min), new_max), max(abs(new_min), new_max)
        new_delta = (new_max - new_min) / (level - 2)
    if always_zero:
        new_delta = new_max / (level - 1)

    new_zero_point = torch.round(-new_min / (new_delta)) if not (symmetric or always_zero) else 0
    NB, PB = -level // 2 if symmetric and not always_zero else 0,\
            level // 2 - 1 if symmetric and not always_zero else level - 1

    x = x.reshape(1, -1).expand(search_range, -1)
    new_delta = new_delta.view(search_range, 1)
    new_zero_point = new_zero_point.view(search_range, 1)

    x_q = torch.clamp(torch.round(x / (new_delta)) + new_zero_point, NB, PB)
    x_dq = new_delta * (x_q - new_zero_point)
    new_s = (x_dq - x).abs().pow(2.4).reshape(search_range,-1).mean(axis=1)
    idx = torch.argmin(new_s)

    delta = new_delta[idx].reshape(-1)
    zero_point = new_zero_point[idx].reshape(-1)

    if return_minmax:
        return delta, zero_point, new_min[idx].reshape(-1), new_max[idx].reshape(-1)

    return delta, zero_point


class Scaler(Enum):
    MINMAX = minmax
    MSE = mse