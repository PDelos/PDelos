"""Residual policy/value network (PyTorch), used for training only.

Design follows what has been shown to reach solver-level Connect 4 play: a small
AlphaZero-style ResNet (5-6 blocks x 64-96 channels) with batch normalisation.

    input    3x6x7: stones of the player to move, opponent stones, playable cells
    stem     3x3 conv -> BN -> ReLU                    (96 channels)
    body     6 residual blocks: conv-BN-ReLU-conv-BN + skip -> ReLU
    policy   1x1 conv (32) -> BN -> ReLU -> linear -> 7 logits
    value    1x1 conv (8)  -> BN -> ReLU -> linear(128) -> ReLU -> dropout -> linear -> tanh

For the move workflow, export_inference() folds every batch-norm into the conv
before it, so connect4/net.py can run the network with plain NumPy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = 96
BLOCKS = 6
IN_PLANES = 3


class ConvBN(nn.Module):
    def __init__(self, cin: int, cout: int, k: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class Block(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv_a = ConvBN(ch, ch, 3)
        self.conv_b = ConvBN(ch, ch, 3)

    def forward(self, x):
        return F.relu(x + self.conv_b(F.relu(self.conv_a(x))))


class Connect4Net(nn.Module):
    def __init__(self, ch: int = CHANNELS, blocks: int = BLOCKS):
        super().__init__()
        self.stem = ConvBN(IN_PLANES, ch, 3)
        self.blocks = nn.ModuleList(Block(ch) for _ in range(blocks))
        self.policy_conv = ConvBN(ch, 32, 1)
        self.policy_fc = nn.Linear(32 * 42, 7)
        self.value_conv = ConvBN(ch, 8, 1)
        self.value_fc1 = nn.Linear(8 * 42, 128)
        self.value_drop = nn.Dropout(0.3)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.stem(x))
        for b in self.blocks:
            x = b(x)
        p = self.policy_fc(F.relu(self.policy_conv(x)).flatten(1))
        v = F.relu(self.value_fc1(F.relu(self.value_conv(x)).flatten(1)))
        v = torch.tanh(self.value_fc2(self.value_drop(v))).squeeze(1)
        return p, v


def loss_fn(logits, value, pi, z, legal_mask, value_weight: float = 1.0):
    """AlphaZero loss: (z - v)^2 - pi . log p   (L2 is applied through weight decay)."""
    logits = logits.masked_fill(~legal_mask, -1e9)
    policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(1).mean()
    value_loss = F.mse_loss(value, z)
    return policy_loss + value_weight * value_loss, policy_loss, value_loss


# ---- checkpoints (full training state, kept in a GitHub Release, not in git) -----------
def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def load_checkpoint(path: str | Path) -> Connect4Net:
    m = Connect4Net()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


# ---- inference export: batch-norm folded into the convolutions, float16 ------------------
def _fold(cb: ConvBN) -> tuple[np.ndarray, np.ndarray]:
    w = cb.conv.weight.detach().double()
    bn = cb.bn
    scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
    w = w * scale[:, None, None, None]
    b = bn.bias.detach().double() - bn.running_mean.double() * scale
    return w.numpy(), b.numpy()


def export_inference(model: Connect4Net, path: str | Path) -> None:
    model = model.eval()
    out = {}
    for name, cb in [("stem", model.stem), ("policy_conv", model.policy_conv), ("value_conv", model.value_conv)] + \
            [(f"blocks.{i}.conv_{s}", getattr(b, f"conv_{s}")) for i, b in enumerate(model.blocks) for s in "ab"]:
        out[name + ".weight"], out[name + ".bias"] = _fold(cb)
    for name in ("policy_fc", "value_fc1", "value_fc2"):
        lin = getattr(model, name)
        out[name + ".weight"] = lin.weight.detach().numpy()
        out[name + ".bias"] = lin.bias.detach().numpy()
    np.savez_compressed(path, **{k: v.astype(np.float16) for k, v in out.items()})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
