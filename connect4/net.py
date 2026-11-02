"""NumPy-only forward pass of the policy/value network.

The move workflow runs on every visitor move, so it must start in seconds:
it loads plain .npz weights and never imports PyTorch. The same architecture
is defined in PyTorch in connect4/model.py, which is only used for training.

Architecture (input 2x6x7: stones of the player to move, stones of the opponent)
  stem:   3x3 conv -> ReLU
  body:   residual blocks of two 3x3 convs
  policy: 1x1 conv -> ReLU -> linear -> 7 logits
  value:  1x1 conv -> ReLU -> linear -> ReLU -> linear -> tanh
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .game import Board


def _conv3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = sliding_window_view(xp, (3, 3), axis=(1, 2))  # c, h, w, 3, 3
    cols = cols.transpose(0, 3, 4, 1, 2).reshape(c * 9, h * wd)
    return (w.reshape(w.shape[0], -1) @ cols + b[:, None]).reshape(-1, h, wd)


def _conv1(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, h, wd = x.shape
    return (w.reshape(w.shape[0], c) @ x.reshape(c, -1) + b[:, None]).reshape(-1, h, wd)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class NumpyNet:
    def __init__(self, path: str | Path):
        data = np.load(path)
        self.p = {k: data[k].astype(np.float32) for k in data.files}
        self.blocks = sum(1 for k in self.p if k.startswith("blocks.") and k.endswith(".conv_a.weight"))

    def forward(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        p = self.p
        x = _relu(_conv3(planes, p["stem.weight"], p["stem.bias"]))
        for i in range(self.blocks):
            y = _relu(_conv3(x, p[f"blocks.{i}.conv_a.weight"], p[f"blocks.{i}.conv_a.bias"]))
            y = _conv3(y, p[f"blocks.{i}.conv_b.weight"], p[f"blocks.{i}.conv_b.bias"])
            x = _relu(x + y)
        ph = _relu(_conv1(x, p["policy_conv.weight"], p["policy_conv.bias"])).reshape(-1)
        logits = p["policy_fc.weight"] @ ph + p["policy_fc.bias"]
        vh = _relu(_conv1(x, p["value_conv.weight"], p["value_conv.bias"])).reshape(-1)
        vh = _relu(p["value_fc1.weight"] @ vh + p["value_fc1.bias"])
        value = float(np.tanh(p["value_fc2.weight"] @ vh + p["value_fc2.bias"])[0])
        return logits, value

    def evaluate(self, board: Board) -> tuple[np.ndarray, float]:
        """Policy (probabilities over legal moves, 0 elsewhere) and value for the player to move."""
        logits, value = self.forward(board.planes())
        legal = board.legal_moves()
        masked = np.full(7, -np.inf, dtype=np.float64)
        masked[legal] = logits[legal]
        e = np.exp(masked - masked[legal].max())
        return e / e.sum(), value
