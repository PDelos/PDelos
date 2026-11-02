"""Solver-labelled positions: storage, splits, training tensors and accuracy metrics.

Row format (plain text, one position per line, kept in git):
    <source> <move sequence> <score col1> ... <score col7>
source is one of: seed, game, tree, selfplay. "x" marks a full column.
Solver scores: positive = the player to move wins (bigger = sooner), 0 = draw.

Splits
- test set (data/test_set.txt): 10,000 fixed positions, never trained on.
- validation: 5% of the train set, picked by a hash of the position, used only
  to decide when to stop training (early stopping).
- train: everything else; it grows with every game played.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .game import Board, mirror_sequence

SOURCES = ("seed", "game", "tree", "selfplay")


def canonical_key(seq: str) -> int:
    return min(Board.from_sequence(seq).key(), Board.from_sequence(mirror_sequence(seq)).key())


def is_validation(seq: str) -> bool:
    k = canonical_key(seq)
    return hashlib.blake2b(str(k).encode(), digest_size=2).digest()[0] < 13  # ~5%


def parse_scores(parts: list[str]) -> list[int | None] | None:
    if len(parts) != 7 or "?" in parts:
        return None
    return [None if s == "x" else int(s) for s in parts]


def read_rows(path: str | Path) -> list[tuple[str, str, list]]:
    """Returns (source, sequence, scores) rows; accepts lines with or without a source tag."""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in SOURCES:
            src, parts = parts[0], parts[1:]
        else:
            src = "seed"
        if len(parts) != 8:
            continue
        sc = parse_scores(parts[1:])
        if sc is not None:
            rows.append((src, parts[0], sc))
    return rows


def format_row(src: str, seq: str, scores: list) -> str:
    return f"{src} {seq} " + " ".join("x" if v is None else str(v) for v in scores)


def best_masks(scores: list) -> tuple[np.ndarray, np.ndarray]:
    """strict = moves with the best solver score (fastest win / slowest loss);
    any_win = moves with the best game-theoretic result (win, draw or loss)."""
    legal = [v for v in scores if v is not None]
    best = max(legal)
    strict = np.array([v is not None and v == best for v in scores])
    sign = np.sign(best)
    any_win = np.array([v is not None and np.sign(v) == sign for v in scores])
    return strict, any_win


def encode(rows, augment: bool = True):
    """Tensors for training: planes, legal mask, policy target (uniform over the best moves),
    value target (true result for the player to move)."""
    n = len(rows) * (2 if augment else 1)
    planes = np.zeros((n, 3, 6, 7), dtype=np.float32)
    legal = np.zeros((n, 7), dtype=bool)
    pi = np.zeros((n, 7), dtype=np.float32)
    z = np.zeros(n, dtype=np.float32)
    i = 0
    for _, seq, sc in rows:
        b = Board.from_sequence(seq)
        pl = b.planes()
        strict, _ = best_masks(sc)
        lg = np.array([v is not None for v in sc])
        target = strict / strict.sum()
        value = float(np.sign(max(v for v in sc if v is not None)))
        for flip in ((False, True) if augment else (False,)):
            planes[i] = pl[:, :, ::-1] if flip else pl
            legal[i] = lg[::-1] if flip else lg
            pi[i] = target[::-1] if flip else target
            z[i] = value
            i += 1
    return torch.from_numpy(planes), torch.from_numpy(legal), torch.from_numpy(pi), torch.from_numpy(z)


class EvalSet:
    """Pre-encoded positions for fast accuracy checks."""

    def __init__(self, rows):
        self.planes, self.legal, _, _ = encode(rows, augment=False)
        masks = [best_masks(sc) for _, _, sc in rows]
        self.strict = np.stack([m[0] for m in masks])
        self.any_win = np.stack([m[1] for m in masks])
        self.n = len(rows)

    @torch.no_grad()
    def accuracy(self, model, batch: int = 4096) -> dict:
        model.eval()
        picks = []
        for i in range(0, self.n, batch):
            logits, _ = model(self.planes[i:i + batch])
            picks.append(logits.masked_fill(~self.legal[i:i + batch], -1e9).argmax(1).numpy())
        p = np.concatenate(picks)
        idx = np.arange(self.n)
        return {"strict": float(self.strict[idx, p].mean()), "any_win": float(self.any_win[idx, p].mean())}
