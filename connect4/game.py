"""Connect 4 rules on a 7x6 bitboard.

Bit layout: column c, row r (0 = bottom) is bit c * 7 + r. Each column has one
spare bit on top so shifts never wrap between columns.

`cur` holds the stones of the player to move, `mask` holds every stone.
"""
from __future__ import annotations

import numpy as np

W, H = 7, 6
H1 = H + 1
BOTTOM = sum(1 << (c * H1) for c in range(W))
BOARD = BOTTOM * ((1 << H) - 1)
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)


def _col_mask(c: int) -> int:
    return ((1 << H) - 1) << (c * H1)


def _top(c: int) -> int:
    return 1 << (H - 1 + c * H1)


def _bottom(c: int) -> int:
    return 1 << (c * H1)


def has_four(p: int) -> bool:
    for s in (1, H1, H, H + 2):
        m = p & (p >> s)
        if m & (m >> (2 * s)):
            return True
    return False


class Board:
    __slots__ = ("cur", "mask", "moves", "history")

    def __init__(self, cur: int = 0, mask: int = 0, moves: int = 0, history: str = ""):
        self.cur, self.mask, self.moves, self.history = cur, mask, moves, history

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_sequence(cls, seq: str) -> "Board":
        b = cls()
        for ch in seq:
            b.play(int(ch) - 1)
        return b

    def copy(self) -> "Board":
        return Board(self.cur, self.mask, self.moves, self.history)

    # ---- rules ---------------------------------------------------------------
    def can_play(self, c: int) -> bool:
        return 0 <= c < W and not (self.mask & _top(c))

    def legal_moves(self) -> list[int]:
        return [c for c in range(W) if not (self.mask & _top(c))]

    def is_winning_move(self, c: int) -> bool:
        p = self.cur | ((self.mask + _bottom(c)) & _col_mask(c))
        return has_four(p)

    def play(self, c: int) -> None:
        if not self.can_play(c):
            raise ValueError(f"column {c + 1} is not playable")
        self.cur ^= self.mask
        self.mask |= self.mask + _bottom(c)
        self.moves += 1
        self.history += str(c + 1)

    def last_player_won(self) -> bool:
        """True if the player who made the last move has four in a row."""
        return self.moves > 0 and has_four(self.cur ^ self.mask)

    def is_full(self) -> bool:
        return self.moves >= W * H

    def outcome(self) -> int | None:
        """+1/-1/0 from the point of view of the player to move, None if not over."""
        if self.last_player_won():
            return -1
        if self.is_full():
            return 0
        return None

    # ---- views ---------------------------------------------------------------
    def grid(self) -> np.ndarray:
        """6x7 array, row 0 = top. 1 = first player, 2 = second player, 0 = empty."""
        g = np.zeros((H, W), dtype=np.int8)
        to_move = 1 + (self.moves % 2)
        other = 3 - to_move
        opp = self.cur ^ self.mask
        for c in range(W):
            for r in range(H):
                bit = 1 << (c * H1 + r)
                if self.cur & bit:
                    g[H - 1 - r, c] = to_move
                elif opp & bit:
                    g[H - 1 - r, c] = other
        return g

    def planes(self) -> np.ndarray:
        """3x6x7 float32 input for the network, row 0 = top:
        [stones of the player to move, opponent stones, cells you can play into right now]."""
        g = self.grid()
        to_move = 1 + (self.moves % 2)
        playable = np.zeros((H, W), dtype=bool)
        for c in range(W):
            empty = np.flatnonzero(g[:, c] == 0)
            if len(empty):
                playable[empty[-1], c] = True
        return np.stack([(g == to_move), (g == 3 - to_move), playable]).astype(np.float32)

    def key(self) -> int:
        return self.cur + self.mask


def mirror_sequence(seq: str) -> str:
    return "".join(str(8 - int(ch)) for ch in seq)
