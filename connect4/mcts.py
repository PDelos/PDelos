"""Monte Carlo Tree Search guided by the network (PUCT, as in AlphaZero)."""
from __future__ import annotations

import math

import numpy as np

from .game import Board

C_PUCT = 1.5


class Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior: float):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}

    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _expand(node: Node, board: Board, net) -> float:
    policy, value = net.evaluate(board)
    for c in board.legal_moves():
        node.children[c] = Node(float(policy[c]))
    return value


def search(board: Board, net, simulations: int = 300, dirichlet: float | None = None,
           rng: np.random.Generator | None = None, return_root: bool = False):
    """Run MCTS from `board` and return visit-count probabilities over the 7 columns
    (plus the search tree's root node when return_root=True)."""
    root = Node(1.0)
    _expand(root, board, net)
    if dirichlet is not None and root.children:
        rng = rng or np.random.default_rng()
        noise = rng.dirichlet([dirichlet] * len(root.children))
        for (c, child), n in zip(root.children.items(), noise):
            child.prior = 0.75 * child.prior + 0.25 * n

    for _ in range(simulations):
        node, b, path = root, board.copy(), [root]
        value = None
        while node.children:
            total = math.sqrt(node.visits + 1)
            best, best_score = None, -1e9
            for c, child in node.children.items():
                # child.q() is stored from the child's player's view, so negate it
                score = -child.q() + C_PUCT * child.prior * total / (1 + child.visits)
                if score > best_score:
                    best, best_score = c, score
            if b.is_winning_move(best):
                b.play(best)
                node = node.children[best]
                path.append(node)
                value = -1.0  # the player to move at this node has just lost
                break
            b.play(best)
            node = node.children[best]
            path.append(node)
            if b.is_full():
                value = 0.0
                break
        if value is None:
            value = _expand(node, b, net)
        for n in reversed(path):  # value is from the view of the player to move at `n`
            n.visits += 1
            n.value_sum += value
            value = -value

    visits = np.zeros(7, dtype=np.float64)
    for c, child in root.children.items():
        visits[c] = child.visits
    pi = visits / visits.sum()
    return (pi, root) if return_root else pi


def tree_positions(root: Node, board: Board, k: int, rng: np.random.Generator,
                   min_visits: int = 2, min_ply: int = 8) -> list[str]:
    """Up to k positions the search actually explored (visited at least `min_visits` times),
    sampled at random so they come from different branches. Returned as move sequences."""
    found, stack = [], [(root, board.history)]
    while stack:
        node, seq = stack.pop()
        for c, child in node.children.items():
            if child.visits < min_visits:
                continue
            s = seq + str(c + 1)
            if len(s) >= min_ply and child.children:  # expanded = not a finished game
                found.append(s)
            stack.append((child, s))
    if len(found) <= k:
        return found
    return [found[i] for i in rng.choice(len(found), size=k, replace=False)]


def choose(pi: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(np.argmax(pi))
    p = pi ** (1.0 / temperature)
    return int(rng.choice(7, p=p / p.sum()))
