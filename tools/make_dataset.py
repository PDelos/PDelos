"""Build the solver-labelled dataset used for pretraining and evaluation.

1. Simulate games between noisy, semi-skilled players to get realistic positions.
2. Label every legal move of each position with the perfect solver (tools/solver).
3. Save the raw labels; tools/pretrain.py turns them into train/test tensors.

Run locally once:  gcc -O3 -o tools/solver tools/solver.c && python tools/make_dataset.py 30000
"""
from __future__ import annotations

import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from connect4.game import Board, mirror_sequence  # noqa: E402

PLY_WEIGHTS = {p: (3 if 10 <= p <= 26 else 1) for p in range(8, 37)}


def semi_skilled_move(b: Board, rng: random.Random) -> int:
    moves = b.legal_moves()
    if rng.random() < 0.3:
        return rng.choice(moves)
    for c in moves:  # win now
        if b.is_winning_move(c):
            return c
    opp = b.copy()
    opp.cur ^= opp.mask  # view the board as the opponent
    for c in moves:  # block an immediate threat
        if opp.is_winning_move(c):
            return c
    safe = []
    for c in moves:  # avoid handing the opponent a win on top of our stone
        nb = b.copy()
        nb.play(c)
        if not any(nb.is_winning_move(d) for d in nb.legal_moves()):
            safe.append(c)
    pool = safe or moves
    weights = [4 - abs(3 - c) for c in pool]
    return rng.choices(pool, weights=weights)[0]


def sample_position(rng: random.Random) -> str | None:
    target = rng.choices(list(PLY_WEIGHTS), weights=list(PLY_WEIGHTS.values()))[0]
    b = Board()
    while b.moves < target:
        c = semi_skilled_move(b, rng)
        if b.is_winning_move(c) or b.moves + 1 >= 42:
            return None  # the game would end before the target ply
        b.play(c)
    return b.history


def main() -> None:
    """python tools/make_dataset.py <total positions>
    Positions already labelled in tools/build/labels.txt are kept; only new ones are solved."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    out_dir = ROOT / "tools" / "build"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = (out_dir / "labels.txt").read_text().strip().split("\n") if (out_dir / "labels.txt").exists() else []
    canon = lambda s: min(Board.from_sequence(s).key(), Board.from_sequence(mirror_sequence(s)).key())
    seen = {canon(l.split()[0]) for l in existing if l}
    rng = random.Random(2027 + len(existing))
    seqs = []
    while len(seqs) + len(existing) < n:
        s = sample_position(rng)
        if s is None:
            continue
        k = canon(s)
        if k in seen:
            continue
        seen.add(k)
        seqs.append(s)
    print(f"keeping {len(existing)} labelled positions, sampled {len(seqs)} new ones", flush=True)
    chunks = [seqs[i::2] for i in range(2)]
    procs, files = [], []
    t0 = time.time()
    for i, chunk in enumerate(chunks):
        inp = out_dir / f"seqs_{i}.txt"
        inp.write_text("\n".join(chunk) + "\n")
        f = open(out_dir / f"labels_{i}.txt", "w")
        files.append(f)
        procs.append(subprocess.Popen([str(ROOT / "tools" / "solver"), "5000000"],
                                      stdin=open(inp), stdout=f))
    for p in procs:
        p.wait()
    for f in files:
        f.close()
    lines = list(existing)
    for i in range(2):
        lines += (out_dir / f"labels_{i}.txt").read_text().strip().split("\n")
    (out_dir / "labels.txt").write_text("\n".join(l for l in lines if l) + "\n")
    print(f"labelled {len(lines)} positions in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
