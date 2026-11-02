"""Nightly training, run by GitHub Actions with PyTorch on a CPU runner.

1. Review: the perfect solver (tools/solver.c, compiled in the workflow) labels every
   position queued during the day (data/pending.txt): a few positions sampled from each
   real game plus positions the bot's tree search explored while playing. New positions
   (not already in the train or test set, mirror images included) are appended to the
   train set (data/train.txt). Optionally the bot also adds self-play positions.
2. Retrain: once the train set has grown 10% since the last retrain, the network is trained
   from scratch on the whole train set with early stopping on a validation split (see
   connect4/trainer.py). Small nightly additions barely move a converged network, and
   retraining from scratch reproduces the scaling test: accuracy follows train-set size.
3. Gate: the new network goes live only if its strict accuracy on the fixed 10,000-position
   test set doesn't drop by more than 0.3 points. The test set is never trained on.
4. Measure the live model with the full tree search on the first 400 test positions.

Usage: python -m connect4.train [--selfplay 0] [--retrain-growth 0.10]
"""
from __future__ import annotations

import argparse
import subprocess
import time
from collections import Counter
from datetime import date

import numpy as np
import torch

from .dataset import EvalSet, best_masks, canonical_key, format_row, is_validation, parse_scores, read_rows
from .game import Board
from .mcts import choose, search, tree_positions
from .model import Connect4Net, export_inference, load_checkpoint, save_checkpoint
from .net import NumpyNet
from .state import (CHECKPOINT_FILE, MODEL_FILE, PENDING_FILE, ROOT, TEST_FILE, TRAIN_FILE, load_metrics,
                    load_state, save_metrics)
from .trainer import train

SOLVER = ROOT / "tools" / "solver"
MIN_PLY = 8
SEARCH_EVAL_POSITIONS = 400   # test positions also played with the full 400-simulation search


def search_accuracy(net, rows, sims: int = 400) -> dict:
    strict = any_win = 0
    for _, seq, sc in rows:
        s_mask, a_mask = best_masks(sc)
        c = int(np.argmax(search(Board.from_sequence(seq), net, sims)))
        strict += s_mask[c]
        any_win += a_mask[c]
    return {"strict": strict / len(rows), "any_win": any_win / len(rows)}


def solve(seqs: list[str], node_budget: int = 6_000_000) -> dict[str, list]:
    if not seqs:
        return {}
    out = subprocess.run([str(SOLVER), str(node_budget)], input="\n".join(seqs) + "\n",
                         capture_output=True, text=True, check=True).stdout
    labelled = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 8:
            sc = parse_scores(parts[1:])
            if sc is not None:
                labelled[parts[0]] = sc
    return labelled


def self_play_positions(net, games: int, rng) -> list[str]:
    seqs = []
    for _ in range(games):
        b = Board()
        while True:
            pi, root = search(b, net, 100, dirichlet=1.0, rng=rng, return_root=True)
            seqs += tree_positions(root, b, 3, rng, min_ply=MIN_PLY)
            c = choose(pi, temperature=1.0 if b.moves < 8 else 0.0, rng=rng)
            done = b.is_winning_move(c)
            b.play(c)
            if done or b.is_full():
                break
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfplay", type=int, default=0, help="self-play games to add tonight (0 = real games only)")
    ap.add_argument("--time-budget", type=float, default=10800, help="max training seconds")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--retrain-growth", type=float, default=0.10,
                    help="retrain once the train set has grown this much since the last retrain")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng()

    # ---- 1. review new positions with the solver
    train_rows = read_rows(TRAIN_FILE)
    test_rows = read_rows(TEST_FILE)
    seen = {canonical_key(s) for _, s, _ in train_rows} | {canonical_key(s) for _, s, _ in test_rows}
    queued = []
    if PENDING_FILE.exists():
        for line in PENDING_FILE.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                queued.append((parts[0], parts[1]))
    if args.selfplay:
        queued += [("selfplay", s) for s in self_play_positions(NumpyNet(MODEL_FILE), args.selfplay, rng)]
    todo, src_of = [], {}
    for src, seq in queued:
        k = canonical_key(seq)
        if len(seq) < MIN_PLY or k in seen:
            continue
        seen.add(k)
        todo.append(seq)
        src_of[seq] = src
    labels = solve(todo)
    new_rows = [(src_of[s], s, labels[s]) for s in todo if s in labels]
    with TRAIN_FILE.open("a") as f:
        f.writelines(format_row(*r) + "\n" for r in new_rows)
    PENDING_FILE.write_text("")
    train_rows += new_rows
    added = Counter(r[0] for r in new_rows)
    print(f"reviewed {len(queued)} queued positions: {len(new_rows)} new rows {dict(added)}, "
          f"train set now {len(train_rows)} rows ({time.time() - t0:.0f}s)", flush=True)

    # ---- 2. retrain from scratch once the train set has grown enough since the last retrain
    m = load_metrics()
    last = m["runs"][-1] if m["runs"] else None
    last_trained = m.get("last_trained_rows", m.get("seed_rows", 0))
    due = len(train_rows) >= last_trained * (1 + args.retrain_growth)
    if not due:
        print(f"no retrain tonight: train set is {len(train_rows) / max(last_trained, 1) - 1:+.1%} since the last "
              f"retrain ({last_trained} rows); retraining at {args.retrain_growth:.0%} growth")
        m["runs"].append({
            "date": date.today().isoformat(), "trained": False, "accepted": False, "version": m["version"],
            "train_rows": len(train_rows), "rows_by_source": dict(Counter(r[0] for r in train_rows)),
            "new_rows": dict(added), "test_strict": last["test_strict"] if last else m["pretrained"]["strict"],
            "test_any_win": last["test_any_win"] if last else m["pretrained"]["any_win"],
            "search_strict": last["search_strict"] if last else m["pretrained_search"]["strict"],
            "search_any_win": last["search_any_win"] if last else m["pretrained_search"]["any_win"],
            "human_games_total": load_state()["stats"]["games"], "seconds": round(time.time() - t0),
        })
        save_metrics(m)
        return

    test = EvalSet(test_rows)
    before = test.accuracy(load_checkpoint(CHECKPOINT_FILE))
    val_rows = [r for r in train_rows if is_validation(r[1])]
    fit_rows = [r for r in train_rows if not is_validation(r[1])]
    torch.manual_seed(len(train_rows))
    cand = Connect4Net()
    result = train(cand, fit_rows, val_rows, max_epochs=args.max_epochs, patience=3, lr=2e-3,
                   time_budget=args.time_budget)
    after = test.accuracy(cand)

    # ---- 3. gate on the fixed test set
    accepted = after["strict"] >= before["strict"] - 0.003
    print(f"test strict {before['strict']:.2%} -> {after['strict']:.2%}, any-win {before['any_win']:.2%} -> "
          f"{after['any_win']:.2%}: {'ACCEPTED' if accepted else 'kept the current model'} "
          f"({time.time() - t0:.0f}s)")
    if accepted:
        save_checkpoint(cand, CHECKPOINT_FILE)
        export_inference(cand, MODEL_FILE)
        m["version"] += 1
        m["last_trained_rows"] = len(train_rows)
        search_acc = search_accuracy(NumpyNet(MODEL_FILE), test_rows[:SEARCH_EVAL_POSITIONS])
        print(f"with tree search ({SEARCH_EVAL_POSITIONS} test positions): strict {search_acc['strict']:.1%}, "
              f"any-win {search_acc['any_win']:.1%} ({time.time() - t0:.0f}s)")
    else:
        search_acc = ({"strict": last["search_strict"], "any_win": last["search_any_win"]} if last
                      else m["pretrained_search"])
    live = after if accepted else before
    m["runs"].append({
        "date": date.today().isoformat(), "trained": True, "accepted": accepted, "version": m["version"],
        "train_rows": len(train_rows), "rows_by_source": dict(Counter(r[0] for r in train_rows)),
        "new_rows": dict(added), "test_strict": round(live["strict"], 4), "test_any_win": round(live["any_win"], 4),
        "candidate_strict": round(after["strict"], 4), "candidate_any_win": round(after["any_win"], 4),
        "epochs": result["epochs_run"],
        "search_strict": round(search_acc["strict"], 4), "search_any_win": round(search_acc["any_win"], 4),
        "human_games_total": load_state()["stats"]["games"], "seconds": round(time.time() - t0),
    })
    save_metrics(m)

if __name__ == "__main__":
    main()
