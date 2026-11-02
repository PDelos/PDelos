"""Builds the fixed test set, runs the scaling test and trains the starting model (v1).

Run locally after tools/make_dataset.py has labelled the position pool:

    python tools/pretrain.py scaling 5000 15000 40000 80000   # accuracy vs train-set size
    python tools/pretrain.py seed 15000                        # train v1 on the first 15,000 rows

Outputs (in data/): test_set.txt (10,000 fixed positions), train.txt (the seed rows),
checkpoint.pt + model.npz (v1), metrics.json, scaling.json.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from connect4.dataset import EvalSet, best_masks, format_row, is_validation, read_rows  # noqa: E402
from connect4.game import Board  # noqa: E402
from connect4.model import Connect4Net, count_parameters, export_inference, save_checkpoint  # noqa: E402
from connect4.trainer import train  # noqa: E402

DATA = ROOT / "data"
POOL = ROOT / "tools" / "build" / "labels.txt"
TEST_SIZE = 10_000


def informative(seq: str, sc: list) -> bool:
    """Test positions must be real decisions: no instant win, and not every move equally good."""
    b = Board.from_sequence(seq)
    if any(b.is_winning_move(c) for c in b.legal_moves()):
        return False
    strict, _ = best_masks(sc)
    return not strict[[v is not None for v in sc]].all()


def split_pool():
    rows = read_rows(POOL)
    random.Random(0).shuffle(rows)
    test_idx = [i for i, (_, s, sc) in enumerate(rows) if informative(s, sc)][:TEST_SIZE]
    tset = set(test_idx)
    return [rows[i] for i in test_idx], [r for i, r in enumerate(rows) if i not in tset]


def baselines(test_rows) -> dict:
    out = {"random": {"strict": [], "any_win": []}, "heuristic": {"strict": [], "any_win": []}}
    for _, seq, sc in test_rows:
        strict, anyw = best_masks(sc)
        legal = [c for c, v in enumerate(sc) if v is not None]
        out["random"]["strict"].append(strict[legal].mean())
        out["random"]["any_win"].append(anyw[legal].mean())
        b = Board.from_sequence(seq)
        opp = b.copy()
        opp.cur ^= opp.mask
        blocks = [c for c in legal if opp.is_winning_move(c)]
        pick = blocks[0] if blocks else min(legal, key=lambda c: abs(3 - c))
        out["heuristic"]["strict"].append(strict[pick])
        out["heuristic"]["any_win"].append(anyw[pick])
    return {k: {m: round(float(np.mean(v)), 4) for m, v in d.items()} for k, d in out.items()}


def fit(rows, seed: int = 0):
    torch.manual_seed(seed)
    model = Connect4Net()
    val = [r for r in rows if is_validation(r[1])]
    fit_rows = [r for r in rows if not is_validation(r[1])]
    res = train(model, fit_rows, val, max_epochs=40, patience=3, lr=2e-3)
    return model, res


def main() -> None:
    mode, sizes = sys.argv[1], [int(x) for x in sys.argv[2:]]
    test_rows, pool = split_pool()
    test = EvalSet(test_rows)
    print(f"pool: {len(pool)} train-eligible rows, {len(test_rows)} test rows; "
          f"model has {count_parameters(Connect4Net()):,} parameters", flush=True)

    if mode == "scaling":
        out_path = DATA / "scaling.json"
        results = json.loads(out_path.read_text()) if out_path.exists() else []
        for n in sizes:
            t = time.time()
            model, res = fit(pool[:n])
            acc = test.accuracy(model)
            save_checkpoint(model, ROOT / "tools" / "build" / f"scaling_{n}.pt")
            (ROOT / "tools" / "build" / f"scaling_{n}_history.json").write_text(json.dumps(res["history"]))
            results = [r for r in results if r["train_rows"] != n] + [{
                "train_rows": n, "test_strict": round(acc["strict"], 4), "test_any_win": round(acc["any_win"], 4),
                "best_epoch": res["best_epoch"], "minutes": round((time.time() - t) / 60, 1)}]
            results.sort(key=lambda r: r["train_rows"])
            out_path.write_text(json.dumps(results, indent=1) + "\n")
            print(f"== {n} rows: test strict {acc['strict']:.1%}, any-win {acc['any_win']:.1%} "
                  f"(best epoch {res['best_epoch']}, {(time.time() - t) / 60:.0f} min)", flush=True)

    elif mode == "seed":
        n = sizes[0]
        DATA.mkdir(exist_ok=True)
        (DATA / "test_set.txt").write_text("".join(format_row("seed", s, sc).split(" ", 1)[1] + "\n"
                                                   for _, s, sc in test_rows))
        (DATA / "train.txt").write_text("".join(format_row("seed", s, sc) + "\n" for _, s, sc in pool[:n]))
        saved = ROOT / "tools" / "build" / f"scaling_{n}.pt"
        if saved.exists():  # reuse the model the scaling test already trained on exactly these rows
            from connect4.model import load_checkpoint
            model = load_checkpoint(saved)
            res = {"history": json.loads((ROOT / "tools" / "build" / f"scaling_{n}_history.json").read_text())}
        else:
            model, res = fit(pool[:n])
        acc = test.accuracy(model)
        save_checkpoint(model, DATA / "checkpoint.pt")
        export_inference(model, DATA / "model.npz")
        scaling = json.loads((DATA / "scaling.json").read_text()) if (DATA / "scaling.json").exists() else []
        (DATA / "metrics.json").write_text(json.dumps({
            "version": 1, "seed_rows": n, "params": count_parameters(model),
            "pretrained": {"strict": round(acc["strict"], 4), "any_win": round(acc["any_win"], 4)},
            "baselines": baselines(test_rows), "scaling": scaling, "pretrain_history": res["history"], "runs": [],
        }, indent=1) + "\n")
        print(f"v1 trained on {n} rows: test strict {acc['strict']:.1%}, any-win {acc['any_win']:.1%}")


if __name__ == "__main__":
    main()
