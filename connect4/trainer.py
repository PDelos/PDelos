"""Supervised training with early stopping, shared by pretraining and the nightly run.

Overfitting controls: weight decay, mirror augmentation, dropout in the value head,
batch norm, a learning rate that halves when validation stalls, and early stopping on the validation
split (training stops once validation accuracy hasn't improved for `patience` epochs,
and the best epoch's weights are kept). The test set is never looked at here.
"""
from __future__ import annotations

import copy
import time

import torch

from .dataset import EvalSet, encode
from .model import loss_fn


def train(model, train_rows, val_rows, max_epochs: int = 40, patience: int = 3, lr: float = 2e-3,
          weight_decay: float = 1e-4, batch: int = 256, time_budget: float | None = None,
          log=print) -> dict:
    X, L, PI, Z = encode(train_rows)
    val = EvalSet(val_rows)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # halve the learning rate whenever validation accuracy stalls for an epoch
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    best = {"val": val.accuracy(model), "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    log(f"  start: val strict {best['val']['strict']:.1%}  ({len(train_rows)} train rows, {len(val_rows)} val rows)")
    t0, since_best, history = time.time(), 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            p, v = model(X[idx])
            loss, _, _ = loss_fn(p, v, PI[idx], Z[idx], L[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        acc = val.accuracy(model)
        sched.step(acc["strict"])
        history.append({"epoch": epoch, "loss": tot / len(X), **{f"val_{k}": v for k, v in acc.items()}})
        log(f"  epoch {epoch:2d}  loss {tot / len(X):.3f}  val strict {acc['strict']:.1%}  any-win {acc['any_win']:.1%}"
            f"  ({time.time() - t0:.0f}s)")
        if acc["strict"] > best["val"]["strict"]:
            best = {"val": acc, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            since_best = 0
        else:
            since_best += 1
        if since_best >= patience:
            log(f"  early stop: no validation improvement for {patience} epochs")
            break
        if time_budget and time.time() - t0 > time_budget:
            log("  stopping: time budget reached")
            break
    model.load_state_dict(best["state"])
    model.eval()
    return {"best_epoch": best["epoch"], "val": best["val"], "epochs_run": len(history),
            "seconds": round(time.time() - t0), "history": history}
