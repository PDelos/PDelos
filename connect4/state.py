"""Persistent game state, stored as JSON/JSONL files in the repo."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATE_FILE = DATA / "state.json"
GAMES_FILE = DATA / "games.jsonl"
PENDING_FILE = DATA / "pending.txt"     # positions waiting for the nightly solver review
TRAIN_FILE = DATA / "train.txt"         # the growing, solver-labelled train set
TEST_FILE = DATA / "test_set.txt"       # 10,000 fixed positions, never trained on
CHECKPOINT_FILE = DATA / "checkpoint.pt"
METRICS_FILE = DATA / "metrics.json"
MODEL_FILE = DATA / "model.npz"
CHART_FILE = ROOT / "assets" / "c4_progress.svg"
MIND_FILE = ROOT / "assets" / "c4_mind.svg"
README_FILE = ROOT / "README.md"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_game(game_id: int, bot_first: bool) -> dict:
    return {"id": game_id, "moves": "", "bot_first": bot_first, "log": [], "started": now()}


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "repo": "PDelos/PDelos",
        "game": new_game(1, bot_first=False),
        "stats": {"games": 0, "bot_wins": 0, "human_wins": 0, "draws": 0},
        "recent": [],
        "players": {},
    }


def save_state(state: dict) -> None:
    DATA.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1) + "\n")


def append_game(record: dict) -> None:
    with GAMES_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def load_games() -> list[dict]:
    if not GAMES_FILE.exists():
        return []
    return [json.loads(l) for l in GAMES_FILE.read_text().splitlines() if l.strip()]


def queue_positions(source: str, seqs: list[str]) -> None:
    if seqs:
        with PENDING_FILE.open("a") as f:
            f.writelines(f"{source} {s}\n" for s in seqs)


def load_metrics() -> dict:
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text())
    return {"version": 1, "runs": []}


def save_metrics(m: dict) -> None:
    METRICS_FILE.write_text(json.dumps(m, indent=1) + "\n")
