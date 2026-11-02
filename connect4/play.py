"""Process visitor moves submitted as GitHub issues titled "c4|move|<column>" or "c4|botfirst"
(the only setting a visitor can choose: let the bot make the first move of a new game).

Usage (from the move workflow):
    python -m connect4.play issues.json replies.json

issues.json is the output of `gh issue list --json number,title,author,createdAt`.
Every open move issue is handled oldest first, so a run that GitHub cancelled
or skipped is picked up by the next one. replies.json lists the comment to post
on each issue before it is closed.
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

from .game import Board
from .mcts import choose, search, tree_positions
from .net import NumpyNet
from .render import render_readme
from .state import MODEL_FILE, append_game, load_state, new_game, now, queue_positions, save_state

SIMULATIONS = 400
TREE_POSITIONS_PER_MOVE = 40   # positions the search explored, queued for the solver
GAME_POSITIONS_PER_GAME = 4    # positions sampled from the game itself (not all: they're correlated)
MIN_PLY = 8                    # earlier positions take the solver too long
TITLE = re.compile(r"^c4\|(?:move\|([1-7])|botfirst)$")


def bot_move(state: dict, board: Board, net: NumpyNet, rng: np.random.Generator) -> int:
    pi, root = search(board, net, SIMULATIONS, return_root=True)
    queue_positions("tree", tree_positions(root, board, TREE_POSITIONS_PER_MOVE, rng, min_visits=1, min_ply=MIN_PLY))
    # a little randomness in the opening so games don't all repeat
    c = choose(pi, temperature=0.6 if board.moves < 4 else 0.0, rng=rng)
    # what the README shows about this decision: instinct, search and evaluation
    prior = [0.0] * 7
    for col, child in root.children.items():
        prior[col] = round(child.prior, 4)
    state["analysis"] = {
        "game": state["game"]["id"], "move": board.moves + 1, "col": c + 1, "sims": SIMULATIONS,
        "visits": [round(float(v), 4) for v in pi], "policy": prior,
        "value": round(-root.children[c].q(), 4),  # bot's expected result after its move, in [-1, 1]
        "wins_now": board.is_winning_move(c),
    }
    return c


def finish(state: dict, board: Board, result: str) -> None:
    g = state["game"]
    players = sorted({e["user"] for e in g["log"] if e["who"] == "human"})
    append_game({"id": g["id"], "moves": board.history, "bot_first": g["bot_first"],
                 "result": result, "players": players, "started": g["started"], "ended": now()})
    candidates = [board.history[:i] for i in range(MIN_PLY, board.moves)]
    if candidates:
        pick = np.random.default_rng().choice(len(candidates), size=min(GAME_POSITIONS_PER_GAME, len(candidates)),
                                              replace=False)
        queue_positions("game", [candidates[i] for i in sorted(pick)])
    s = state["stats"]
    s["games"] += 1
    s[{"bot": "bot_wins", "human": "human_wins", "draw": "draws"}[result]] += 1
    state["recent"] = ([{"id": g["id"], "result": result, "moves": board.history}] + state["recent"])[:5]


def start_next_game(state: dict) -> None:
    """New games start with the visitor to move; they can hand the first move to the bot."""
    state["game"] = new_game(state["game"]["id"] + 1, bot_first=False)


def handle_bot_first(state: dict, user: str, net: NumpyNet, rng: np.random.Generator) -> str:
    g = state["game"]
    if g["moves"]:
        return "The bot can only start a fresh game, and this one is already under way. Your move in the README!"
    g["bot_first"] = True
    board = Board()
    c = bot_move(state, board, net, rng)
    board.play(c)
    g["moves"] = board.history
    g["log"].append({"who": "bot", "col": c + 1, "requested_by": user})
    return f"🟡 The bot opened game #{g['id']} in column {c + 1}. Your move! Head back to the README to answer."


def handle_move(state: dict, user: str, col: int, net: NumpyNet, rng: np.random.Generator) -> str:
    g = state["game"]
    board = Board.from_sequence(g["moves"])
    human_turn = (board.moves % 2 == 1) if g["bot_first"] else (board.moves % 2 == 0)
    if not human_turn:
        return "It wasn't the humans' turn, so this move was skipped. Try again from the README!"
    if not board.can_play(col):
        return f"Column {col + 1} is already full, so the move was skipped. Pick another column in the README!"

    won = board.is_winning_move(col)
    board.play(col)
    g["log"].append({"who": "human", "user": user, "col": col + 1})
    state["players"][user] = state["players"].get(user, 0) + 1

    if won:
        finish(state, board, "human")
        start_next_game(state)
        return (f"🔴 Column {col + 1}: **you won game #{g['id']}!** 🎉 The bot will study this game "
                "in tonight's training run. A new game has started in the README.")
    if board.is_full():
        finish(state, board, "draw")
        start_next_game(state)
        return f"🔴 Column {col + 1}: the board is full, game #{g['id']} is a **draw**. A new game has started."

    reply = bot_move(state, board, net, rng)
    bot_won = board.is_winning_move(reply)
    board.play(reply)
    g["log"].append({"who": "bot", "col": reply + 1})
    g["moves"] = board.history

    if bot_won:
        finish(state, board, "bot")
        start_next_game(state)
        return (f"🔴 You played column {col + 1}, 🟡 the bot answered in column {reply + 1} and **won game "
                f"#{g['id']}**. 🤖 Want revenge? A new game has started in the README.")
    if board.is_full():
        finish(state, board, "draw")
        start_next_game(state)
        return f"🔴 Column {col + 1}, 🟡 column {reply + 1}: the board is full, game #{g['id']} is a **draw**."
    return (f"🔴 You played column {col + 1} and 🟡 the bot answered in column {reply + 1}. "
            "Your move! Head back to the README to continue.")


def main(issues_path: str, replies_path: str) -> None:
    issues = json.loads(open(issues_path).read())
    issues = sorted((i for i in issues if TITLE.match(i["title"].strip())), key=lambda i: i["createdAt"])
    state = load_state()
    net = NumpyNet(MODEL_FILE)
    rng = np.random.default_rng()
    replies = []
    for issue in issues:
        match = TITLE.match(issue["title"].strip())
        user = issue["author"]["login"]
        if match.group(1):
            msg = handle_move(state, user, int(match.group(1)) - 1, net, rng)
        else:
            msg = handle_bot_first(state, user, net, rng)
        replies.append({"number": issue["number"], "body": f"@{user} {msg}"})
    save_state(state)
    render_readme(state)
    with open(replies_path, "w") as f:
        json.dump(replies, f)
    print(f"processed {len(replies)} move(s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
