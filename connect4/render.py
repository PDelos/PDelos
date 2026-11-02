"""Rewrites the live Connect 4 block of README.md and draws its two graphics:
assets/c4_mind.svg (the bot's last decision + evaluation bar) and assets/c4_progress.svg
(test accuracy as the train set grows). The static story and math around it stay hand-written."""
from __future__ import annotations

import json
import math
from urllib.parse import quote

from .game import Board
from .state import CHART_FILE, MIND_FILE, README_FILE, load_metrics

START, END = "<!-- C4:START -->", "<!-- C4:END -->"
HUMAN, BOT, EMPTY = "🔴", "🟡", "⚪"
ISSUE_BODY = ("Just submit this issue as it is, no need to write anything. Within a minute a GitHub Action will "
              "make the move, the bot will answer, and the board on my profile will update.")


def issue_link(repo: str, title: str) -> str:
    return f"https://github.com/{repo}/issues/new?title={quote(title)}&body={quote(ISSUE_BODY)}"


def board_table(state: dict) -> str:
    g = state["game"]
    board = Board.from_sequence(g["moves"])
    grid = board.grid()
    bot_piece = 1 if g["bot_first"] else 2
    header = []
    for c in range(7):
        header.append(f"[**{c + 1}**]({issue_link(state['repo'], f'c4|move|{c + 1}')})" if board.can_play(c) else "—")
    lines = ["| " + " | ".join(header) + " |", "|" + ":-:|" * 7]
    for row in grid:
        cells = [EMPTY if v == 0 else (BOT if v == bot_piece else HUMAN) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def status_line(state: dict) -> str:
    g = state["game"]
    if not g["moves"]:
        parts = [f"**Game #{g['id']}**: you play {HUMAN} and move first. Click a column number to drop a piece, "
                 f"or [**let the bot start** {BOT}]({issue_link(state['repo'], 'c4|botfirst')})."]
    else:
        parts = [f"**Game #{g['id']}**: your turn, you play {HUMAN}. Click a column number to drop a piece."]
    log = g["log"]
    if log:
        last_h = next((e for e in reversed(log) if e["who"] == "human"), None)
        last_b = next((e for e in reversed(log) if e["who"] == "bot"), None)
        bits = []
        if last_h:
            bits.append(f"@{last_h['user']} played column {last_h['col']}")
        if last_b:
            bits.append(f"the bot answered in column {last_b['col']}" if last_h else f"the bot opened in column {last_b['col']}")
        parts.append("Last moves: " + ", ".join(bits) + ".")
    if state["recent"]:
        r = state["recent"][0]
        word = {"bot": "the bot won 🤖", "human": "humans won 🎉", "draw": "it was a draw 🤝"}[r["result"]]
        parts.append(f"In game #{r['id']}, {word}.")
    return " ".join(parts)


def model_line() -> str:
    m = load_metrics()
    pre = m.get("pretrained")
    if not pre:
        return ""
    runs = m["runs"]
    last = runs[-1] if runs else None
    rows_now = last["train_rows"] if last else m.get("seed_rows", 0)
    by_src = last["rows_by_source"] if last else {"seed": m.get("seed_rows", 0)}
    from_games = by_src.get("game", 0) + by_src.get("tree", 0)
    net = {"strict": last["test_strict"], "any_win": last["test_any_win"]} if last else pre
    srch = ({"strict": last["search_strict"], "any_win": last["search_any_win"]} if last
            else m.get("pretrained_search"))
    lines = [f"**Model v{m['version']}** · {m.get('params', 0) / 1e6:.2f}M parameters · trained on **{rows_now:,}** "
             f"solver-labelled positions, **{from_games:,}** of them from games against visitors"]
    table = ["", "| Picks… | network alone | network + tree search |", "|:--|:-:|:-:|",
             f"| a perfect move (fastest win) | **{net['strict']:.1%}** | **{srch['strict']:.1%}** |" if srch else "",
             f"| a winning move | **{net['any_win']:.1%}** | **{srch['any_win']:.1%}** |" if srch else ""]
    lines.append("\n".join(t for t in table if t is not None))
    ref = m.get("search_reference")
    if ref:
        lines.append(f"<sub>Measured on 10,000 fixed test positions it never trains on (tree search: 400 of them, "
                     f"400 simulations each). It started at {pre['strict']:.1%} / {pre['any_win']:.1%} on "
                     f"{m.get('seed_rows', 0):,} positions; in testing, the same network trained on "
                     f"{ref['train_rows']:,} positions reached {ref['search']['any_win']:.0%} winning moves with search."
                     + (f" {sum(1 for r in runs if r['accepted'])} of {sum(1 for r in runs if r.get('trained', True))} "
                        f"retrains accepted (it retrains each time the train set grows 10%); last nightly run "
                        f"{last['date']}." if runs else " It retrains each time the train set grows 10%.") + "</sub>")
    return "\n\n".join(lines)


def stats_line(state: dict) -> str:
    s = state["stats"]
    top = sorted(state["players"].items(), key=lambda kv: -kv[1])[:3]
    def n(k: int, word: str) -> str:
        return f"{k} {word}{'' if k == 1 else 's'}"

    line = (f"**Record vs humans:** 🤖 bot {n(s['bot_wins'], 'win')} · 🧑 humans {n(s['human_wins'], 'win')} · "
            f"🤝 {n(s['draws'], 'draw')} ({n(s['games'], 'game')})")
    if top:
        line += " · **Most moves:** " + ", ".join(f"@{u} ({n})" for u, n in top)
    return line


def section(state: dict) -> str:
    return f"""{START}
### ▶️ Play it

{status_line(state)}

{board_table(state)}

{stats_line(state)}

### 🧠 What it was thinking

![The bot's last decision: its evaluation and where its search looked](assets/c4_mind.svg)

### 📈 How your games make it better

![Test accuracy as the train set grows](assets/c4_progress.svg)

{model_line()}
{END}"""


def render_readme(state: dict) -> None:
    text = README_FILE.read_text() if README_FILE.exists() else ""
    new = section(state)
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + new + after
    else:
        text = text.rstrip() + "\n\n" + new + "\n"
    README_FILE.write_text(text)
    render_mind(state)
    render_chart()


def render_mind(state: dict) -> None:
    """Evaluation bar plus a per-column view of the bot's last decision: the network's instinct
    (dashed marks) against where the tree search spent its simulations (bars)."""
    a = state.get("analysis")
    w, h = 760, 300
    css = """<style>
.bg{fill:#ffffff} .title{fill:#1f2328;font-size:15px;font-weight:600} .lbl{fill:#59636e;font-size:11px}
.txt{fill:#1f2328;font-size:12px} .big{fill:#1f2328;font-size:13px;font-weight:600}
.you{fill:#cf222e} .bot{fill:#d4a72c} .track{fill:#eaeef2} .grid{stroke:#d1d9e0;stroke-width:1}
.vis{fill:#d4a72c} .vis.pick{fill:#bf8700} .pol{stroke:#1f2328;stroke-width:2;stroke-dasharray:4 3}
.pickbox{fill:none;stroke:#bf8700;stroke-width:1.5;stroke-dasharray:3 3} .mid{stroke:#1f2328;stroke-width:1.5}
@media (prefers-color-scheme: dark) {
 .bg{fill:#0d1117} .title,.txt,.big{fill:#f0f6fc} .lbl{fill:#9198a1} .track{fill:#21262d} .grid{stroke:#3d444d}
 .you{fill:#ff7b72} .bot,.vis{fill:#e3b341} .vis.pick{fill:#f2cc60} .pol,.mid{stroke:#f0f6fc} .pickbox{stroke:#f2cc60}
}
</style>"""
    out = [f'<rect width="{w}" height="{h}" rx="10" class="bg"/>']
    if not a:
        out.append(f'<text x="24" y="30" class="title">The bot hasn\'t made a move yet</text>'
                   f'<text x="24" y="56" class="txt">After its first move, this panel shows how it judged the position '
                   f'and which columns its search considered.</text>')
    else:
        v = max(-1.0, min(1.0, a["value"]))
        bot_share = (v + 1) / 2
        if a.get("wins_now"):
            verdict = "it has just connected four"
        elif v > 0.6:
            verdict = "it thinks it is winning"
        elif v > 0.2:
            verdict = "it thinks it has the edge"
        elif v >= -0.2:
            verdict = "it thinks the game is balanced"
        elif v >= -0.6:
            verdict = "it thinks you have the edge"
        else:
            verdict = "it thinks you are winning"
        if a.get("preview"):
            out.append(f'<text x="24" y="30" class="title">Before anyone has played: on an empty board the bot would open '
                       f'in column {a["col"]}</text>')
        else:
            out.append(f'<text x="24" y="30" class="title">Game #{a["game"]}, move {a["move"]}: the bot played column '
                       f'{a["col"]} after {a["sims"]} simulations, and {verdict}</text>')
        # evaluation bar
        x0, x1, y0 = 24, w - 24, 52
        split = x0 + (x1 - x0) * (1 - bot_share)
        out.append(f'<text x="{x0}" y="{y0 - 6}" class="lbl">EVALUATION · expected result if both sides play like the bot '
                   f'from here</text>')
        out.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="22" rx="5" class="track"/>'
                   f'<rect x="{x0}" y="{y0}" width="{split - x0:.1f}" height="22" rx="5" class="you"/>'
                   f'<rect x="{split:.1f}" y="{y0}" width="{x1 - split:.1f}" height="22" rx="5" class="bot"/>'
                   f'<line x1="{(x0 + x1) / 2}" x2="{(x0 + x1) / 2}" y1="{y0 - 3}" y2="{y0 + 25}" class="mid"/>')
        out.append(f'<circle cx="{x0 + 6}" cy="{y0 + 36}" r="6" class="you"/>'
                   f'<text x="{x0 + 18}" y="{y0 + 40}" class="big">you {1 - bot_share:.0%}</text>'
                   f'<circle cx="{x1 - 6}" cy="{y0 + 36}" r="6" class="bot"/>'
                   f'<text x="{x1 - 18}" y="{y0 + 40}" text-anchor="end" class="big">bot {bot_share:.0%}</text>')
        # decision: per-column instinct vs search
        top, base = 130, 262
        out.append(f'<text x="{x0}" y="{top - 8}" class="lbl">DECISION · bars: share of the search spent on each column · '
                   f'dashes: the network\'s instinct before searching</text>')
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{base}" y2="{base}" class="grid"/>')
        colw = (x1 - x0) / 7
        for c in range(7):
            cx = x0 + colw * (c + 0.5)
            share, prior = a["visits"][c], a["policy"][c]
            bh = (base - top - 16) * share
            pick = c + 1 == a["col"]
            if pick:
                out.append(f'<rect x="{cx - colw / 2 + 4:.1f}" y="{top:.1f}" width="{colw - 8:.1f}" '
                           f'height="{base - top + 2}" rx="6" class="pickbox"/>')
            out.append(f'<rect x="{cx - colw * 0.3:.1f}" y="{base - bh:.1f}" width="{colw * 0.6:.1f}" '
                       f'height="{max(bh, 0.5):.1f}" rx="3" class="vis{" pick" if pick else ""}"/>')
            py = base - (base - top - 16) * prior
            if prior > 0.002:
                out.append(f'<line x1="{cx - colw * 0.38:.1f}" x2="{cx + colw * 0.38:.1f}" y1="{py:.1f}" y2="{py:.1f}" '
                           f'class="pol"/>')
            if share >= 0.01:
                ty = min(base - bh, py if prior > 0.002 else base) - 6
                out.append(f'<text x="{cx:.1f}" y="{ty:.1f}" text-anchor="middle" class="txt">{share:.0%}</text>')
            out.append(f'<text x="{cx:.1f}" y="{base + 18}" text-anchor="middle" class="{"big" if pick else "lbl"}">'
                       f'{("would play " if a.get("preview") else "played ") if pick else ""}{c + 1}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           f'font-family="ui-sans-serif,system-ui,sans-serif">\n{css}\n' + "\n".join(out) + "\n</svg>\n")
    MIND_FILE.parent.mkdir(exist_ok=True)
    MIND_FILE.write_text(svg)


def render_chart() -> None:
    """Test accuracy against train-set size: the offline scaling test as the expected path,
    the live model's nightly results on top."""
    m = load_metrics()
    pre = m.get("pretrained") or {"strict": 0, "any_win": 0}
    seed_rows = m.get("seed_rows", 1000)
    scaling = m.get("scaling", [])
    runs = m["runs"]
    retrains = [r for r in runs if r.get("trained", True)]
    live = [(seed_rows, pre["strict"], pre["any_win"], True)] + \
           [(r["train_rows"], r.get("candidate_strict", r["test_strict"]), r.get("candidate_any_win", r["test_any_win"]),
             r["accepted"]) for r in retrains]
    xs_all = [seed_rows] + [s["train_rows"] for s in scaling] + [p[0] for p in live]
    xmin = 10 ** math.floor(math.log10(min(xs_all)))
    xmax = max(10 ** math.ceil(math.log10(max(xs_all) * 1.05)), xmin * 100)
    w, h, left, right, top, bottom = 760, 320, 56, 150, 50, 50
    lo, hi = 0.4, 1.0

    def X(n: float) -> float:
        return left + (w - left - right) * (math.log10(n) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))

    def Y(v: float) -> float:
        return top + (h - top - bottom) * (1 - (min(max(v, lo), hi) - lo) / (hi - lo))

    out = []
    for k in range(7):
        v = lo + 0.1 * k
        out.append(f'<line x1="{left}" x2="{w - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{left - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" class="lbl">{v:.0%}</text>')
    n = xmin
    while n <= xmax:
        for mult in (1, 2, 5):
            t = n * mult
            if xmin <= t <= xmax:
                label = f"{t / 1e6:g}M" if t >= 1e6 else (f"{t // 1000:,.0f}k" if t >= 1000 else str(int(t)))
                out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{h - bottom}" class="vgrid"/>'
                           f'<text x="{X(t):.1f}" y="{h - bottom + 16}" text-anchor="middle" class="lbl">{label}</text>')
        n *= 10
    out.append(f'<text x="{(left + w - right) / 2:.0f}" y="{h - 12}" text-anchor="middle" class="lbl">'
               'positions in the train set (log scale)</text>')
    yv = Y(0.95)
    out.append(f'<line x1="{left}" x2="{w - right}" y1="{yv:.1f}" y2="{yv:.1f}" class="goal"/>'
               f'<text x="{left + 6}" y="{yv - 5:.1f}" class="lbl">goal 95%</text>')

    def path(pts, cls):
        return f'<path class="{cls}" d="' + " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}"
                                                     for i, (a, b) in enumerate(pts)) + '"/>'

    if scaling:
        out.append(path([(s["train_rows"], s["test_any_win"]) for s in scaling], "exp2"))
        out.append(path([(s["train_rows"], s["test_strict"]) for s in scaling], "exp1"))
    if len(live) > 1:
        out.append(path([(p[0], p[2]) for p in live], "anyw"))
        out.append(path([(p[0], p[1]) for p in live], "strict"))
    for rows, st, aw, acc in live:
        cls_s, cls_a = ("dot", "dot2") if acc else ("rej", "rej2")
        out.append(f'<circle cx="{X(rows):.1f}" cy="{Y(aw):.1f}" r="3.5" class="{cls_a}"/>'
                   f'<circle cx="{X(rows):.1f}" cy="{Y(st):.1f}" r="4" class="{cls_s}"/>')
    last = live[-1]
    lx = w - right + 10
    out.append(f'<text x="{lx}" y="{Y(last[2]) + 4:.1f}" class="val2">winning move {last[2]:.1%}</text>'
               f'<text x="{lx}" y="{Y(last[1]) + 4:.1f}" class="val">perfect move {last[1]:.1%}</text>')
    if scaling:
        out.append(f'<text x="{lx}" y="{Y(scaling[-1]["test_strict"]) + 22:.1f}" class="lbl">dashed: scaling test</text>')
    sub = (f"{len(retrains)} retrains · {runs[-1]['train_rows']:,} positions · "
           f"{runs[-1]['human_games_total']} games vs visitors" if runs else "dots: live model · dashed: offline scaling test")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="ui-sans-serif,system-ui,sans-serif">
<style>
.bg{{fill:#ffffff}} .title{{fill:#1f2328;font-size:15px;font-weight:600}} .lbl{{fill:#59636e;font-size:11px}}
.val{{fill:#cf222e;font-size:12px;font-weight:600}} .val2{{fill:#9a6700;font-size:12px;font-weight:600}}
.grid{{stroke:#d1d9e0;stroke-width:1}} .vgrid{{stroke:#eaeef2;stroke-width:1}} .goal{{stroke:#1a7f37;stroke-width:1.2;stroke-dasharray:2 3}}
.strict{{fill:none;stroke:#cf222e;stroke-width:2.5;stroke-linejoin:round}} .anyw{{fill:none;stroke:#bf8700;stroke-width:2;stroke-linejoin:round}}
.exp1{{fill:none;stroke:#cf222e;stroke-width:1.4;stroke-dasharray:5 4;opacity:.55}} .exp2{{fill:none;stroke:#bf8700;stroke-width:1.4;stroke-dasharray:5 4;opacity:.55}}
.dot{{fill:#cf222e}} .dot2{{fill:#bf8700}} .rej{{fill:#ffffff;stroke:#cf222e;stroke-width:2}} .rej2{{fill:#ffffff;stroke:#bf8700;stroke-width:2}}
@media (prefers-color-scheme: dark) {{
 .bg{{fill:#0d1117}} .title{{fill:#f0f6fc}} .lbl{{fill:#9198a1}} .grid{{stroke:#3d444d}} .vgrid{{stroke:#21262d}}
 .val{{fill:#ff7b72}} .val2{{fill:#d29922}} .goal{{stroke:#3fb950}}
 .strict,.exp1{{stroke:#ff7b72}} .anyw,.exp2{{stroke:#d29922}} .dot{{fill:#ff7b72}} .dot2{{fill:#d29922}}
 .rej{{fill:#0d1117;stroke:#ff7b72}} .rej2{{fill:#0d1117;stroke:#d29922}}
}}
</style>
<rect width="{w}" height="{h}" rx="10" class="bg"/>
<text x="{left - 30}" y="26" class="title">Test accuracy as visitors' games grow the train set</text>
<text x="{w - 18}" y="26" text-anchor="end" class="lbl">{sub}</text>
{chr(10).join(out)}
</svg>
"""
    CHART_FILE.parent.mkdir(exist_ok=True)
    CHART_FILE.write_text(svg)


if __name__ == "__main__":
    from .state import load_state

    render_readme(load_state())
    print(json.dumps(load_state()["stats"]))
