"""Builds the profile README's terminal windows as SVGs, one light and one dark version of each.

Every section is a terminal window whose title is the section name; commands print the content.
Logos and icons are drawn as square "terminal pixels". The font (JetBrains Mono, OFL) is subset to
the characters used and embedded, because GitHub serves these SVGs as images and blocks external files.

    pip install -r profile/requirements.txt
    python profile/build.py          # writes assets/{light,dark}/*.svg and README.md
"""
from __future__ import annotations

import base64
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = ROOT / "assets"

# ---------------------------------------------------------------------------------------------------- content

USER, HOST = "pol", "barcelona"

ABOUT = {
    "whoami": ("pol de los santos", "software, data & ML engineer · barcelona"),
    "about": [
        "I'm studying two degrees at once, data science & engineering and biomedical",
        "engineering, through CFIS, the excellence programme of UPC in Barcelona.",
        "",
        "Two internships at Google, Mixture-of-Experts models on custom AI chips at",
        "Esperanto, and ML for oncology trials at Science4Tech. What drives me is",
        "curiosity: I like pushing a technology until I find out what it can really do.",
    ],
    "status": "graduating jun 2027 · open to software, ML & data roles",
}

LINKS = [  # label, url (the buttons under the about window)
    ("cv", "https://github.com/PDelos/PDelos/blob/main/cv/Pol_de_los_Santos_CV.pdf"),
    ("mail", "mailto:delossantospol001@gmail.com"),
    ("portfolio", "https://polsantos.com"),
    ("linkedin", "https://www.linkedin.com/in/pol-de-los-santos-subirats/"),
]

CFIS = [
    "CFIS (Centre de Formació Interdisciplinària Superior) is UPC's excellence programme:",
    "admission by exam, and students take two official UPC bachelor's degrees in parallel.",
]

EDUCATION = [
    {"file": "data-science", "icon": "chart", "title": "data science & engineering",
     "fields": [("degree", "BSc Data Science and Engineering"), ("where", "UPC · CFIS · barcelona"),
                ("status", "graduating jun 2027")],
     "notes": ["maths, statistics, algorithms and machine learning on real data"]},
    {"file": "biomedical", "icon": "heart", "title": "biomedical engineering",
     "fields": [("degree", "BSc Biomedical Engineering"), ("where", "UPC · CFIS · barcelona"),
                ("status", "graduating jun 2027")],
     "notes": ["engineering applied to medicine and biology, taken alongside data science"]},
    {"file": "high-school", "icon": "cap", "title": "ins jaume vicens vives",
     "fields": [("track", "CiMs-Cellex excellence track"), ("where", "girona"),
                ("grades", "spanish baccalaureate 13.167/14 · IB 40/45")],
     "notes": ["completed both the Spanish Scientific Baccalaureate and the International",
               "Baccalaureate, with a rigorous science and mathematics focus"]},
]

JOBS = [
    {"file": "2026-google", "logo": "google.png", "org": "google", "city": "munich",
     "fields": [("role", "Site Reliability Engineer Intern"), ("when", "jul 2026 – oct 2026"),
                ("stack", "c++ · sql · rpc · ai agents")],
     "notes": ["built RPC endpoints inside production binaries to make data pipelines observable",
               "engineers can now diagnose probe configuration issues on their own",
               "built an agent skill wrapper for automated debugging: +97% successful debugging runs"]},
    {"file": "2025-google", "logo": "google.png", "org": "google", "city": "zurich",
     "fields": [("role", "Associate Software Engineer Intern"), ("when", "jun 2025 – sep 2025"),
                ("stack", "python · sql · llms · rag")],
     "notes": ["built an AI system that turns natural-language questions into SQL",
               "retrieval-augmented generation maps user intent to the right database schemas",
               "query validation + targeted schema context cut errors and hallucinations"]},
    {"file": "2024-esperanto", "logo": "esperanto.png", "org": "esperanto", "city": "barcelona",
     "fields": [("role", "Data Engineer"), ("when", "jun 2024 – sep 2024"), ("stack", "python · c++ · pytorch")],
     "notes": ["edited MoE model graphs so they ran correctly on the company's own AI chips",
               "sped them up with quantization and sparse execution",
               "statistical analysis of expert gating to understand how the models route tokens"]},
    {"file": "2023-science4tech", "logo": "science4tech.png", "org": "science4tech", "city": "barcelona",
     "fields": [("role", "Data Scientist"), ("when", "jul 2023 – apr 2024"), ("stack", "python · sql · ml · dataviz")],
     "notes": ["analyzed administrative, public and clinical patient data for oncology trials",
               "built ML models for trial optimization, risk monitoring and efficiency prediction"]},
]

STACK = [
    ("languages", "python · c++ · rust · r · matlab · typescript · javascript"),
    ("ml", "pytorch · tensorflow · scikit-learn · numpy · pandas"),
    ("data", "postgresql · mysql · sqlite · mongodb · supabase"),
    ("web", "sveltekit · node.js · pocketbase · flutter · vercel"),
    ("infra", "docker · nginx · github actions · linux"),
    ("workflow", "nixos · arch · neovim · git · bash · latex"),
]

INTERESTS = [
    ("web development", "full-stack apps with sveltekit, pocketbase and docker"),
    ("ai research", "research-hub: agents that sweep papers, grade them and grow a knowledge base"),
    ("ai harnesses", "mypi: my own personal AI agent built on the pi agent harness"),
]

# ---------------------------------------------------------------------------------------------------- style

WIDTH = 840
FONT_SIZE = 13
CHAR = FONT_SIZE * 0.6            # JetBrains Mono advance width is 600/1000 em
LINE = 20
PAD_X = 28
TITLE_BAR = 34
LOGO = 76                          # logo / icon size in the fetch blocks
INFO_X = PAD_X + LOGO + 22         # where the text next to a logo starts

THEMES = {
    "dark": {"page": "#0d1117", "bg": "#0b0f0c", "chrome": "#151b17", "border": "#253028", "fg": "#d7e3d7",
             "dim": "#7d917f", "faint": "#2f3b32", "green": "#3ddc6f", "amber": "#ffb000", "red": "#ff5f57"},
    "light": {"page": "#ffffff", "bg": "#fbfaf5", "chrome": "#eeece2", "border": "#dcd9cc", "fg": "#1d2b1d",
              "dim": "#66766a", "faint": "#cfd3c5", "green": "#1a7f37", "amber": "#9a5b00", "red": "#d93a32"},
}

# 16x16 pixel icons; letters are theme colours: f=fg d=dim g=green a=amber r=red
ICONS = {
    "chart": """
................
............gg..
............gg..
............gg..
........gg..gg..
........gg..gg..
........gg..gg..
....gg..gg..gg..
....gg..gg..gg..
....gg..gg..gg..
d...gg..gg..gg..
d...gg..gg..gg..
d...gg..gg..gg..
dddddddddddddddd
................
................""",
    "heart": """
................
..rrrr....rrrr..
.rrrrrr..rrrrrr.
rrrrrrrrrrrrrrrr
rrrrrrrrrrrrrrrr
rrrrrrrfrrrrrrrr
rrrrrrfrfrrrrrrr
ffffffrrrfrfffff
.rrrrrrrrrfrrrr.
..rrrrrrrrrrrr..
...rrrrrrrrrr...
....rrrrrrrr....
.....rrrrrr.....
......rrrr......
.......rr.......
................""",
    "cap": """
................
................
.......ff.......
.....ffffff.....
...ffffffffff...
.ffffffffffffff.
...ffffffffffa..
.....ffffff..a..
....ffffffff.a..
....ffffffff.a..
....ffffffff.aa.
.....ffffff..aa.
................
................
................
................""",
}

# ---------------------------------------------------------------------------------------------------- drawing


def text(x: float, y: float, spans, size: int = FONT_SIZE, anchor: str = "start") -> str:
    """spans: (string, colour, bold) triples in one <text>, so monospace columns line up exactly."""
    inner = "".join(f'<tspan fill="{color}"{" font-weight=\"700\"" if bold else ""}>{escape(s)}</tspan>'
                    for s, color, bold in spans)
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" xml:space="preserve">{inner}</text>'


class Window:
    """A terminal window drawn top to bottom; `y` is the baseline of the next line."""

    def __init__(self, colors: dict, title: str):
        self.c, self.title = colors, title
        self.rows: list[str] = []
        self.y = 8 + TITLE_BAR + 28

    def line(self, spans, x: float = PAD_X, gap: float = LINE) -> None:
        content = "".join(s for s, _, _ in spans)
        if x + CHAR * len(content) > WIDTH - PAD_X:
            raise ValueError(f"line too wide for the {self.title!r} window, shorten it: {content!r}")
        self.rows.append(text(x, self.y, spans))
        self.y += gap

    def prompt(self, command: str, cwd: str = "~") -> None:
        c = self.c
        self.line([(f"{USER}@{HOST}", c["green"], True), (f":{cwd}$ ", c["dim"], False), (command, c["fg"], False)])

    def blank(self, height: float = 12) -> None:
        self.y += height

    def cursor(self) -> None:
        x = PAD_X + CHAR * len(f"{USER}@{HOST}:~$ ")
        self.prompt("")
        self.rows.append(f'<rect x="{x:.1f}" y="{self.y - LINE - 11:.1f}" width="{CHAR:.1f}" height="15" '
                         f'fill="{self.c["green"]}"><animate attributeName="opacity" values="1;1;0;0" dur="1.1s" '
                         'repeatCount="indefinite"/></rect>')

    def fetch(self, art: str, header, fields, notes) -> None:
        """neofetch-style block: pixel art on the left, a header, key/value fields and notes on the right."""
        c = self.c
        top = self.y - 6
        self.rows.append(art)
        y = top + 14
        self.rows.append(text(INFO_X, y, header))
        y += 8
        self.rows.append(f'<line x1="{INFO_X}" x2="{INFO_X + CHAR * 28:.1f}" y1="{y}" y2="{y}" stroke="{c["faint"]}"/>')
        for key, value in fields:
            y += LINE - 2
            first = key == fields[0][0]
            self.rows.append(text(INFO_X, y, [(f"{key:<7}", c["amber"], True), (value, c["fg"] if first else c["dim"], first)]))
        self.y = max(top + LOGO, y) + 8 + LINE - 2
        width = int((WIDTH - PAD_X - INFO_X) / CHAR) - 3
        for note in notes:
            for i, part in enumerate(textwrap.wrap(note, width)):
                self.line([("› " if i == 0 else "  ", c["green"], True), (part, c["fg"], False)], INFO_X, LINE - 2)
        self.blank(16)

    def render(self) -> tuple[int, str]:
        c, height = self.c, int(self.y + 4)
        w = WIDTH - 16
        frame = [
            f'<rect x="8" y="8" width="{w}" height="{height - 16}" rx="10" fill="{c["bg"]}" stroke="{c["border"]}"/>',
            f'<path d="M8 18a10 10 0 0 1 10-10h{w - 20}a10 10 0 0 1 10 10v{TITLE_BAR - 10}H8z" fill="{c["chrome"]}"/>',
            f'<line x1="8" x2="{8 + w}" y1="{8 + TITLE_BAR}" y2="{8 + TITLE_BAR}" stroke="{c["border"]}"/>',
            "".join(f'<circle cx="{30 + 18 * i}" cy="{8 + TITLE_BAR / 2}" r="5.5" fill="{col}"/>'
                    for i, col in enumerate(("#ff5f57", "#febc2e", "#28c840"))),
            text(WIDTH / 2, 8 + TITLE_BAR / 2 + 4, [(self.title, c["dim"], True)], 12, "middle"),
        ]
        return height, "\n".join(frame + self.rows)


def pixel_grid(cells, x: float, y: float, grid: int) -> str:
    """cells: {(col, row): colour}; draws them as square 'terminal pixels' filling a LOGO-sized box."""
    size = LOGO / grid
    return "".join(f'<rect x="{x + gx * size:.1f}" y="{y + gy * size:.1f}" width="{size - 0.8:.1f}" '
                   f'height="{size - 0.8:.1f}" fill="{color}"/>' for (gx, gy), color in cells.items())


def logo_pixels(colors: dict, path: Path, x: float, y: float, grid: int = 16) -> str:
    img = Image.open(path).convert("RGBA")
    if img.getpixel((0, 0))[:3] == (255, 255, 255):    # flat white background -> transparent
        red, green, blue, alpha = img.split()
        darkest = ImageChops.darker(ImageChops.darker(red, green), blue)
        img.putalpha(ImageChops.multiply(alpha, darkest.point(lambda v: 0 if v > 235 else 255)))
    img = img.resize((grid, grid), Image.LANCZOS)
    cells = {}
    for gy in range(grid):
        for gx in range(grid):
            r, g, b, a = img.getpixel((gx, gy))
            if a >= 110:
                cells[(gx, gy)] = colors["fg"] if max(r, g, b) < 60 else f"#{r:02x}{g:02x}{b:02x}"  # black ink follows the theme
    return pixel_grid(cells, x, y, grid)


def icon_pixels(colors: dict, name: str, x: float, y: float) -> str:
    palette = {"f": colors["fg"], "d": colors["dim"], "g": colors["green"], "a": colors["amber"], "r": colors["red"]}
    rows = ICONS[name].strip("\n").splitlines()
    cells = {(gx, gy): palette[ch] for gy, row in enumerate(rows) for gx, ch in enumerate(row) if ch != "."}
    return pixel_grid(cells, x, y, len(rows))


# ---------------------------------------------------------------------------------------------------- windows


def about(c: dict) -> Window:
    w = Window(c, "01 · about")
    w.prompt("whoami")
    name, tagline = ABOUT["whoami"]
    w.line([(name, c["fg"], True), (f" — {tagline}", c["dim"], False)])
    w.blank()
    w.prompt("cat about.txt")
    for line in ABOUT["about"]:
        w.line([(line, c["fg"], False)], gap=LINE if line else 10)
    w.blank()
    w.prompt("echo $STATUS")
    w.line([("● ", c["green"], False), (ABOUT["status"], c["fg"], False)])
    w.blank()
    w.prompt("ls ~/links")
    w.line([("   ".join(label for label, _ in LINKS), c["green"], True), ("    ↓ click below", c["dim"], False)])
    w.blank(4)
    return w


def education(c: dict) -> Window:
    w = Window(c, "02 · education")
    w.prompt("cat ~/education/cfis.txt")
    for line in CFIS:
        w.line([(line, c["fg"], False)])
    w.blank()
    for entry in EDUCATION:
        w.prompt(f"fetch {entry['file']}", "~/education")
        art = icon_pixels(c, entry["icon"], PAD_X, w.y - 6)
        w.fetch(art, [(entry["title"], c["green"], True)], entry["fields"], entry["notes"])
    return w


def experience(c: dict) -> Window:
    w = Window(c, "03 · experience")
    w.prompt("ls ~/career")
    w.line([("   ".join(job["file"] for job in JOBS), c["green"], True)])
    w.blank()
    for job in JOBS:
        w.prompt(f"fetch {job['file']}", "~/career")
        art = logo_pixels(c, HERE / "logos" / job["logo"], PAD_X, w.y - 6)
        header = [(job["org"], c["green"], True), ("@", c["dim"], False), (job["city"], c["green"], True)]
        w.fetch(art, header, job["fields"], job["notes"])
    return w


def stack(c: dict) -> Window:
    w = Window(c, "04 · stack & interests")
    w.prompt("tree ~/stack")
    w.line([("~/stack", c["green"], True)])
    width = max(len(name) for name, _ in STACK) + 3
    for i, (name, tools) in enumerate(STACK):
        branch = "└── " if i == len(STACK) - 1 else "├── "
        w.line([(branch, c["faint"], False), (f"{name:<{width}}", c["amber"], True), (tools, c["fg"], False)])
    w.blank()
    w.prompt("cat ~/now.md")
    w.line([("# currently exploring", c["dim"], True)])
    width = max(len(name) for name, _ in INTERESTS) + 3
    for name, what in INTERESTS:
        w.line([("› ", c["green"], True), (f"{name:<{width}}", c["fg"], True), (what, c["dim"], False)])
    w.blank()
    w.cursor()
    w.blank(2)
    return w


def button(c: dict, label: str) -> tuple[int, int, str]:
    """A small clickable-looking key for one link; the link itself lives in the README."""
    width, height = 150, 40
    body = (f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="7" fill="{c["bg"]}" stroke="{c["border"]}"/>'
            + text(width / 2, height / 2 + 5, [("> ", c["green"], True), (label, c["fg"], True)], 13, "middle"))
    return width, height, body


# ---------------------------------------------------------------------------------------------------- output


def font_css(chars: str) -> str:
    rules = []
    for weight, name in ((400, "Regular"), (700, "Bold")):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "font.woff2"
            subprocess.run([sys.executable, "-m", "fontTools.subset", str(HERE / "fonts" / f"JetBrainsMono-{name}.ttf"),
                            f"--text={chars}", "--flavor=woff2", f"--output-file={out}"], check=True)
            data = base64.b64encode(out.read_bytes()).decode()
        rules.append(f"@font-face{{font-family:'JetBrains Mono';font-weight:{weight};"
                     f"src:url(data:font/woff2;base64,{data}) format('woff2')}}")
    return "".join(rules)


def svg(width: int, height: int, body: str, page: str | None, css: str) -> str:
    background = f'<rect width="{width}" height="{height}" fill="{page}"/>' if page else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
            f"<style>{css}text{{font-family:'JetBrains Mono',ui-monospace,monospace}}</style>"
            f"{background}\n{body}\n</svg>\n")


def picture(name: str, alt: str, extra: str = "") -> str:
    """One image that follows GitHub's light/dark theme."""
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="assets/dark/{name}.svg"/>'
            f'<img src="assets/light/{name}.svg" alt="{escape(alt)}"{extra}/></picture>')


def write_readme() -> None:
    buttons = "\n".join(f'<a href="{url}">{picture("link-" + label, label, " height=\"40\"")}</a>'
                        for label, url in LINKS)
    visitors = ('<img src="https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FPDelos%2FPDelos'
                '&label=VISITORS&labelColor=%230b0f0c&countColor=%231a7f37&style=flat-square" alt="visitors"/>')
    (ROOT / "README.md").write_text(f"""<!-- Generated by profile/build.py: edit the content there and rerun it. -->

<div align="center">

{picture("about", "01 · about: Pol de los Santos, software, data and ML engineer in Barcelona")}

{buttons}

{picture("education", "02 · education: data science & engineering and biomedical engineering at UPC (CFIS); "
                      "Spanish Scientific and International Baccalaureate")}

{picture("experience", "03 · experience: Google (Munich, Zurich), Esperanto Technologies, Science4Tech")}

{picture("stack", "04 · stack & interests")}

{visitors}

</div>
""")


def main() -> None:
    panels = {}  # (mode, name) -> (width, height, body, has_page_background)
    for mode, colors in THEMES.items():
        for name, make in (("about", about), ("education", education), ("experience", experience), ("stack", stack)):
            height, body = make(colors).render()
            panels[(mode, name)] = (WIDTH, height, body, True)
        for label, _ in LINKS:
            width, height, body = button(colors, label)
            panels[(mode, f"link-{label}")] = (width, height, body, False)

    css_by_chars: dict[str, str] = {}
    for (mode, name), (width, height, body, has_page) in panels.items():
        chars = "".join(sorted(set(re.sub(r"<[^>]+>", "", body))))   # each file embeds only the glyphs it uses
        if chars not in css_by_chars:
            css_by_chars[chars] = font_css(chars)
        out = ASSETS / mode / f"{name}.svg"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(svg(width, height, body, THEMES[mode]["page"] if has_page else None, css_by_chars[chars]))
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KB")
    write_readme()


if __name__ == "__main__":
    main()
