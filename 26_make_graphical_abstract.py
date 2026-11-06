"""Graphical abstract: thermoelectric transport-mechanism classification and cross-database transfer."""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

FIG_DIR = Path(os.environ.get("FIG_DIR", "figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8.0, "axes.grid": False, "font.family": "DejaVu Sans"})

W, H = 5.12, 6.9
fig, ax = plt.subplots(figsize=(W, H))
fig.subplots_adjust(0, 0, 1, 1)
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

border = Rectangle((0.04, 0.04), W - 0.08, H - 0.08, linewidth=1.5,
                    edgecolor="#333333", facecolor="none", zorder=10)
ax.add_patch(border)


def box(cx, cy, w, h, text, fc="#eef3fb", ec="#2b5a8c", fontsize=8, fontweight="normal"):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight)


def arrow(x1, y1, x2, y2, color="#555555"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=color)
    ax.add_patch(a)


ax.text(W / 2, 6.5, "Composition-Only ML for Thermoelectric\nTransport Mechanism",
        ha="center", va="center", fontsize=10, fontweight="bold")

box(1.35, 5.45, 2.0, 0.95, "JARVIS-DFT\n23,218 records\n(BoltzTraP)", fc="#eef3fb")
box(3.77, 5.45, 2.0, 0.95, "ESTM\n5,205 measurements\n880 compounds", fc="#eef3fb")
box(2.56, 3.95, 3.4, 1.15, "Derive band-like / hopping-like label\nfrom $\\sigma(T)$ shape\n(Arrhenius + Spearman)",
    fc="#fdf1e3", ec="#c77d2e")
arrow(1.6, 4.95, 2.1, 4.55)
arrow(3.5, 4.95, 3.0, 4.55)

box(1.35, 2.35, 2.2, 1.45, "Composition-only\nmechanism classifier\nAUC = 0.841\n(100% hopping-side\nliterature agreement)",
    fc="#e9f5ec", ec="#3f7a4a")
box(3.77, 2.35, 2.2, 1.45, "Cross-database\nproperty transfer\nSeebeck: partial\nsuccess; Conductivity:\npoor (target-definition\nmismatch)",
    fc="#f6e9ec", ec="#b23b3b")
arrow(2.0, 3.37, 1.55, 3.1)
arrow(3.1, 3.37, 3.55, 3.1)

box(2.56, 0.95, 4.2, 0.85, "Screening demo: ranked candidates\nbeyond the labeled set (one literature-supported)", fc="#f1ecf5", ec="#6a4c93")
arrow(1.35, 1.62, 1.9, 1.38)
arrow(3.77, 1.62, 3.2, 1.38)

ax.text(W / 2, 0.33,
        "A ~14-order-of-magnitude JARVIS unit inconsistency is found and\nreported with a Wiedemann-Franz check.",
        ha="center", va="center", fontsize=8, style="italic", color="#444444")

from matplotlib.text import Text
mn = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip())
assert mn >= 8, mn
print("graphical abstract min font", mn)
plt.savefig(FIG_DIR / "graphical_abstract.pdf")
plt.savefig(FIG_DIR / "graphical_abstract.png", dpi=400)
print("saved graphical abstract")
