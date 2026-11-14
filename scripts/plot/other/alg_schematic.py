"""Schematic of one round of nvMolKit's batched BitBIRCH (plots/other/.../alg_schematic.*).

Toy tree: branching factor 4, a root over leaves A and B, and a batch of rows 1-6 (numbered in input order).
Panels: 1 route against the snapshot, 2 per-leaf work (a group, then the rest in order, until the leaf
overflows), 3 summaries refreshed upwards, 4 the overflowing leaf split around its least-similar pair.
No data; drawn from the algorithm in nvMolKit src/bitbirch.cu (main) and bblean csrc/bitbirch_batched.hpp.

  python scripts/plot/other/alg_schematic.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import COLW, save, style  # noqa: E402


GREEN = "#76B900"  # nvMolKit
GREEN_FILL = "#dcefb8"
GREY = "#8a8a8a"
GREY_FILL = "#f0f0f0"
INK = "#1f3a68"

CW, CH = 0.72, 0.62  # cell width, height
BF = 4  # branching factor of the toy tree


def cell(ax, x, y, text, new=False, seed=False):
    ax.add_patch(Rectangle((x, y), CW, CH, facecolor=GREEN_FILL if new else GREY_FILL,
                           edgecolor=GREEN if new else GREY, linewidth=1.6 if seed else 0.6, zorder=2))
    ax.text(x + CW / 2, y + CH / 2, text, ha="center", va="center", fontsize=5.6, color=INK, zorder=3)


def node(ax, x, y, labels, new=(), seeds=()):
    """A node: a box with BF slots; cells past BF stick out of it (overflow). Returns cell centres."""
    pad = 0.08
    ax.add_patch(FancyBboxPatch((x - pad, y - pad), BF * CW + 2 * pad, CH + 2 * pad,
                                boxstyle="round,pad=0,rounding_size=0.12", facecolor="white",
                                edgecolor=GREY, linewidth=0.6, zorder=1))
    centres = []
    for i, lab in enumerate(labels):
        cell(ax, x + i * CW, y, lab, new=lab in new, seed=lab in seeds)
        centres.append((x + i * CW + CW / 2, y + CH / 2))
    return centres


def row(ax, x, y, k, waiting=False):
    ax.add_patch(Circle((x, y), 0.27, facecolor="white", edgecolor=GREY if waiting else INK,
                        linewidth=0.7, linestyle=(0, (2, 1.5)) if waiting else "-", zorder=3))
    ax.text(x, y, str(k), ha="center", va="center", fontsize=5.6, color=GREY if waiting else INK, zorder=4)


def link(ax, top, bottom_x, bottom_y, color=GREY, arrow=False, lw=0.6):
    ax.add_patch(FancyArrowPatch(top, (bottom_x, bottom_y), arrowstyle="-|>" if arrow else "-",
                                 mutation_scale=5, color=color, linewidth=lw, zorder=1,
                                 shrinkA=0, shrinkB=0))


def tree(ax, root_labels, leaves, root_new=(), rx=3.9, up=False):
    """leaves: list of (x, labels, new, seeds). Root entry i points at leaf i."""
    ry, ly = 4.3, 2.2
    names = [lab[1:] for lab in root_labels]
    rc = node(ax, rx - BF * CW / 2, ry, root_labels, new=root_new)
    out = []
    for (x, labels, new, seeds), (cx, cy) in zip(leaves, rc):
        lc = node(ax, x, ly, labels, new=new, seeds=seeds)
        if up:
            link(ax, (x + BF * CW / 2, ly + CH + 0.08), cx, ry - 0.1, color=GREEN, arrow=True, lw=0.8)
        else:
            link(ax, (cx, ry - 0.08), x + BF * CW / 2, ly + CH + 0.08)
        # leaf name above its top-left corner, clear of overflow cells and of the neighbouring leaf
        ax.text(x - 0.08, ly + CH + 0.16, names.pop(0), ha="left", va="bottom", fontsize=5.6, color=GREY)
        out.append(lc)
    return out


def panel(ax, title):
    ax.set_xlim(-0.75, 9.9)
    ax.set_ylim(0.4, 5.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=7, loc="left", pad=1)


def main():
    style()
    fig, axes = plt.subplots(2, 2, figsize=(COLW, 2.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, wspace=0.06, hspace=0.2)
    ax1, ax2, ax3, ax4 = axes.ravel()
    xa, xb = 0.0, 4.6
    full_a = ["a1", "a2", "a3", "a4", "a5"]
    rows_a = [xa + 0.35 + i * 0.75 for i in range(4)]  # rows 1, 3, 4, 6
    rows_b = [xb + 0.35 + i * 0.75 for i in range(2)]  # rows 2, 5
    ry = 0.85

    # 1: every row walks down the snapshot to a leaf, in parallel
    panel(ax1, "1  route against snapshot")
    tree(ax1, ["rA", "rB"], [(xa, ["a1", "a2", "a3"], (), ()), (xb, ["b1", "b2"], (), ())])
    for x, k in zip(rows_a, [1, 3, 4, 6]):
        row(ax1, x, ry, k)
    for x, k in zip(rows_b, [2, 5]):
        row(ax1, x, ry, k)

    # 2: B: rows 2 and 5 both picked b1 and join it as a group; A: rows in order until the leaf overflows
    panel(ax2, "2  per leaf: groups, then in order")
    la, lb = tree(ax2, ["rA", "rB"], [(xa, full_a, ("a2", "a4", "a5"), ()), (xb, ["b1", "b2"], ("b1",), ())])
    for x, k, target in zip(rows_a, [1, 3, 4], [la[1], la[3], la[4]]):
        row(ax2, x, ry, k)
        link(ax2, (x, ry + 0.27), target[0], target[1] - CH / 2, color=GREEN, arrow=True, lw=0.7)
    row(ax2, rows_a[3], ry, 6, waiting=True)
    for x, k in zip(rows_b, [2, 5]):
        row(ax2, x, ry, k)
        link(ax2, (x, ry + 0.27), lb[0][0], lb[0][1] - CH / 2, color=GREEN, arrow=True, lw=0.7)

    # 3: the accepted rows are added to the summaries above them
    panel(ax3, "3  refresh summaries")
    tree(ax3, ["rA", "rB"], [(xa, full_a, (), ()), (xb, ["b1", "b2"], (), ())], root_new=("rA", "rB"), up=True)
    row(ax3, rows_a[3], ry, 6, waiting=True)

    # 4: A splits around its least-similar pair (a1, a5); its new sibling joins the root at the end
    panel(ax4, "4  split; waiting rows next round")
    tree(ax4, ["rA", "rB", "rA\u2032"],
         [(0.0, ["a1", "a2", "a3"], (), ("a1",)), (3.45, ["b1", "b2"], (), ()),
          (6.9, ["a4", "a5"], ("a4", "a5"), ("a5",))],
         root_new=("rA", "rA\u2032"), rx=4.9)
    row(ax4, 6.9 + 0.35, ry, 6, waiting=True)

    save(fig, "alg_schematic")


if __name__ == "__main__":
    main()
