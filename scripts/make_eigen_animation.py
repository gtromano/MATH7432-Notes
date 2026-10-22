# Eigenvectors as "directions that only stretch", for the week 1 covariance matrix
# Sigma = [[4, 2], [2, 3]] (after the Wikipedia animation "Eigenvectors of a linear operator").
#
#   phase 1: a grid of points moves from x to Sigma x (through (1 - t) I + t Sigma);
#            points on the two eigen-directions (red) only slide along them; each eigenvector
#            arrow keeps its unit part (blue) and grows a green part: the stretch, up to lambda.
#   phase 2: the picture rotates until the eigen-directions are the axes: in the eigenbasis,
#            Sigma is nothing but a stretch by lambda_1 along one axis and lambda_2 along the other.
#   then everything plays backwards, and loops.
#
# Writes the GIF (HTML outputs) and a static three-panel PNG (PDF outputs) to the notes'
# images/chapter1/, relative to this script, so the notes folder can be called anything.
# 01-week1.qmd runs it during the render whenever either file is missing; to remake them by
# hand, run from the notes folder:
#     python3 scripts/make_eigen_animation.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

SIGMA = np.array([[4.0, 2.0], [2.0, 3.0]])
evals, evecs = np.linalg.eigh(SIGMA)          # ascending
order = np.argsort(evals)[::-1]               # largest first, as eigen() prints
evals, evecs = evals[order], evecs[:, order]
evecs[:, 0] *= np.sign(evecs[0, 0])           # point both into a readable quadrant
evecs[:, 1] *= np.sign(evecs[1, 1])
theta = np.arctan2(evecs[1, 0], evecs[0, 0])  # angle of the first eigen-direction

g = np.linspace(-1, 1, 11)
GRID = np.array([(a, b) for a in g for b in g]).T   # 2 x 121 points
W = np.array([1.0, 0.0])                            # a vector that is NOT an eigenvector

BLUE, STRETCH, RED, GREY = "#0072B2", "#009E73", "#CC0000", "0.45"
LIM = 7


def rot(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def arrow(ax, start, end, colour, width=0.06, z=3):
    d = end - start
    if np.hypot(*d) < 1e-9:
        return
    ax.arrow(*start, *d, width=width, head_width=0.32, head_length=0.38,
             length_includes_head=True, color=colour, zorder=z)


def draw(ax, t, phi, title=None):
    """t in [0, 1]: how far along the map x -> Sigma x; phi: rotation of the whole picture."""
    ax.cla()
    M = (1 - t) * np.eye(2) + t * SIGMA
    R = rot(phi)
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)

    # eigen-directions: the points on them only slide along them
    for j in range(2):
        v = R @ evecs[:, j]
        ax.plot([-LIM * 2 * v[0], LIM * 2 * v[0]], [-LIM * 2 * v[1], LIM * 2 * v[1]],
                color=RED, lw=1.3, ls="--", alpha=0.6, zorder=1)

    pts = R @ M @ GRID
    ax.scatter(pts[0], pts[1], s=7, color="black", zorder=2)

    # a generic vector: changes direction
    arrow(ax, np.zeros(2), R @ M @ W, GREY, width=0.045)

    # eigenvectors: unit part (blue) + stretch part (orange), lambda as the stretch factor
    for j, sub in enumerate(["₁", "₂"]):
        v = evecs[:, j]
        s = 1 + t * (evals[j] - 1)                   # current stretch factor along v
        arrow(ax, np.zeros(2), R @ v, BLUE)
        arrow(ax, R @ v, R @ (s * v), STRETCH)
        tip = R @ (s * v)
        off = R @ (0.9 * np.array([-v[1], v[0]]))      # beside the tip, not beyond it
        label = f"λ{sub} = {evals[j]:.2f}" if t > 0.999 else f"×{s:.2f}"
        ax.text(*(tip + off), label, color=STRETCH, fontsize=12, fontweight="bold",
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("0.8")
    if title:
        ax.set_title(title, fontsize=11)


def smooth(u):
    return 0.5 - 0.5 * np.cos(np.pi * u)   # ease in and out


# ---- the animation -------------------------------------------------------------------------
N_MORPH, N_HOLD, N_ROT, N_END = 60, 18, 45, 30
frames = (
    [(smooth(k / N_MORPH), 0.0, "x  →  Σx: the red directions only stretch") for k in range(N_MORPH + 1)]
    + [(1.0, 0.0, "x  →  Σx: the red directions only stretch")] * N_HOLD
    + [(1.0, -theta * smooth(k / N_ROT), "rotate: eigenvectors as the basis") for k in range(N_ROT + 1)]
    + [(1.0, -theta, "in the eigenbasis: Σ = diag(λ₁, λ₂), a pure stretch")] * N_END
)
frames = frames + frames[::-1][1:-1]        # play it back, then loop

fig, ax = plt.subplots(figsize=(5.4, 5.6))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)


def update(k):
    t, phi, title = frames[k]
    draw(ax, t, phi, title)


here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "..", "images", "chapter1")
os.makedirs(out_dir, exist_ok=True)
out_gif = os.path.join(out_dir, "w1_eigen.gif")
anim = FuncAnimation(fig, update, frames=len(frames))
anim.save(out_gif, writer=PillowWriter(fps=20), dpi=80)
plt.close(fig)

# ---- the static version (PDF outputs) ------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88, wspace=0.05)
draw(axes[0], 0.0, 0.0, "before: x")
draw(axes[1], 1.0, 0.0, "after: Σx (red directions only stretch)")
draw(axes[2], 1.0, -theta, "rotated: Σ = diag(λ₁, λ₂) in the eigenbasis")
out_png = os.path.join(out_dir, "w1_eigen_static.png")
fig.savefig(out_png, dpi=150)
plt.close(fig)

print("wrote", os.path.normpath(out_gif), "and", os.path.normpath(out_png))
