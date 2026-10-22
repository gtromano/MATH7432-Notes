# Preconditioning as a change of coordinates, for the week 5 valley
# f(x) = x1^2 / 2 + 10 x2^2 / 2, start (-9, 1.5), P = diag(1, 1/10).
#
#   phase 1: in the original coordinates x, the direction -grad f (grey) is perpendicular to the
#            elliptical contours, so it points across the valley rather than at the minimum;
#            plain GD (gamma = 0.18, blue) zig-zags.
#   phase 2: stretch the x2 axis by sqrt(10), i.e. move to z = P^(-1/2) x = (x1, sqrt(10) x2).
#            The contours become circles and -grad f, recomputed in the current coordinates,
#            turns until it points straight at the minimum.
#   phase 3: in z the bowl is round, so one plain GD step with gamma = 1 lands on the minimum (green).
#   phase 4: squeeze back to x: the green step is still a straight line to the minimum, and it is
#            exactly the preconditioned step -P grad f.
#   then it loops.
#
# Writes the GIF (HTML outputs) and a static three-panel PNG (PDF outputs) to the notes'
# images/chapter5/, relative to this script, so the notes folder can be called anything.
# 05-week5.qmd runs it during the render whenever either file is missing; to remake them by
# hand, run from the notes folder:
#     python3 scripts/make_precond_animation.py

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

CURV = np.array([1.0, 10.0])                  # Hessian diag(1, 10)
X0 = np.array([-9.0, 1.5])
S_MAX = np.sqrt(CURV[1])                      # the x2 stretch that makes the bowl round
GAMMA_GD = 0.18

BLUE, GREEN, GREY = "#0072B2", "#009E73", "0.35"
XLIM, YLIM = (-11, 4), (-6, 6)
RADII = np.arange(1, 13)                      # contour levels f = r^2 / 2, as circles in z


def gd_path(n):
    path = [X0]
    for _ in range(n):
        path.append(path[-1] - GAMMA_GD * CURV * path[-1])
    return np.array(path)


ZIG = gd_path(8)


def to_view(x, s):
    """original coordinates x (2 x m) -> the picture, with the x2 axis stretched by s"""
    return np.vstack([x[0], s * x[1]])


def arrow(ax, start, end, colour, width=0.07, z=4):
    d = end - start
    if np.hypot(*d) < 1e-9:
        return
    ax.arrow(*start, *d, width=width, head_width=0.38, head_length=0.45,
             length_includes_head=True, color=colour, zorder=z)


def draw(ax, s, n_zig=0, show_dir=True, green=0.0, title=None, sub=None):
    """s: stretch of the x2 axis (1 = original x, sqrt(10) = round coordinates z);
    n_zig: GD steps drawn; green in [0, 1]: how much of the preconditioned step is drawn."""
    ax.cla()
    th = np.linspace(0, 2 * np.pi, 200)
    for r in RADII:
        circ = np.vstack([r * np.cos(th), r * np.sin(th) / S_MAX])  # level set, in x
        v = to_view(circ, s)
        ax.plot(v[0], v[1], color="0.8", lw=0.7, zorder=1)
    ax.scatter([0], [0], marker="x", s=60, color="black", zorder=5)

    # plain GD, in the original coordinates (only meaningful at s = 1, drawn wherever asked)
    if n_zig > 0:
        v = to_view(ZIG[: n_zig + 1].T, s)
        ax.plot(v[0], v[1], color=BLUE, lw=1.2, zorder=3)
        ax.scatter(v[0], v[1], color=BLUE, s=14, zorder=3)

    p0 = to_view(X0[:, None], s)[:, 0]
    # steepest descent in the *current* coordinates u = (x1, s x2):
    # f = u1^2 / 2 + (10 / s^2) u2^2 / 2, so -grad_u f = -(u1, 10 u2 / s^2)
    if show_dir:
        d = -np.array([p0[0], CURV[1] * p0[1] / s**2])
        d /= np.hypot(*d)
        ax.plot([p0[0], p0[0] + 30 * d[0]], [p0[1], p0[1] + 30 * d[1]],
                color=GREY, lw=1, ls="--", zorder=2)
        if n_zig == 0 and green == 0:          # otherwise a coloured step covers it
            arrow(ax, p0, p0 + 3.2 * d, GREY, width=0.05)

    # the preconditioned step: straight from x0 to the minimum, in any coordinates
    if green > 0:
        arrow(ax, p0, (1 - green) * p0, GREEN, width=0.09, z=6)
    ax.scatter(*p0, color="black", s=28, zorder=7)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("0.8")
    if title:
        ax.set_title(title, fontsize=11)
    if sub:
        ax.text(0.02, 0.03, sub, transform=ax.transAxes, fontsize=10, color="0.25",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def smooth(u):
    return 0.5 - 0.5 * np.cos(np.pi * u)   # ease in and out


# ---- the animation -------------------------------------------------------------------------
# each frame: (s, n_zig, show_dir, green, title, subtitle)
SUB_X = "coordinates x = (x₁, x₂)"
SUB_Z = "coordinates z = (x₁, √10·x₂)"
T1 = "(a) GD: −∇f points across the valley"
T2 = "stretch x₂ by √10: the valley becomes round"
T3 = "(b) round bowl: −∇f points at the minimum"
T4 = "back in x: the same step is −P∇f"
N_ZIG, N_HOLD, N_MORPH, N_STEP, N_END = 8, 25, 50, 25, 45
frames = (
    [(1.0, 0, True, 0.0, T1, SUB_X)] * N_HOLD
    + [(1.0, k, True, 0.0, T1, SUB_X) for k in range(1, N_ZIG + 1) for _ in range(5)]
    + [(1.0, N_ZIG, True, 0.0, T1, SUB_X)] * N_HOLD
    + [(1 + (S_MAX - 1) * smooth(k / N_MORPH), 0, True, 0.0, T2, None) for k in range(N_MORPH + 1)]
    + [(S_MAX, 0, True, 0.0, T3, SUB_Z)] * N_HOLD
    + [(S_MAX, 0, True, smooth(k / N_STEP), T3 + "; γ = 1 lands on it", SUB_Z) for k in range(1, N_STEP + 1)]
    + [(S_MAX, 0, True, 1.0, T3 + "; γ = 1 lands on it", SUB_Z)] * N_HOLD
    + [(S_MAX + (1 - S_MAX) * smooth(k / N_MORPH), 0, False, 1.0, T4, None) for k in range(N_MORPH + 1)]
    + [(1.0, 1, False, 1.0, T4, SUB_X + ";  blue: GD step,  green: −P∇f")] * N_END
)

fig, ax = plt.subplots(figsize=(6.4, 5.6))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.93)


def update(k):
    s, n_zig, show_dir, green, title, sub = frames[k]
    draw(ax, s, n_zig, show_dir, green, title, sub)


here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "..", "images", "chapter5")
os.makedirs(out_dir, exist_ok=True)
out_gif = os.path.join(out_dir, "w5_precond.gif")
anim = FuncAnimation(fig, update, frames=len(frames))
anim.save(out_gif, writer=PillowWriter(fps=20), dpi=80)
plt.close(fig)

# ---- the static version (PDF outputs) ------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.05)
draw(axes[0], 1.0, N_ZIG, True, 0.0, "(a) in x: −∇f points across the valley", SUB_X)
draw(axes[1], S_MAX, 0, True, 1.0, "(b) in z: round bowl, one step with γ = 1", SUB_Z)
draw(axes[2], 1.0, 1, False, 1.0, "back in x: the green step is −P∇f",
     "blue: GD step,  green: −P∇f")
out_png = os.path.join(out_dir, "w5_precond_static.png")
fig.savefig(out_png, dpi=150)
plt.close(fig)

print("wrote", os.path.normpath(out_gif), "and", os.path.normpath(out_png))
