"""Intro effect figure: iid vs zero-sum ensemble-mean trajectory on a 2D quadratic."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"iid": "#1f77b4", "zero_sum": "#2ca02c"}


def run_ensemble(rng, method, h, eta, k, steps, x0):
    d = x0.shape[0]
    x = np.tile(x0, (k, 1))
    path = np.empty((steps + 1, d))
    path[0] = x.mean(axis=0)
    for t in range(steps):
        noise = rng.standard_normal((k, d))
        if method == "zero_sum":
            noise = (noise - noise.mean(axis=0, keepdims=True)) * np.sqrt(k / (k - 1))
        x = x - eta * (x * h) + np.sqrt(2 * eta) * noise
        path[t + 1] = x.mean(axis=0)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("paper/figures/intro_effect.png"))
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    h = np.array([1.0, 3.0])
    eta, k, steps = 0.15, 8, 120
    x0 = np.array([2.5, -1.5])
    probe = np.array([1.0, 1.0]) / np.sqrt(2.0)

    rng = np.random.default_rng(args.seed)
    iid = run_ensemble(rng, "iid", h, eta, k, steps, x0) @ probe
    zs = run_ensemble(rng, "zero_sum", h, eta, k, steps, x0) @ probe
    gd = np.empty(steps + 1)
    y = x0.copy()
    gd[0] = y @ probe
    for t in range(steps):
        y = y - eta * (y * h)
        gd[t + 1] = y @ probe

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ts = np.arange(steps + 1)
    ax.plot(ts, iid, color=COLORS["iid"], linewidth=1.4, label=f"iid noise, $K={k}$")
    ax.plot(ts, zs, color=COLORS["zero_sum"], linewidth=2.4, label=f"zero-sum noise, $K={k}$")
    ax.plot(ts, gd, color="black", linestyle="--", linewidth=1.0, label="noiseless gradient descent")
    ax.text(43, 0.92, "iid: the average still jitters", fontsize=9, color=COLORS["iid"])
    ax.annotate("zero-sum: the average is exactly\nthe noiseless path",
                xy=(56, 0.0), xytext=(60, -0.72), fontsize=9, color=COLORS["zero_sum"],
                arrowprops={"arrowstyle": "-", "color": COLORS["zero_sum"], "linewidth": 0.8})
    ax.set_xlabel("ULA step $t$", fontsize=10)
    ax.set_ylabel(r"ensemble mean  $a^\top \bar X_t$", fontsize=10)
    ax.set_ylim(-0.88, 1.72)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    max_dev = float(np.abs(zs - gd).max())
    print(f"wrote {args.out}; max |zero_sum - gd| = {max_dev:.3e}")


if __name__ == "__main__":
    main()
