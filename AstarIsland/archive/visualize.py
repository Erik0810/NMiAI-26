"""
Visualization utilities for Astar Island maps and predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from config import CLASS_NAMES, TERRAIN_TO_CLASS


# Color scheme for terrain classes
CLASS_COLORS = {
    0: "#4A90D9",  # Empty (ocean/plains) — blue
    1: "#E8A83E",  # Settlement — gold
    2: "#8B4513",  # Port — brown
    3: "#808080",  # Ruin — gray
    4: "#228B22",  # Forest — green
    5: "#696969",  # Mountain — dark gray
}

# More detailed colors for raw terrain
RAW_COLORS = {
    0:  "#D2B48C",  # Empty — tan
    1:  "#E8A83E",  # Settlement — gold
    2:  "#8B4513",  # Port — brown
    3:  "#808080",  # Ruin — gray
    4:  "#228B22",  # Forest — green
    5:  "#696969",  # Mountain — dark gray
    10: "#4A90D9",  # Ocean — blue
    11: "#90EE90",  # Plains — light green
}


def plot_initial_states(detail: dict, save_path: str = None):
    """Plot all seed initial states side by side."""
    seeds = detail["seeds_count"]
    fig, axes = plt.subplots(1, seeds, figsize=(4 * seeds, 4))
    if seeds == 1:
        axes = [axes]

    for i, state in enumerate(detail["initial_states"]):
        grid = np.array(state["grid"])
        H, W = grid.shape

        # Create RGB image
        img = np.zeros((H, W, 3))
        for code, color in RAW_COLORS.items():
            rgb = mcolors.to_rgb(color)
            mask = grid == code
            img[mask] = rgb

        axes[i].imshow(img, interpolation="nearest")
        axes[i].set_title(f"Seed {i}")

        # Mark settlements
        for s in state["settlements"]:
            if s["alive"]:
                marker = "^" if s["has_port"] else "o"
                color = "red" if s["has_port"] else "white"
                axes[i].plot(s["x"], s["y"], marker, color=color,
                             markersize=6, markeredgecolor="black", markeredgewidth=0.5)

        axes[i].set_xlim(-0.5, W - 0.5)
        axes[i].set_ylim(H - 0.5, -0.5)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_prediction(prediction: np.ndarray, title: str = "Prediction",
                    seed_info: dict = None, save_path: str = None):
    """
    Plot a prediction tensor as argmax class map + confidence heatmap.
    """
    H, W, C = prediction.shape
    argmax = np.argmax(prediction, axis=-1)
    confidence = np.max(prediction, axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Argmax class map
    img = np.zeros((H, W, 3))
    for cls, color in CLASS_COLORS.items():
        rgb = mcolors.to_rgb(color)
        mask = argmax == cls
        img[mask] = rgb
    ax1.imshow(img, interpolation="nearest")
    ax1.set_title(f"{title} — Predicted Class")

    # Mark settlements from initial state
    if seed_info:
        for s in seed_info["settlements"]:
            if s["alive"]:
                marker = "^" if s["has_port"] else "o"
                ax1.plot(s["x"], s["y"], marker, color="white",
                         markersize=5, markeredgecolor="black", markeredgewidth=0.5)

    # Confidence heatmap
    im = ax2.imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_title(f"{title} — Confidence")
    plt.colorbar(im, ax=ax2, shrink=0.8)

    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_observation_coverage(observations: list, W: int, H: int,
                              seeds_count: int, save_path: str = None):
    """Plot heatmap of how many times each cell was observed."""
    fig, axes = plt.subplots(1, seeds_count, figsize=(4 * seeds_count, 4))
    if seeds_count == 1:
        axes = [axes]

    for seed_idx in range(seeds_count):
        coverage = np.zeros((H, W))
        seed_obs = [o for o in observations if o["seed_index"] == seed_idx]

        for obs in seed_obs:
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            coverage[vy:vy + vh, vx:vx + vw] += 1

        im = axes[seed_idx].imshow(coverage, cmap="YlOrRd", vmin=0,
                                    interpolation="nearest")
        axes[seed_idx].set_title(f"Seed {seed_idx} ({len(seed_obs)} queries)")
        axes[seed_idx].set_xticks([])
        axes[seed_idx].set_yticks([])
        plt.colorbar(im, ax=axes[seed_idx], shrink=0.8)

    plt.suptitle("Observation Coverage")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_comparison(prediction: np.ndarray, ground_truth: np.ndarray,
                    title: str = "", save_path: str = None):
    """Plot prediction vs ground truth side by side (post-round analysis)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    pred_argmax = np.argmax(prediction, axis=-1)
    gt_argmax = np.argmax(ground_truth, axis=-1)
    pred_conf = np.max(prediction, axis=-1)
    gt_entropy = -np.sum(ground_truth * np.log(ground_truth + 1e-10), axis=-1)

    for ax, data, cmap, t in [
        (axes[0, 0], pred_argmax, None, "Prediction (argmax)"),
        (axes[0, 1], gt_argmax, None, "Ground Truth (argmax)"),
        (axes[1, 0], pred_conf, "RdYlGn", "Prediction Confidence"),
        (axes[1, 1], gt_entropy, "hot", "Ground Truth Entropy"),
    ]:
        if cmap is None:
            img = np.zeros((*data.shape, 3))
            for cls, color in CLASS_COLORS.items():
                img[data == cls] = mcolors.to_rgb(color)
            ax.imshow(img, interpolation="nearest")
        else:
            im = ax.imshow(data, cmap=cmap, interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
