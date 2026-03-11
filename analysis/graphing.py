import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.io_utils import load_benchmark_results
import os
import json


def load_latest_results(folder="results"):
    files = [
        f for f in os.listdir(folder)
        if f.startswith("benchmark_") and f.endswith(".json")
    ]

    if not files:
        raise FileNotFoundError("No benchmark files found in results/")

    files.sort()  # timestamps sort naturally
    latest = files[-1]

    filepath = os.path.join(folder, latest)

    with open(filepath, "r") as f:
        data = json.load(f)

    print(f"\nLoaded results from: {filepath}")

    return data


def json_to_rows(data):
    """
    Converts new JSON structure into flat rows
    compatible with existing plotting functions.
    """

    rows = []

    for entry in data["results"]:
        # Support both "variant" (new) and "config" (legacy) for backward compatibility
        variant = entry.get("variant", entry.get("config"))
        summary = entry["summary"]

        rows.append({
            "controller": variant["controller_type"],
            "estimator": variant["estimator_type"] if variant["estimator_type"] is not None else "none",
            "noise": variant["noise_std"],
            "delay": variant["delay_steps"],
            "stability": summary["stability_rate"],
            "settling": summary["avg_settling_time"],
            "avg_acc": summary["avg_acc"],
            "max_acc": summary["max_acc"]
        })

    return rows


# ============================================================
# 1) Robustness Curves: Stability vs Noise
# ============================================================

def plot_stability_vs_noise(rows):
    """
    For each controller, create 4 subplots (one per estimator).
    Lines show delay levels.
    """

    controllers = sorted(set(r["controller"] for r in rows))
    estimators = sorted(set(r["estimator"] for r in rows))
    delays = sorted(set(r["delay"] for r in rows))

    for controller in controllers:

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        fig.suptitle(f"Stability vs Noise — Controller: {controller}", fontsize=14)

        for idx, estimator in enumerate(estimators):
            ax = axes[idx]

            subset = [
                r for r in rows
                if r["controller"] == controller
                and r["estimator"] == estimator
            ]

            noises = sorted(set(r["noise"] for r in subset))

            for delay in delays:
                y = []
                for noise in noises:
                    val = next(
                        r["stability"]
                        for r in subset
                        if r["noise"] == noise and r["delay"] == delay
                    )
                    y.append(val)

                ax.plot(noises, y, marker="o", label=f"Delay={delay}")

            ax.set_title(f"Estimator: {estimator}")
            ax.set_xlabel("Noise Std")
            ax.set_ylabel("Stability Rate")
            ax.set_ylim(0, 1.05)
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()


# ============================================================
# 2) Stability Heatmaps (Noise x Delay)
# ============================================================

def plot_stability_heatmaps(rows):
    """
    Heatmap per (controller, estimator).
    """

    controllers = sorted(set(r["controller"] for r in rows))
    estimators = sorted(set(r["estimator"] for r in rows))

    for controller in controllers:
        for estimator in estimators:

            subset = [
                r for r in rows
                if r["controller"] == controller
                and r["estimator"] == estimator
            ]

            noises = sorted(set(r["noise"] for r in subset))
            delays = sorted(set(r["delay"] for r in subset))

            Z = np.zeros((len(delays), len(noises)))

            for r in subset:
                i = delays.index(r["delay"])
                j = noises.index(r["noise"])
                Z[i, j] = r["stability"]

            plt.figure(figsize=(6, 4))
            plt.imshow(Z, aspect="auto", origin="lower")
            plt.xticks(range(len(noises)), noises)
            plt.yticks(range(len(delays)), delays)
            plt.xlabel("Noise Std")
            plt.ylabel("Delay Steps")
            plt.title(f"Stability Heatmap — {controller} + {estimator}")
            plt.colorbar(label="Stability Rate")
            plt.tight_layout()
            plt.show()


# ============================================================
# 3) Effort vs Stability Tradeoff
# ============================================================

def plot_effort_vs_stability(rows):
    """
    Scatter plot showing control effort vs stability.
    """

    controllers = sorted(set(r["controller"] for r in rows))
    markers = ["o", "s", "^", "D"]
    estimator_list = sorted(set(r["estimator"] for r in rows))

    plt.figure(figsize=(8, 6))

    for i, estimator in enumerate(estimator_list):
        subset = [r for r in rows if r["estimator"] == estimator]

        x = [r["avg_acc"] for r in subset]
        y = [r["stability"] for r in subset]

        plt.scatter(
            x,
            y,
            marker=markers[i % len(markers)],
            label=f"{estimator}",
            alpha=0.7
        )

    plt.xlabel("Average Acceleration (m/s²)")
    plt.ylabel("Stability Rate")
    plt.title("Control Effort vs Stability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 4) Settling Time Comparison (Only Stable Cases)
# ============================================================

def plot_settling_time(rows, stability_threshold=0.95):
    """
    Bar chart of settling times for stable configurations.
    """

    stable_rows = [
        r for r in rows
        if r["stability"] >= stability_threshold
        and r["settling"] is not None
    ]

    labels = []
    settling_times = []

    for r in stable_rows:
        label = f"{r['controller']}-{r['estimator']}\nN={r['noise']},D={r['delay']}"
        labels.append(label)
        settling_times.append(r["settling"])

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(settling_times)), settling_times)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("Settling Time (s)")
    plt.title("Settling Time (Stable Configurations Only)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5) Acceleration Explosion Detection
# ============================================================

def plot_max_acceleration(rows):
    """
    Visualizes worst-case acceleration (log scale useful).
    """

    labels = []
    max_accs = []

    for r in rows:
        label = f"{r['controller']}-{r['estimator']}\nN={r['noise']},D={r['delay']}"
        labels.append(label)
        max_accs.append(r["max_acc"])

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(max_accs)), max_accs)
    plt.yscale("log")
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.ylabel("Max Acceleration (log scale)")
    plt.title("Maximum Acceleration Across Configurations")
    plt.tight_layout()
    plt.show()


# ============================================================
# Master Function
# ============================================================

def run_full_analysis():
    """
    Loads latest benchmark and runs full analysis.
    """

    data = load_latest_results()
    rows = json_to_rows(data)

    plot_stability_vs_noise(rows)
    plot_stability_heatmaps(rows)
    plot_effort_vs_stability(rows)
    plot_settling_time(rows)
    plot_max_acceleration(rows)
    
    


def plot_state_history(state_history, x_ref, dt=0.001):
    """
    Plot all 8 state variables over time with reference lines.

    state_history : (N,8) numpy array
    x_ref         : SystemState
    dt            : timestep (seconds)
    """

    x_ref_vec = x_ref.as_vector()
    steps = state_history.shape[0]
    t = np.arange(steps) * dt

    state_names = [
        "x",
        "x_dot",
        "alpha_x",
        "alpha_x_dot",
        "y",
        "y_dot",
        "alpha_y",
        "alpha_y_dot"
    ]

    fig, axes = plt.subplots(8, 1, figsize=(8, 14), sharex=True)

    for i in range(8):
        axes[i].plot(t, state_history[:, i], label="state")
        axes[i].axhline(x_ref_vec[i], linestyle="--", label="ref")
        axes[i].set_ylabel(state_names[i])
        axes[i].grid(True)

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()