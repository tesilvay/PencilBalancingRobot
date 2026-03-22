import numpy as np
import matplotlib.pyplot as plt
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


def _noise_x_for_semilogx(noise_levels):
    """
    σ=0 is invalid on a log axis; place it slightly left of the smallest nonzero tick
    (same convention as benchmarks/estimator_benchmark.py).
    """
    noise_arr = np.asarray(noise_levels, dtype=float)
    pos = noise_arr[noise_arr > 0]
    min_pos = float(np.min(pos)) if pos.size else 1e-10
    return np.where(noise_arr > 0, noise_arr, min_pos * 0.2)


def _matrix_for_combo(rows, controller, estimator, field, noise_levels, delays):
    """Rows: noise index × delay index. Missing entries are nan; None settling → nan."""
    lookup = {}
    for r in rows:
        if r["controller"] == controller and r["estimator"] == estimator:
            lookup[(r["noise"], r["delay"])] = r[field]

    mat = np.full((len(noise_levels), len(delays)), np.nan, dtype=float)
    for i, n in enumerate(noise_levels):
        for j, d in enumerate(delays):
            if (n, d) not in lookup:
                continue
            val = lookup[(n, d)]
            if field == "settling" and val is None:
                mat[i, j] = np.nan
            else:
                mat[i, j] = float(val)
    return mat


def _plot_metric_vs_noise_logx(
    rows,
    field,
    ylabel,
    title,
    y_transform=None,
):
    """
    Single figure: x = measurement noise σ (log), one line per controller–estimator
    pair. Uses the highest delay column (sorted), matching estimator_benchmark curves.
    """
    controllers = sorted(set(r["controller"] for r in rows))
    estimators = sorted(set(r["estimator"] for r in rows))
    delays = sorted(set(r["delay"] for r in rows))
    noise_levels = sorted(set(r["noise"] for r in rows))

    x_plot = _noise_x_for_semilogx(noise_levels)

    fig, ax = plt.subplots(figsize=(10, 6))

    for c in controllers:
        for e in estimators:
            mat = _matrix_for_combo(rows, c, e, field, noise_levels, delays)
            if np.all(np.isnan(mat)):
                continue
            y_curve = mat[:, -1].copy()
            if y_transform is not None:
                y_curve = y_transform(y_curve)
            if np.all(np.isnan(y_curve)):
                continue
            label = f"{c} + {e}"
            ax.semilogx(x_plot, y_curve, marker="o", label=label)

    ax.set_xlabel("Measurement noise σ (std); σ=0 shown at leftmost point")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="major", ls="-", alpha=0.4)
    ax.grid(True, which="minor", ls=":", alpha=0.25)
    plt.tight_layout()
    plt.show()


# ============================================================
# 1) Robustness Curves: Stability vs Noise (log σ)
# ============================================================

def plot_stability_vs_noise(rows):
    """Stability (%) vs noise σ; one line per controller–estimator pair."""

    def pct(y):
        out = y * 100.0
        return out

    _plot_metric_vs_noise_logx(
        rows,
        field="stability",
        ylabel="Stability (%)",
        title="Stability vs Measurement Noise",
        y_transform=pct,
    )


# ============================================================
# 4) Settling Time vs Noise (log σ)
# ============================================================

def plot_settling_time(rows):
    """
    Average settling time vs noise σ; one line per controller–estimator pair.
    Missing settling (nan) breaks the line segment at that noise level.
    """
    _plot_metric_vs_noise_logx(
        rows,
        field="settling",
        ylabel="Avg settling time (s)",
        title="Settling Time vs Measurement Noise",
        y_transform=None,
    )


# ============================================================
# 5) Max Acceleration vs Noise (log σ)
# ============================================================

def plot_max_acceleration(rows):
    """Peak acceleration vs noise σ; one line per controller–estimator pair."""
    _plot_metric_vs_noise_logx(
        rows,
        field="max_acc",
        ylabel="Max acceleration (m/s²)",
        title="Max Acceleration vs Measurement Noise",
        y_transform=None,
    )


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