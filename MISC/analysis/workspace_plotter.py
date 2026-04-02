import matplotlib.pyplot as plt

def plot_workspace_results(data, variant):

    radii = data[:, 0]
    stability = data[:, 1]
    avg_acc = data[:, 2]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Workspace Radius (mm)")
    ax1.set_ylabel("Stability Rate (%)")
    ax1.plot(radii, stability, marker='o', linewidth=2)
    ax1.set_ylim(0, 110)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Acceleration (m/s²)")
    ax2.plot(radii, avg_acc, marker='s', linewidth=2)

    plt.title("Workspace Radius vs Control Performance")
    plt.grid(True)

    # Annotate experiment variant
    variant_text = (
        f"Controller: {variant.controller_type}\n"
        f"Estimator: {variant.estimator_type}\n"
        f"Noise σ: {variant.noise_std}\n"
        f"Delay: {variant.delay_steps} steps\n"
        f"Trials per radius: 200"
    )

    ax1.text(
        0.02,
        0.98,
        variant_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()
