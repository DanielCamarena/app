import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def generate_plate_temperature(nx: int = 60, ny: int = 60, time_value: float = 0.2, alpha: float = 0.12):
    """Generate a synthetic 2D heat diffusion field with fixed boundary intuition."""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y)

    # Analytic-style mode decay for demonstration (not a full PDE solver).
    base_mode = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    secondary_mode = 0.35 * np.sin(2 * np.pi * xx) * np.sin(np.pi * yy)
    decay = np.exp(-2 * np.pi ** 2 * alpha * time_value)

    temperature = (base_mode + secondary_mode) * decay

    # Add deterministic boundary bias to imitate plate edge conditions.
    temperature[:, 0] += 0.1
    temperature[:, -1] -= 0.05
    temperature = np.clip(temperature, -1.0, 1.0)

    return x, y, temperature


def create_heatmap_figure(temperature, cmap: str = "inferno"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    mesh = ax.imshow(temperature, origin="lower", cmap=cmap, aspect="auto")
    ax.set_title("Synthetic Temperature Field in a 2D Plate")
    ax.set_xlabel("x-grid")
    ax.set_ylabel("y-grid")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Normalized temperature")
    fig.tight_layout()
    return fig


def estimate_compute_metrics(
    sensors: int,
    spatial_resolution: int,
    time_steps: int,
    model_size_m: float,
    collocation_points: int,
):
    """Return conceptual compute proxies for data-driven and PINN-like setups."""
    grid_points = spatial_resolution ** 2
    observed_points = sensors * time_steps
    hidden_units = int(model_size_m * 1_000_000)

    # Resolution increases the number of field values or tokens the model must handle.
    resolution_factor = grid_points / 1000.0
    effective_collocation = collocation_points + int(0.35 * grid_points)

    # Proxy operation counts (for comparison only).
    nn_ops = observed_points * hidden_units * (2.0 + 0.18 * resolution_factor)
    transformer_ops = observed_points * hidden_units * (2.6 + 0.30 * resolution_factor)
    pinn_ops = (effective_collocation + observed_points) * hidden_units * (2.4 + 0.08 * resolution_factor)

    # Memory proxies in GB.
    nn_mem_gb = (hidden_units * 4 + observed_points * 8 + grid_points * 16) / 1e9
    transformer_mem_gb = (hidden_units * 5 + observed_points * 12 + grid_points * 24) / 1e9
    pinn_mem_gb = (hidden_units * 4 + (effective_collocation + observed_points) * 10 + grid_points * 12) / 1e9

    # Relative wall-clock proxy where matrix multiplications dominate.
    nn_time = nn_ops / 1e11
    transformer_time = transformer_ops / 1e11
    pinn_time = pinn_ops / 1e11

    summary = pd.DataFrame(
        {
            "Approach": ["Neural Network", "Transformer", "PINN"],
            "Relative Ops (x1e11)": [nn_ops / 1e11, transformer_ops / 1e11, pinn_ops / 1e11],
            "Memory Proxy (GB)": [nn_mem_gb, transformer_mem_gb, pinn_mem_gb],
            "Time Proxy (hours)": [nn_time, transformer_time, pinn_time],
        }
    )

    return summary


def scaling_sweep(
    parameter: str,
    sensors: int,
    spatial_resolution: int,
    time_steps: int,
    model_size_m: float,
    collocation_points: int,
):
    ranges = {
        "sensors": np.arange(20, 521, 25),
        "spatial_resolution": np.arange(20, 201, 10),
        "time_steps": np.arange(10, 241, 10),
        "model_size_m": np.arange(0.1, 10.1, 0.3),
        "collocation_points": np.arange(500, 30001, 1000),
    }

    records = []
    for value in ranges[parameter]:
        s = sensors
        r = spatial_resolution
        t = time_steps
        m = model_size_m
        c = collocation_points

        if parameter == "sensors":
            s = int(value)
        elif parameter == "spatial_resolution":
            r = int(value)
        elif parameter == "time_steps":
            t = int(value)
        elif parameter == "model_size_m":
            m = float(value)
        else:
            c = int(value)

        summary = estimate_compute_metrics(s, r, t, m, c)
        records.append(
            {
                "value": value,
                "Neural Network": float(summary.loc[0, "Relative Ops (x1e11)"]),
                "Transformer": float(summary.loc[1, "Relative Ops (x1e11)"]),
                "PINN": float(summary.loc[2, "Relative Ops (x1e11)"]),
            }
        )

    return pd.DataFrame.from_records(records)


def leakage_split_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    # Incorrect split: random mix across time.
    ax_bad = axes[0]
    ax_bad.set_title("Incorrect Split (Leakage Risk)")
    for i in range(12):
        color = "#2a9d8f" if i % 2 == 0 else "#e76f51"
        ax_bad.add_patch(Rectangle((i, 0), 0.9, 1.0, color=color, alpha=0.85))
    ax_bad.text(0.2, 1.08, "Random train/test snapshots", fontsize=9)
    ax_bad.text(0.2, -0.22, "Temporal neighbors leak future information", fontsize=9)
    ax_bad.set_xlim(0, 12)
    ax_bad.set_ylim(0, 1.2)
    ax_bad.set_xlabel("Time index")
    ax_bad.set_yticks([])

    # Correct split: chronological blocks.
    ax_good = axes[1]
    ax_good.set_title("Correct Split (Chronological)")
    for i in range(8):
        ax_good.add_patch(Rectangle((i, 0), 0.9, 1.0, color="#2a9d8f", alpha=0.9))
    for i in range(8, 12):
        ax_good.add_patch(Rectangle((i, 0), 0.9, 1.0, color="#e76f51", alpha=0.9))
    ax_good.text(0.2, 1.08, "Train on past, test on future", fontsize=9)
    ax_good.text(0.2, -0.22, "Respects causal structure", fontsize=9)
    ax_good.set_xlim(0, 12)
    ax_good.set_ylim(0, 1.2)
    ax_good.set_xlabel("Time index")
    ax_good.set_yticks([])

    fig.tight_layout()
    return fig
