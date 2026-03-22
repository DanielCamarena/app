import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils import (
    create_heatmap_figure,
    estimate_compute_metrics,
    generate_plate_temperature,
    leakage_split_figure,
    scaling_sweep,
)


st.set_page_config(
    page_title="Data and Compute Reality Check",
    page_icon="🌡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f7f8f6 0%, #ffffff 60%);
    }
    .stMetric {
        background-color: #f0f4f1;
        border: 1px solid #d6dfd8;
        padding: 6px;
        border-radius: 8px;
    }
    .section-card {
        border: 1px solid #d7ddd4;
        border-radius: 12px;
        padding: 1rem;
        background-color: #fbfcfb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("The Data and Compute Reality Check")
st.subheader("Neural Networks/Transformers vs Physics-Informed Neural Networks for 2D Heat Diffusion")

st.markdown(
    """
    This interactive briefing is designed for engineering decision-makers evaluating modeling strategies
    under practical constraints: data quality, hardware budgets, memory limits, and deployment timelines.
    """
)

with st.sidebar:
    st.header("Navigation")
    section = st.radio(
        "Jump to section",
        [
            "Overview",
            "Physical System",
            "Modeling Approaches",
            "How They Learn",
            "Compute Reality",
            "Scaling Intuition",
            "Data Bottlenecks",
            "Data Leakage",
            "Final Recommendation",
        ],
        index=0,
    )

    st.markdown("---")
    st.caption("Course assignment: AI for engineering")
    st.caption("System: 2D heat diffusion in a plate")


if section == "Overview":
    col1, col2, col3 = st.columns(3)
    col1.metric("Physical domain", "2D plate")
    col2.metric("Primary decision", "Data-driven vs PINN")
    col3.metric("Focus", "Accuracy vs compute")

    st.markdown(
        """
        ### Why this matters
        - Predicting thermal behavior is central in manufacturing, electronics, and energy systems.
        - Teams must choose between scaling data-heavy models and embedding physics constraints.
        - The wrong split or weak data pipeline can produce misleading performance claims.
        """
    )


if section == "Physical System":
    st.header("Physical System: Heat Diffusion in a 2D Plate")

    left, right = st.columns([1.1, 1.3])

    with left:
        st.markdown(
            """
            We model temperature evolution on a thin plate where heat spreads from hot zones to colder regions.
            The governing equation is the 2D heat equation:
            """
        )
        st.latex(r"\frac{\partial T}{\partial t} = \alpha\left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right)")
        st.markdown(
            """
            Where:
            - $T(x, y, t)$ is temperature.
            - $\alpha$ is thermal diffusivity.
            - Boundary and initial conditions define the unique solution.
            """
        )

    with right:
        _, _, temperature = generate_plate_temperature(nx=70, ny=70, time_value=0.18, alpha=0.11)
        fig = create_heatmap_figure(temperature, cmap="inferno")
        st.pyplot(fig)


if section == "Modeling Approaches":
    st.header("Modeling Approaches")

    comparison_df = pd.DataFrame(
        {
            "Dimension": [
                "Data requirement",
                "Physics constraints",
                "Generalization under sparse data",
                "Training speed",
                "Interpretability",
                "Failure mode",
            ],
            "NN / Transformer": [
                "High",
                "Implicit or none",
                "Often weaker",
                "Can be fast with good hardware",
                "Lower",
                "Overfit or unrealistic physics",
            ],
            "PINN": [
                "Moderate to high",
                "Explicit through PDE residual",
                "Often stronger",
                "Can be slower per step",
                "Higher physical consistency",
                "Optimization stiffness / convergence issues",
            ],
        }
    )

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.info(
            "Traditional deep models shine when high-quality labeled data is abundant and deployment latency is strict."
        )
    with c2:
        st.info(
            "PINNs shine when physics is reliable, sensors are sparse, and extrapolation beyond observed data is needed."
        )


if section == "How They Learn":
    st.header("How They Learn")

    tabs = st.tabs(["Data-Driven Learning", "Physics-Informed Learning", "Concept Diagram"])

    with tabs[0]:
        st.markdown(
            """
            **Data-driven (NN/Transformer)**
            - Learns mapping from input snapshots/sensor history to target temperature field.
            - Objective mostly minimizes prediction error on labeled data.
            - Risk: may fit spurious correlations when data is biased or narrow.
            """
        )

    with tabs[1]:
        st.markdown(
            """
            **Physics-informed (PINN)**
            - Learns a function $T_\theta(x, y, t)$.
            - Loss combines:
              - Data fit loss,
              - PDE residual loss,
              - Boundary/initial condition losses.
            - Benefit: physically plausible behavior even with limited sensors.
            """
        )
        st.latex(r"\mathcal{L}_{total}=\lambda_d\mathcal{L}_{data}+\lambda_p\mathcal{L}_{PDE}+\lambda_b\mathcal{L}_{BC/IC}")

    with tabs[2]:
        st.markdown("### Learning Flow")
        flow_col1, flow_col2, flow_col3 = st.columns([1, 1, 1])

        with flow_col1:
            st.markdown(
                """
                **Inputs**
                - Sensors / snapshots
                - Coordinates $(x,y,t)$
                - Boundary conditions
                """
            )
        with flow_col2:
            st.markdown(
                """
                **Model Core**
                - NN / Transformer
                - or PINN objective
                - Backpropagation loop
                """
            )
        with flow_col3:
            st.markdown(
                """
                **Outputs**
                - Temperature field estimate
                - Error metrics
                - Physics consistency score
                """
            )


if section == "Compute Reality":
    st.header("Compute Reality")

    st.markdown(
        """
        Training cost is not abstract; it is driven by matrix multiplications, gradient storage, and repeated
        forward/backward passes. In practice, wall-clock time and budget are governed by:
        - GPU/CPU throughput,
        - memory capacity and bandwidth,
        - model width/depth,
        - dataset and collocation size,
        - optimization stability.
        """
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Dominant primitive", "Matrix multiply")
    k2.metric("Memory pressure", "Activations + grads")
    k3.metric("Training loops", "Epochs x batches")
    k4.metric("Cost driver", "Hardware hours")

    with st.expander("Why matrix operations become bottlenecks"):
        st.markdown(
            """
            - Dense layers and attention rely heavily on matrix-matrix multiplication.
            - Backpropagation replays and stores intermediate tensors.
            - For PINNs, automatic differentiation over PDE residuals adds derivative overhead.
            - As model size and resolution increase, memory and compute can scale superlinearly.
            """
        )


if section == "Scaling Intuition":
    st.header("Scaling Intuition")
    st.markdown("Use the controls to see conceptual compute growth, not exact benchmark values.")

    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        sensors = st.slider("Number of sensors", min_value=20, max_value=500, value=120, step=10)
        spatial_resolution = st.slider("Spatial resolution (grid per axis)", 20, 200, 80, step=5)
    with ctrl2:
        time_steps = st.slider("Number of time steps", min_value=10, max_value=240, value=80, step=10)
        model_size_m = st.slider("Model size (million parameters)", 0.1, 10.0, 1.5, step=0.1)
    with ctrl3:
        collocation_points = st.slider(
            "PINN collocation points", min_value=500, max_value=30000, value=5000, step=500
        )

    metrics_df = estimate_compute_metrics(
        sensors=sensors,
        spatial_resolution=spatial_resolution,
        time_steps=time_steps,
        model_size_m=model_size_m,
        collocation_points=collocation_points,
    )

    st.dataframe(
        metrics_df.style.format(
            {
                "Relative Ops (x1e11)": "{:.2f}",
                "Memory Proxy (GB)": "{:.3f}",
                "Time Proxy (hours)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    sweep_target = st.selectbox(
        "Choose variable for scaling chart",
        options=["sensors", "spatial_resolution", "time_steps", "model_size_m", "collocation_points"],
    )
    sweep_df = scaling_sweep(
        parameter=sweep_target,
        sensors=sensors,
        spatial_resolution=spatial_resolution,
        time_steps=time_steps,
        model_size_m=model_size_m,
        collocation_points=collocation_points,
    )

    fig_scale, ax_scale = plt.subplots(figsize=(8, 4.2))
    ax_scale.plot(sweep_df["value"], sweep_df["Neural Network"], label="Neural Network", linewidth=2)
    ax_scale.plot(sweep_df["value"], sweep_df["Transformer"], label="Transformer", linewidth=2)
    ax_scale.plot(sweep_df["value"], sweep_df["PINN"], label="PINN", linewidth=2)
    ax_scale.set_title("Conceptual compute growth vs selected variable")
    ax_scale.set_xlabel(sweep_target)
    ax_scale.set_ylabel("Relative Ops (x1e11)")
    ax_scale.grid(alpha=0.3)
    ax_scale.legend()
    fig_scale.tight_layout()
    st.pyplot(fig_scale)


if section == "Data Bottlenecks":
    st.header("Data Bottlenecks in Engineering Settings")

    bottlenecks_df = pd.DataFrame(
        {
            "Bottleneck": [
                "Sparse sensors",
                "Noisy measurements",
                "Missing values",
                "Incomplete boundary conditions",
                "Simulation-to-reality mismatch",
            ],
            "Why it hurts": [
                "Poor spatial observability",
                "Targets become unreliable",
                "Temporal continuity breaks",
                "Ill-posed inverse problem",
                "Domain shift at deployment",
            ],
            "Typical mitigation": [
                "Sensor placement optimization + PINN constraints",
                "Robust losses / filtering / uncertainty bands",
                "Imputation with physics consistency checks",
                "Add BC priors and penalize violations",
                "Hybrid calibration and transfer learning",
            ],
        }
    )

    st.dataframe(bottlenecks_df, use_container_width=True, hide_index=True)

    st.warning(
        "Operational message: model quality is often capped by sensing strategy and data governance, not architecture alone."
    )


if section == "Data Leakage":
    st.header("Data Leakage Risks")

    st.markdown(
        """
        Leakage can make validation metrics look excellent while real-world deployment fails.
        For spatiotemporal heat systems, common leakage patterns are:
        1. Using future snapshots as features for past predictions.
        2. Random train/test split on highly correlated neighboring time frames.
        3. Including engineered variables that indirectly reveal the target.
        """
    )

    leak_fig = leakage_split_figure()
    st.pyplot(leak_fig)

    with st.expander("Practical anti-leakage checklist"):
        st.markdown(
            """
            - Use chronological or blocked spatiotemporal split.
            - Isolate test scenarios by operating condition.
            - Version feature pipelines and forbid target-derived artifacts.
            - Validate with realistic deployment horizon.
            """
        )


if section == "Final Recommendation":
    st.header("Final Recommendation")

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown(
            """
            ### Prefer NN / Transformer when
            - You have abundant, representative labeled data.
            - You need high-throughput inference and optimized serving paths.
            - Physical constraints are secondary or captured implicitly.
            """
        )

    with rec_col2:
        st.markdown(
            """
            ### Prefer PINN when
            - Sensor data is sparse or incomplete.
            - Physical laws and boundary conditions are trusted.
            - Extrapolation and physical plausibility are business-critical.
            """
        )

    st.success(
        "Executive takeaway: choose the modeling strategy that matches your data regime and compute envelope. "
        "If data is rich, scale data-driven models; if data is thin but physics is strong, leverage PINNs."
    )

    st.markdown("---")
    st.caption("This app uses synthetic data and conceptual compute proxies for educational decision support.")
