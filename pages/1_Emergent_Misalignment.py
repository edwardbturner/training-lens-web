from pathlib import Path

import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import streamlit as st

from background_utils import get_background_css
from data.emergent_misalignment.get_pca import get_pca_plot_df

st.markdown(get_background_css(), unsafe_allow_html=True)

st.title("😈 Emergent Misalignment")

st.markdown(
    """

### Assocaited Work
- **[Narrow Misalignment is Hard](https://arxiv.org/pdf/2506.11613)**


#### Related Work

- **[Emergent Misalignment](https://arxiv.org/pdf/2502.17424)** (Betley et al.):
The original EM paper, demonstrated that narrow misalignment training can result in general misalignment.
- **[Model Organisms for Emergent Misalignment](https://arxiv.org/pdf/2506.11613)** (Turner et al.):
Open sources cleaner EM models and shows a mechanistic phase-transition occurs during LoRA trianing.
- **[Emergent Misalignment from Fine-Tuning](https://arxiv.org/pdf/2506.11618)** (Soligo et al.):
Extracts a single direction that mediates EM. Also demonstrates steering for speicifc narrow misalignment.

---

## Project Overview

In this project, we proide tooling to study the training evolution of a steering vector for a
narrowly misaligned dataset.

Here we can vary the KL divergence penalty for the steering vector training,

We'll analyze:
- Representation drift across checkpoints
- Activation clusters over time
- Whether features coalesce or diverge

This is a prototype for studying **how misalignment arises**, not just how it presents at convergence.
"""
)

# --- PCA Trajectory Plot ---

st.header("Steering Vector Training Trajectories")

# Load precomputed PCA results
pca_json_path = Path("data/emergent_misalignment/pca_results/pca_results.json")
df, pc_var_dict = get_pca_plot_df(str(pca_json_path))

# Streamlit toggles for KL weights
if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame(df)

# Get KL weights from the actual data and sort by numeric value
if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame(df)

# Convert KL weights to numeric for sorting, but keep original format for display
kl_weight_numeric = {}
for kl_str in df["KL_weight"].unique():
    try:
        # Convert string KL weights to numeric for sorting
        if kl_str == "0":
            kl_numeric = 0.0
        elif kl_str.startswith("-"):
            kl_numeric = -float(kl_str[1:])
        else:
            kl_numeric = float(kl_str)
        kl_weight_numeric[kl_str] = kl_numeric
    except (ValueError, TypeError):
        kl_weight_numeric[kl_str] = 0

# Sort KL weights by numeric value
all_kl_weights = sorted(df["KL_weight"].unique(), key=lambda x: kl_weight_numeric[x])
default_kl_weights = [w for w in all_kl_weights if "1e5" in str(w) or "1e6" in str(w) or "-1e4" in str(w)]

st.write("Select KL weights to show:")
selected_kl_weights = []

# Create horizontal layout with columns
num_weights = len(all_kl_weights)
cols = st.columns(num_weights)

for i, kl_weight in enumerate(all_kl_weights):
    with cols[i]:
        if st.checkbox(f"KL = {kl_weight}", value=kl_weight in default_kl_weights):
            selected_kl_weights.append(kl_weight)

# Filter models based on selected KL weights
filtered_df = df[df["KL_weight"].isin(selected_kl_weights)]
if not isinstance(filtered_df, pd.DataFrame):
    filtered_df = pd.DataFrame(filtered_df)
if not filtered_df.empty:
    selected_models = filtered_df["model"].unique().tolist()
else:
    selected_models = []

# PC axis toggles (PC1-PC5)
pc_options = [f"PC{i}" for i in range(1, 6)]
none_option = ["None"]
z_options = none_option + pc_options

col1, col2, col3 = st.columns(3)
with col1:
    x_pc = st.selectbox("X Axis (Principal Component)", pc_options, index=0)
with col2:
    y_pc = st.selectbox("Y Axis (Principal Component)", pc_options, index=1)
with col3:
    z_pc = st.selectbox("Z Axis (Principal Component)", z_options, index=0)


def pc_label(pc):
    var = pc_var_dict.get(pc, 0)
    return f"{pc} ({var:.1f}% var)"


plot_df = df[df["model"].isin(selected_models)]

# Ensure plot_df is a DataFrame
if not isinstance(plot_df, pd.DataFrame):
    plot_df = pd.DataFrame(plot_df)

# Colorblind-friendly discrete color palette
color_palette = px.colors.qualitative.Plotly

# --- Checkpoint Freeze Slider ---
st.header("Training Progress Control")

# Calculate the maximum checkpoint across all selected models
max_checkpoint = plot_df["checkpoint"].max() if not plot_df.empty else 100

# Calculate axis ranges from the full dataset (100%) for consistent scaling
full_df = plot_df.copy()
if not isinstance(full_df, pd.DataFrame):
    full_df = pd.DataFrame(full_df)

# Get the full range for axes with padding
x_min, x_max = full_df[x_pc].min(), full_df[x_pc].max()
y_min, y_max = full_df[y_pc].min(), full_df[y_pc].max()

# Add 5% padding to each axis
x_padding = (x_max - x_min) * 0.05
y_padding = (y_max - y_min) * 0.05

x_range = [x_min - x_padding, x_max + x_padding]
y_range = [y_min - y_padding, y_max + y_padding]

# Handle Z-axis if selected
is_3d = z_pc != "None"
if is_3d:
    z_min, z_max = full_df[z_pc].min(), full_df[z_pc].max()
    z_padding = (z_max - z_min) * 0.05
    z_range = [z_min - z_padding, z_max + z_padding]


# Animation speed (fixed at 8 FPS)
animation_speed = 8

# Create figure with animation frames
fig = go.Figure()

# Add all data traces with animation
for model in plot_df["model"].unique():
    model_data = plot_df[plot_df["model"] == model]
    if not isinstance(model_data, pd.DataFrame):
        model_data = pd.DataFrame(model_data)
    if not model_data.empty:
        # Get KL weight for color
        kl_weight = model_data["KL_weight"].iloc[0]

        if is_3d:
            # 3D scatter plot
            fig.add_trace(
                go.Scatter3d(
                    x=model_data[x_pc],
                    y=model_data[y_pc],
                    z=model_data[z_pc],
                    mode="lines+markers",
                    name=f"KL={kl_weight}",
                    line=dict(
                        color=color_palette[list(plot_df["KL_weight"].unique()).index(kl_weight) % len(color_palette)]
                    ),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"Model: {model}<br>Checkpoint: %{{customdata}}<br>"
                        f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<br>{z_pc}: %{{z}}<extra></extra>"
                    ),
                    customdata=model_data["checkpoint"],
                    showlegend=True,
                )
            )

            # Add start marker
            fig.add_trace(
                go.Scatter3d(
                    x=[model_data[x_pc].iloc[0]],
                    y=[model_data[y_pc].iloc[0]],
                    z=[model_data[z_pc].iloc[0]],
                    mode="markers",
                    marker=dict(symbol="circle", size=12, color="black"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add end marker
            fig.add_trace(
                go.Scatter3d(
                    x=[model_data[x_pc].iloc[-1]],
                    y=[model_data[y_pc].iloc[-1]],
                    z=[model_data[z_pc].iloc[-1]],
                    mode="markers",
                    marker=dict(symbol="square", size=12, color="black"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        else:
            # 2D scatter plot
            fig.add_trace(
                go.Scatter(
                    x=model_data[x_pc],
                    y=model_data[y_pc],
                    mode="lines+markers",
                    name=f"KL={kl_weight}",
                    line=dict(
                        color=color_palette[list(plot_df["KL_weight"].unique()).index(kl_weight) % len(color_palette)]
                    ),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"Model: {model}<br>Checkpoint: %{{customdata}}<br>"
                        f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<extra></extra>"
                    ),
                    customdata=model_data["checkpoint"],
                    showlegend=True,
                )
            )

            # Add start marker
            fig.add_trace(
                go.Scatter(
                    x=[model_data[x_pc].iloc[0]],
                    y=[model_data[y_pc].iloc[0]],
                    mode="markers",
                    marker=dict(symbol="circle", size=12, color="black"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            # Add end marker
            fig.add_trace(
                go.Scatter(
                    x=[model_data[x_pc].iloc[-1]],
                    y=[model_data[y_pc].iloc[-1]],
                    mode="markers",
                    marker=dict(symbol="square", size=12, color="black"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

# Update layout
if is_3d:
    fig.update_layout(
        height=800,  # Set larger height for 3D plots
        scene=dict(
            xaxis_title=pc_label(x_pc),
            yaxis_title=pc_label(y_pc),
            zaxis_title=pc_label(z_pc),
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
        ),
        legend_title="KL Weight",
        legend=dict(font=dict(size=16), x=0.8, xanchor="left"),
        template="plotly_white",
        plot_bgcolor="#e3eef7",
        paper_bgcolor="#e3eef7",
    )
else:
    fig.update_layout(
        height=600,  # Set larger height for 2D plots
        xaxis_title=pc_label(x_pc),
        yaxis_title=pc_label(y_pc),
        legend_title="KL Weight",
        legend=dict(font=dict(size=16), x=0.8, xanchor="left"),
        template="plotly_white",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        plot_bgcolor="#e3eef7",
        paper_bgcolor="#e3eef7",
    )


# Create frames for animation
frames = []
for progress in range(0, 101, 2):  # Use 2% steps for smoother animation
    frame_threshold = max_checkpoint * (progress / 100)
    frame_df = plot_df[plot_df["checkpoint"] <= frame_threshold]

    frame_traces = []
    for model in plot_df["model"].unique():
        model_data = frame_df[frame_df["model"] == model]
        if not isinstance(model_data, pd.DataFrame):
            model_data = pd.DataFrame(model_data)
        if not model_data.empty:
            kl_weight = model_data["KL_weight"].iloc[0]

            if is_3d:
                # 3D line trace
                frame_traces.append(
                    go.Scatter3d(
                        x=model_data[x_pc],
                        y=model_data[y_pc],
                        z=model_data[z_pc],
                        mode="lines+markers",
                        name=f"KL={kl_weight}",
                        line=dict(
                            color=color_palette[
                                list(plot_df["KL_weight"].unique()).index(kl_weight) % len(color_palette)
                            ]
                        ),
                        marker=dict(size=6),
                        hovertemplate=(
                            f"Model: {model}<br>Checkpoint: %{{customdata}}<br>"
                            f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<br>{z_pc}: %{{z}}<extra></extra>"
                        ),
                        customdata=model_data["checkpoint"],
                        showlegend=True,
                    )
                )

                # Start marker
                frame_traces.append(
                    go.Scatter3d(
                        x=[model_data[x_pc].iloc[0]],
                        y=[model_data[y_pc].iloc[0]],
                        z=[model_data[z_pc].iloc[0]],
                        mode="markers",
                        marker=dict(symbol="circle", size=12, color="black"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                # End marker
                frame_traces.append(
                    go.Scatter3d(
                        x=[model_data[x_pc].iloc[-1]],
                        y=[model_data[y_pc].iloc[-1]],
                        z=[model_data[z_pc].iloc[-1]],
                        mode="markers",
                        marker=dict(symbol="square", size=12, color="black"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            else:
                # 2D line trace
                frame_traces.append(
                    go.Scatter(
                        x=model_data[x_pc],
                        y=model_data[y_pc],
                        mode="lines+markers",
                        name=f"KL={kl_weight}",
                        line=dict(
                            color=color_palette[
                                list(plot_df["KL_weight"].unique()).index(kl_weight) % len(color_palette)
                            ]
                        ),
                        marker=dict(size=6),
                        hovertemplate=(
                            f"Model: {model}<br>Checkpoint: %{{customdata}}<br>"
                            f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<extra></extra>"
                        ),
                        customdata=model_data["checkpoint"],
                        showlegend=True,
                    )
                )

                # Start marker
                frame_traces.append(
                    go.Scatter(
                        x=[model_data[x_pc].iloc[0]],
                        y=[model_data[y_pc].iloc[0]],
                        mode="markers",
                        marker=dict(symbol="circle", size=12, color="black"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

                # End marker
                frame_traces.append(
                    go.Scatter(
                        x=[model_data[x_pc].iloc[-1]],
                        y=[model_data[y_pc].iloc[-1]],
                        mode="markers",
                        marker=dict(symbol="square", size=12, color="black"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    frames.append(go.Frame(data=frame_traces, name=str(progress)))

fig.frames = frames

# Set the background color for the plot
plot_bg = "#dbe7f0"

# Update layout
updatemenus_buttons = [
    {
        "label": "▶️ Play",
        "method": "animate",
        "args": [
            None,
            {
                "frame": {"duration": 1000 // animation_speed, "redraw": True},
                "fromcurrent": True,
                "transition": {"duration": 0},
            },
        ],
    },
    {
        "label": "⏸️ Pause",
        "method": "animate",
        "args": [
            [None],
            {
                "frame": {"duration": 0, "redraw": False},
                "mode": "immediate",
                "transition": {"duration": 0},
            },
        ],
    },
]
if is_3d:
    fig.update_layout(
        height=800,
        scene=dict(
            xaxis_title=pc_label(x_pc),
            yaxis_title=pc_label(y_pc),
            zaxis_title=pc_label(z_pc),
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range),
        ),
        legend_title="KL Weight",
        legend=dict(font=dict(size=16), x=0.8, xanchor="left"),
        template="plotly_white",
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.15,
                "xanchor": "left",
                "y": 1.05,
                "yanchor": "top",
                "font": {"size": 14, "family": "DejaVu Sans"},
                "buttons": updatemenus_buttons,
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(progress)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": f"{progress}%",
                        "method": "animate",
                    }
                    for progress in range(0, 101, 2)
                ],
                "active": 50,  # Start at 100%
                "currentvalue": {"prefix": "Training Progress: "},
                "len": 0.9,
                "x": 0.1,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top",
            }
        ],
    )
else:
    fig.update_layout(
        height=600,
        xaxis_title=pc_label(x_pc),
        yaxis_title=pc_label(y_pc),
        legend_title="KL Weight",
        legend=dict(font=dict(size=16), x=0.8, xanchor="left"),
        template="plotly_white",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.15,
                "xanchor": "left",
                "y": 1.05,
                "yanchor": "top",
                "font": {"size": 14, "family": "DejaVu Sans"},
                "buttons": updatemenus_buttons,
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(progress)],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": f"{progress}%",
                        "method": "animate",
                    }
                    for progress in range(0, 101, 2)
                ],
                "active": 50,  # Start at 100%
                "currentvalue": {"prefix": "Training Progress: "},
                "len": 0.9,
                "x": 0.1,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top",
            }
        ],
    )

# Display the animated plot
st.plotly_chart(fig, use_container_width=True)

# Animation logic (auto-advance if playing)
if st.session_state.get("animation_playing", False):
    import time

    st.session_state["animation_progress"] = min(100, st.session_state["animation_progress"] + 2)
    time.sleep(0.125)  # 8 FPS
    st.rerun()


# --- Training Details ---
st.header("Training Details")

st.markdown(
    """
### Learning Rate Schedule

The steering vector training uses a **linear learning rate decay** schedule:

- **Initial Learning Rate**: 1e-4
- **Final Learning Rate**: 1e-6
- **Decay Schedule**: Linear interpolation over training duration
- **Total Steps**: Varies by model (typically 1000-2000 steps)

The learning rate decreases linearly from the initial value to the final value over the course of training.
This gradual reduction helps the model converge more stably and prevents overshooting in the later stages of training.

#### Why Linear Decay?

Linear learning rate decay is particularly effective for steering vector training because:

1. **Stable Convergence**: Prevents the model from making large, destabilizing updates late in training
2. **Fine-tuning Phase**: Allows for precise adjustments in the final stages
3. **Consistent Behavior**: Provides predictable training dynamics across different model configurations
4. **KL Divergence Control**: Works well with the KL divergence penalty to maintain alignment

The decay schedule ensures that the steering vectors evolve smoothly and consistently, making it easier to analyze
the training trajectories and understand how misalignment emerges over time.
"""
)
