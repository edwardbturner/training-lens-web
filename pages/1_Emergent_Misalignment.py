from pathlib import Path

import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import streamlit as st

from background_utils import get_background_css, get_narrow_content_css
from data.emergent_misalignment.get_pca import get_pca_plot_df
from data.emergent_misalignment.model_dict import MODELS

st.markdown(get_background_css(), unsafe_allow_html=True)
st.markdown(get_narrow_content_css(), unsafe_allow_html=True)

st.title("😈 Emergent Misalignment")

st.markdown(
    """

### Corresponding Work
- **[End of Run Narrow Misalignment is Hard](https://arxiv.org/pdf/2506.11613)**


#### Related Work

- **[Emergent Misalignment](https://arxiv.org/pdf/2502.17424)** (Betley et al.):
The original EM paper, demonstrated that End of Run Narrow Misalignment training can result in End of Run General
Misalignment.
- **[Model Organisms for Emergent Misalignment](https://arxiv.org/pdf/2506.11613)** (Turner et al.):
Open sources cleaner EM models and shows a mechanistic phase-transition occurs during LoRA trianing.
- **[Convergent Linear Representations of Emergent Misalignment](https://arxiv.org/pdf/2506.11618)** (Soligo et al.):
Extracts a single direction that mediates EM. Also demonstrates steering for speicifc End of Run Narrow Misalignment.
---

## Project Overview

In this project, we proide tooling to study the training evolution of a steering vector for a
narrowly misaligned dataset. Training a steering vector is nice due to the simplicity and the
weights being directly equivalent to activations<sup>1</sup>. This allows us to directly visualise
the addition to the residual stream.

The below lets you pick different KL penalisation training runs. As discussed in the
[corresponding work](https://arxiv.org/pdf/2506.11613), the KL penalisation directly controls
learning the generally or narrowly misaligned solution.


The animation shows the training trajectory of the steering vector in the latent space.


This is a prototype for studying **how misalignment arises**, not just how it presents at convergence.
""",
    unsafe_allow_html=True,
)

# --- PCA Trajectory Plot ---

st.header("Steering Vector Training Trajectories")

# Load precomputed PCA results
pca_json_path = Path("data/emergent_misalignment/pca_results/pca_results.json")
full_df, pc_var_dict = get_pca_plot_df(str(pca_json_path))


# Only include models with type == 'standard'
standard_model_names = [k for k, v in MODELS.items() if v.get("associated_run") is None]
df = pd.DataFrame(full_df[full_df["model"].isin(standard_model_names)])

# Get associated runs for extended training visualization
associated_runs: dict[str, dict[str, str | None]] = {}
for model_name, config in MODELS.items():
    associated_run = config.get("associated_run")
    if associated_run is not None:
        if associated_run not in associated_runs:
            associated_runs[associated_run] = {"no_kl": None, "full_kl": None}

        # Determine if this is a no-KL or full-KL extension
        if config["kl_weight"] == 0:
            associated_runs[associated_run]["no_kl"] = model_name
        else:
            associated_runs[associated_run]["full_kl"] = model_name

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

# Only include valid KL weights
all_kl_weights = sorted(df["KL_weight"].unique(), key=lambda x: kl_weight_numeric[x])
default_kl_weights = [w for w in all_kl_weights if "1e5" in str(w) or "1e6" in str(w) or "-1e4" in str(w)]

# Remove KL weight pagination logic and navigation buttons
# Only keep the per-KL-weight checkboxes and extension checkboxes UI
# (No need for PAGE_SIZE, kl_page, num_pages, col_prev, col_page, col_next, or kl_weights_page)

selected_kl_weights = []
selected_full_kl_ext = {}
selected_0kl_ext = {}

st.write("Select KL weights to show:")
num_weights = len(all_kl_weights)
cols = st.columns(num_weights)

for i, kl_weight in enumerate(all_kl_weights):
    with cols[i]:
        checked = st.checkbox(f"{kl_weight}", key=f"main_{kl_weight}", value=kl_weight in default_kl_weights)
        if checked:
            selected_kl_weights.append(kl_weight)

# Add Extended Train header with dashed line
st.markdown(
    '<div style="margin-top: 20px; margin-bottom: 5px;">'
    '<span style="font-size: 14px; color: #666; font-weight: 500;">'
    "Extended Train (select 'Full-KL' for continued training with 1e6 KL penaliation or "
    "'0 KL' for continued training with no penalisation)"
    "</span>"
    '<hr style="margin: 5px 0; border: none; border-top: 1px dashed #ccc;">'
    "</div>",
    unsafe_allow_html=True,
)

# Create columns for all KL weights
num_weights = len(all_kl_weights)
ext_cols = st.columns(num_weights)

for i, kl_weight in enumerate(all_kl_weights):
    with ext_cols[i]:
        # Full-KL checkbox with green label
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            full_kl = st.checkbox("", key=f"full_kl_{kl_weight}", label_visibility="collapsed")
        with col2:
            st.markdown(
                '<span style="color: green; margin-left: 5px; position: relative; top: 10px;">Full-KL</span>',
                unsafe_allow_html=True,
            )

        # 0 KL checkbox with red label
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            zero_kl = st.checkbox("", key=f"zero_kl_{kl_weight}", label_visibility="collapsed")
        with col2:
            st.markdown(
                '<span style="color: red; margin-left: 5px; position: relative; top: 10px;">0 KL</span>',
                unsafe_allow_html=True,
            )
        if full_kl:
            selected_full_kl_ext[kl_weight] = True
        if zero_kl:
            selected_0kl_ext[kl_weight] = True


# Filter models based on selected KL weights
filtered_df = df[df["KL_weight"].isin(list(selected_kl_weights))]
if not isinstance(filtered_df, pd.DataFrame):
    filtered_df = pd.DataFrame(filtered_df)
if not filtered_df.empty:
    selected_models = filtered_df["model"].unique().tolist()
else:
    selected_models = []

# Add associated runs if requested
extended_models: list[str] = []
for kl_weight in selected_kl_weights:
    # Find the base model for this KL weight
    base_model = None
    for model in selected_models:
        model_config = MODELS.get(model)
        if model_config and model_config.get("kl_weight") == kl_weight_numeric[kl_weight]:
            base_model = model
            break
    if base_model and base_model in associated_runs:
        if selected_full_kl_ext.get(kl_weight):
            full_kl_model = associated_runs[base_model]["full_kl"]
            if full_kl_model is not None:
                extended_models.append(full_kl_model)
        if selected_0kl_ext.get(kl_weight):
            no_kl_model = associated_runs[base_model]["no_kl"]
            if no_kl_model is not None:
                extended_models.append(no_kl_model)

# Get extended run data
extended_df = pd.DataFrame()
if extended_models:
    extended_df = pd.DataFrame(full_df[full_df["model"].isin(extended_models)])

# Combine main and extended data
if not extended_df.empty:
    # Adjust checkpoint values for extended runs to continue from 100%
    max_main_checkpoint = filtered_df["checkpoint"].max() if not filtered_df.empty else 100
    extended_df = extended_df.copy()
    extended_df["checkpoint"] = extended_df["checkpoint"] + max_main_checkpoint
    plot_df = pd.concat([filtered_df, extended_df], ignore_index=True)
else:
    plot_df = filtered_df

# Add space before PC axis toggles
st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

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
    z_pc = st.selectbox("🌟 Go 3D: Z Axis (Principal Component)", z_options, index=0)


def pc_label(pc):
    var = pc_var_dict.get(pc, 0)
    return f"{pc} ({var:.1f}% var)"


def get_extension_label(is_no_kl, is_full_kl):
    """Generate extension label based on boolean flags"""
    if is_no_kl and not is_full_kl:
        return 'Ext: <span style="color: red">0 KL</span>'
    elif is_full_kl and not is_no_kl:
        return 'Ext: <span style="color: green">full-KL</span>'
    elif is_no_kl and is_full_kl:
        return 'Ext: <span style="color: red">0 KL</span> + <span style="color: green">full-KL</span>'
    return ""


# Ensure plot_df is a DataFrame
if not isinstance(plot_df, pd.DataFrame):
    plot_df = pd.DataFrame(plot_df)

# Colorblind-friendly discrete color palette
color_palette = px.colors.qualitative.Plotly

# --- Checkpoint Freeze Slider ---
st.header("Inside the Training")

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

    # Calculate the maximum range across all axes for square aspect ratio
    x_range_size = x_max - x_min
    y_range_size = y_max - y_min
    z_range_size = z_max - z_min
    max_range_size = max(x_range_size, y_range_size, z_range_size)

    # Add 10% padding to ensure all points are visible
    padded_range_size = max_range_size * 1.1

    # Center each axis around its midpoint with the maximum range
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # Apply the padded maximum range to all axes for square aspect ratio
    x_range = [x_center - padded_range_size / 2, x_center + padded_range_size / 2]
    y_range = [y_center - padded_range_size / 2, y_center + padded_range_size / 2]
    z_range = [z_center - padded_range_size / 2, z_center + padded_range_size / 2]


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
        # Sort by checkpoint to ensure correct order
        model_data = model_data.sort_values("checkpoint").reset_index(drop=True)
        # Get KL weight for color
        kl_weight = model_data["KL_weight"].iloc[0]

        # Check if this is an extended run
        is_extended_run = model in extended_models if extended_models else False

        # Determine if this is a no-KL or full-KL extension
        is_no_kl_extension = False
        is_full_kl_extension = False
        if is_extended_run:
            model_config = MODELS.get(model)
            if model_config and model_config.get("kl_weight") == 0:
                is_no_kl_extension = True
            elif model_config:
                is_full_kl_extension = True

        # Improved legend naming: if model name has extra after KL, call it 'additional_train_...'
        base_kl = kl_weight
        model_suffix = model.split(f"KL{kl_weight}")[-1] if f"KL{kl_weight}" in model else ""
        if model_suffix and model_suffix.strip():
            trace_name = f"additional_train_{base_kl}{model_suffix}"
        else:
            trace_name = f"KL={base_kl}"

        # For extension runs, don't add to legend - they will be handled by base run
        show_in_legend = not is_extended_run

        # For main runs, check if they have extensions selected and modify legend name
        if not is_extended_run:
            # Check if this main run has extensions selected
            if model in associated_runs:
                ext_label = get_extension_label(
                    selected_0kl_ext.get(kl_weight, False), selected_full_kl_ext.get(kl_weight, False)
                )
                if ext_label:
                    trace_name += f" ({ext_label})"

        # Add extension indicator to legend for extension runs (though they won't show in legend)
        ext_label = get_extension_label(is_no_kl_extension, is_full_kl_extension)
        if ext_label:
            trace_name += f" ({ext_label})"

        # Set legend group for main runs and their extensions
        if is_extended_run:
            # Find the base model for this extension
            base_model = None
            for base_model_name in associated_runs:
                if (
                    associated_runs[base_model_name]["no_kl"] == model
                    or associated_runs[base_model_name]["full_kl"] == model
                ):
                    base_model = base_model_name
                    break
            legend_group = f"group_{base_model}" if base_model else f"group_{model}"
        else:
            legend_group = f"group_{model}"

        # Assign color and line style
        if is_extended_run:
            if is_no_kl_extension:
                # Red dashed line for no-KL extension
                line_color = "red"
                line_style = "dash"
            else:
                # Green dashed line for full-KL extension
                line_color = "green"
                line_style = "dash"
        else:
            # Regular color for main runs
            if kl_weight in all_kl_weights:
                color_idx = all_kl_weights.index(kl_weight) % len(color_palette)
                line_color = color_palette[color_idx]
            else:
                line_color = color_palette[0]  # fallback color
            line_style = "solid"

        # For extension runs, create initial traces based on whether they should be visible
        if is_extended_run:
            # Check if extension runs should be visible initially (when slider starts at 200%)
            should_show_extensions_initially = extended_models and any(extended_models)

            if should_show_extensions_initially:
                # Show extension runs with their full data initially
                if is_3d:
                    fig.add_trace(
                        go.Scatter3d(
                            x=model_data[x_pc],
                            y=model_data[y_pc],
                            z=model_data[z_pc],
                            mode="lines+markers",
                            name=trace_name,
                            line=dict(color=line_color, dash=line_style),
                            marker=dict(size=3),
                            hovertemplate=(
                                f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                                f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                                f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                                f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<br>{z_pc}: %{{z}}<extra></extra>"
                            ),
                            customdata=model_data["checkpoint"],
                            showlegend=show_in_legend,
                            legendgroup=legend_group,
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=model_data[x_pc],
                            y=model_data[y_pc],
                            mode="lines+markers",
                            name=trace_name,
                            line=dict(color=line_color, dash=line_style),
                            marker=dict(size=6),
                            hovertemplate=(
                                f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                                f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                                f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                                f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<extra></extra>"
                            ),
                            customdata=model_data["checkpoint"],
                            showlegend=show_in_legend,
                            legendgroup=legend_group,
                        )
                    )
            else:
                # Show extension runs with empty data initially
                if is_3d:
                    fig.add_trace(
                        go.Scatter3d(
                            x=[],
                            y=[],
                            z=[],
                            mode="lines+markers",
                            name=trace_name,
                            line=dict(color=line_color, dash=line_style),
                            marker=dict(size=3),
                            hovertemplate=(
                                f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                                f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                                f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                                f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<br>{z_pc}: %{{z}}<extra></extra>"
                            ),
                            customdata=[],
                            showlegend=show_in_legend,
                            legendgroup=legend_group,
                        )
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=[],
                            y=[],
                            mode="lines+markers",
                            name=trace_name,
                            line=dict(color=line_color, dash=line_style),
                            marker=dict(size=6),
                            hovertemplate=(
                                f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                                f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                                f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                                f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<extra></extra>"
                            ),
                            customdata=[],
                            showlegend=show_in_legend,
                            legendgroup=legend_group,
                        )
                    )
            continue  # Skip adding start/end markers for extension runs in initial trace

        if is_3d:
            # 3D scatter plot (always lines+markers)
            fig.add_trace(
                go.Scatter3d(
                    x=model_data[x_pc],
                    y=model_data[y_pc],
                    z=model_data[z_pc],
                    mode="lines+markers",
                    name=trace_name,
                    line=dict(color=line_color, dash=line_style),
                    marker=dict(size=3),
                    hovertemplate=(
                        f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                        f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                        f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                        f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<br>{z_pc}: %{{z}}<extra></extra>"
                    ),
                    customdata=model_data["checkpoint"],
                    showlegend=show_in_legend,
                    legendgroup=legend_group,
                )
            )

            # Add start marker (only for non-extension runs)
            if not is_extended_run:
                fig.add_trace(
                    go.Scatter3d(
                        x=[model_data[x_pc].iloc[0]],
                        y=[model_data[y_pc].iloc[0]],
                        z=[model_data[z_pc].iloc[0]],
                        mode="markers",
                        marker=dict(symbol="circle", size=6, color="black"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        else:
            # 2D scatter plot (always lines+markers)
            fig.add_trace(
                go.Scatter(
                    x=model_data[x_pc],
                    y=model_data[y_pc],
                    mode="lines+markers",
                    name=trace_name,
                    line=dict(color=line_color, dash=line_style),
                    marker=dict(size=6),
                    hovertemplate=(
                        f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                        f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                        f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                        f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<extra></extra>"
                    ),
                    customdata=model_data["checkpoint"],
                    showlegend=show_in_legend,
                    legendgroup=legend_group,
                )
            )

            # Add start marker (only for non-extension runs)
            if not is_extended_run:
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
            aspectmode="cube",  # Force cubic aspect ratio
            aspectratio=dict(x=1, y=1, z=1),  # Ensure equal scaling
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0)  # Fixed camera position  # Center view
            ),
        ),
        legend_title="KL Weight",
        legend=dict(font=dict(size=16), x=1, xanchor="right", y=4, yanchor="top"),
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
        legend=dict(font=dict(size=16), x=1, xanchor="right", y=1, yanchor="top"),
        template="plotly_white",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        plot_bgcolor="#e3eef7",
        paper_bgcolor="#e3eef7",
    )


# Create frames for animation
frames = []

# Determine progress range based on whether extended runs are included
if extended_models and any(extended_models):
    progress_range = range(0, 201, 2)  # Go to 200% for extended runs
else:
    progress_range = range(0, 101, 2)  # Go to 100% for regular runs

for progress in progress_range:  # Use 2% steps for smoother animation
    frame_traces = []

    # Get the maximum checkpoint for main runs (before extension offset)
    max_main_checkpoint = filtered_df["checkpoint"].max() if not filtered_df.empty else 100

    # For each model, determine what data to include based on progress
    for model in plot_df["model"].unique():
        # Check if this is an extended run
        is_extended_run = model in extended_models if extended_models else False

        if is_extended_run:
            # For extension runs, handle visibility based on progress
            if progress < 100:
                # For progress < 100%, include extension runs with empty data to clear them
                model_data_temp = plot_df[plot_df["model"] == model]
                if not isinstance(model_data_temp, pd.DataFrame):
                    model_data_temp = pd.DataFrame(model_data_temp)
                kl_weight = model_data_temp["KL_weight"].iloc[0] if not model_data_temp.empty else "0"

                # Determine if this is a no-KL or full-KL extension
                is_no_kl_extension = False
                is_full_kl_extension = False
                model_config = MODELS.get(model)
                if model_config and model_config.get("kl_weight") == 0:
                    is_no_kl_extension = True
                elif model_config:
                    is_full_kl_extension = True

                base_kl = kl_weight
                model_suffix = model.split(f"KL{kl_weight}")[-1] if f"KL{kl_weight}" in model else ""
                if model_suffix and model_suffix.strip():
                    trace_name = f"additional_train_{base_kl}{model_suffix}"
                else:
                    trace_name = f"KL={base_kl}"

                # Add extension indicator to legend
                ext_label = get_extension_label(is_no_kl_extension, is_full_kl_extension)
                if ext_label:
                    trace_name += f" ({ext_label})"

                # Set legend group for extensions
                base_model = None
                for base_model_name in associated_runs:
                    if (
                        associated_runs[base_model_name]["no_kl"] == model
                        or associated_runs[base_model_name]["full_kl"] == model
                    ):
                        base_model = base_model_name
                        break
                legend_group = f"group_{base_model}" if base_model else f"group_{model}"

                # Assign color and line style
                if is_no_kl_extension:
                    line_color = "red"
                    line_style = "dash"
                else:
                    line_color = "green"
                    line_style = "dash"

                # Add empty trace to clear extension runs
                if is_3d:
                    frame_traces.append(
                        go.Scatter3d(
                            x=[],
                            y=[],
                            z=[],
                            mode="lines+markers",
                            name=trace_name,
                            line=dict(color=line_color, dash=line_style),
                            marker=dict(size=3),
                            hovertemplate=(
                                f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                                f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                                f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                                f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<br>{z_pc}: %{{z}}<extra></extra>"
                            ),
                            customdata=[],
                            showlegend=False,  # Extension runs don't show in legend
                            legendgroup=legend_group,
                        )
                    )
                else:
                    frame_traces.append(
                        go.Scatter(
                            x=[],
                            y=[],
                            mode="lines+markers",
                            name=trace_name,
                            line=dict(color=line_color, dash=line_style),
                            marker=dict(size=6),
                            hovertemplate=(
                                f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                                f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                                f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                                f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<extra></extra>"
                            ),
                            customdata=[],
                            showlegend=False,  # Extension runs don't show in legend
                            legendgroup=legend_group,
                        )
                    )
                continue
            else:
                # For progress >= 100%, calculate extension progress
                extension_progress = progress - 100  # 0-100% for extensions

                # Get extension data and calculate the threshold correctly
                model_data = plot_df[plot_df["model"] == model]
                if not model_data.empty:
                    # Extension data already has checkpoints offset by max_main_checkpoint
                    # So we need to calculate the threshold relative to the extension data
                    extension_max_checkpoint = model_data["checkpoint"].max() - max_main_checkpoint
                    extension_threshold = extension_max_checkpoint * (extension_progress / 100)

                    # Filter to show extension data up to the threshold
                    model_data = model_data[model_data["checkpoint"] <= (max_main_checkpoint + extension_threshold)]
                else:
                    model_data = pd.DataFrame()
        else:
            # For main runs, show them normally up to the current progress
            frame_threshold = max_main_checkpoint * (progress / 100)
            model_data = plot_df[plot_df["model"] == model]
            model_data = model_data[model_data["checkpoint"] <= frame_threshold]

        if not isinstance(model_data, pd.DataFrame):
            model_data = pd.DataFrame(model_data)
        if not model_data.empty:
            kl_weight = model_data["KL_weight"].iloc[0]

            # Determine if this is a no-KL or full-KL extension
            is_no_kl_extension = False
            is_full_kl_extension = False
            if is_extended_run:
                model_config = MODELS.get(model)
                if model_config and model_config.get("kl_weight") == 0:
                    is_no_kl_extension = True
                elif model_config:
                    is_full_kl_extension = True

            base_kl = kl_weight
            model_suffix = model.split(f"KL{kl_weight}")[-1] if f"KL{kl_weight}" in model else ""
            if model_suffix and model_suffix.strip():
                trace_name = f"additional_train_{base_kl}{model_suffix}"
            else:
                trace_name = f"KL={base_kl}"

            # For extension runs, don't add to legend - they will be handled by base run
            show_in_legend = not is_extended_run

            # For main runs, check if they have extensions selected and modify legend name
            if not is_extended_run:
                # Check if this main run has extensions selected
                if model in associated_runs:
                    ext_label = get_extension_label(
                        selected_0kl_ext.get(kl_weight, False), selected_full_kl_ext.get(kl_weight, False)
                    )
                    if ext_label:
                        trace_name += f" ({ext_label})"

            # Add extension indicator to legend for extension runs (though they won't show in legend)
            ext_label = get_extension_label(is_no_kl_extension, is_full_kl_extension)
            if ext_label:
                trace_name += f" ({ext_label})"

            # Set legend group for main runs and their extensions
            if is_extended_run:
                # Find the base model for this extension
                base_model = None
                for base_model_name in associated_runs:
                    if (
                        associated_runs[base_model_name]["no_kl"] == model
                        or associated_runs[base_model_name]["full_kl"] == model
                    ):
                        base_model = base_model_name
                        break
                legend_group = f"group_{base_model}" if base_model else f"group_{model}"
            else:
                legend_group = f"group_{model}"

            # Assign color and line style
            if is_extended_run:
                if is_no_kl_extension:
                    # Red dashed line for no-KL extension
                    line_color = "red"
                    line_style = "dash"
                else:
                    # Green dashed line for full-KL extension
                    line_color = "green"
                    line_style = "dash"
            else:
                # Regular color for main runs
                if kl_weight in all_kl_weights:
                    color_idx = all_kl_weights.index(kl_weight) % len(color_palette)
                    line_color = color_palette[color_idx]
                else:
                    line_color = color_palette[0]  # fallback color
                line_style = "solid"
            model_data = model_data.sort_values("checkpoint").reset_index(drop=True)
            if is_3d:
                if len(model_data) > 1:
                    frame_traces.append(
                        go.Scatter3d(
                            x=model_data[x_pc],
                            y=model_data[y_pc],
                            z=model_data[z_pc],
                            mode="lines+markers",
                            name=trace_name,
                            line=dict(color=line_color, dash=line_style),
                            marker=dict(size=3),
                            hovertemplate=(
                                f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                                f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                                f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                                f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<br>{z_pc}: %{{z}}<extra></extra>"
                            ),
                            customdata=model_data["checkpoint"],
                            showlegend=show_in_legend,
                            legendgroup=legend_group,
                        )
                    )
                else:
                    frame_traces.append(
                        go.Scatter3d(
                            x=model_data[x_pc],
                            y=model_data[y_pc],
                            z=model_data[z_pc],
                            mode="markers",
                            name=trace_name,
                            marker=dict(size=3, color=line_color),
                            hovertemplate=(
                                f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                                f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                                f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                                f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<br>{z_pc}: %{{z}}<extra></extra>"
                            ),
                            customdata=model_data["checkpoint"],
                            showlegend=show_in_legend,
                            legendgroup=legend_group,
                        )
                    )
                # Add start marker if at least one point (only for non-extension runs)
                if len(model_data) > 0 and not is_extended_run:
                    frame_traces.append(
                        go.Scatter3d(
                            x=[model_data[x_pc].iloc[0]],
                            y=[model_data[y_pc].iloc[0]],
                            z=[model_data[z_pc].iloc[0]],
                            mode="markers",
                            marker=dict(symbol="circle", size=6, color="black"),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            else:
                if len(model_data) > 1:
                    frame_traces.append(
                        go.Scatter(
                            x=model_data[x_pc],
                            y=model_data[y_pc],
                            mode="lines+markers",
                            name=trace_name,
                            line=dict(color=line_color, dash=line_style),
                            marker=dict(size=6),
                            hovertemplate=(
                                f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                                f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                                f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                                f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<extra></extra>"
                            ),
                            customdata=model_data["checkpoint"],
                            showlegend=show_in_legend,
                            legendgroup=legend_group,
                        )
                    )
                else:
                    frame_traces.append(
                        go.Scatter(
                            x=model_data[x_pc],
                            y=model_data[y_pc],
                            mode="markers",
                            name=trace_name,
                            marker=dict(size=6, color=line_color),
                            hovertemplate=(
                                f"Model: {model}<br>KL: {MODELS[model]['kl_weight']}<br>Checkpoint: %{{customdata}}<br>"
                                f"End of Run General Misalignment: {MODELS[model]['general_misalignment_percent']}%<br>"
                                f"End of Run Narrow Misalignment: {MODELS[model]['narrow_misalignment_percent']}%<br>"
                                f"{x_pc}: %{{x}}<br>{y_pc}: %{{y}}<extra></extra>"
                            ),
                            customdata=model_data["checkpoint"],
                            showlegend=show_in_legend,
                            legendgroup=legend_group,
                        )
                    )
                # Add start marker if at least one point (only for non-extension runs)
                if len(model_data) > 0 and not is_extended_run:
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

    frames.append(go.Frame(data=frame_traces, name=str(progress)))

fig.frames = frames

# Determine starting position for slider based on whether extension runs are selected
if extended_models and any(extended_models):
    slider_start_index = 100  # Start at 200% when extension runs are selected
else:
    slider_start_index = 50  # Start at 100% when only main runs are selected

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
            aspectmode="cube",  # Force cubic aspect ratio
            aspectratio=dict(x=1, y=1, z=1),  # Ensure equal scaling
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5), center=dict(x=0, y=0, z=0)  # Fixed camera position  # Center view
            ),
        ),
        legend_title="KL Weight",
        legend=dict(font=dict(size=16), x=1.02, xanchor="right", y=0.97, yanchor="top"),
        template="plotly_white",
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg,
        margin=dict(t=0, b=20, l=20, r=20),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": -0.03,
                "xanchor": "left",
                "y": -0.05,
                "yanchor": "top",
                "font": {"size": 14, "family": "DejaVu Sans"},
                "buttons": updatemenus_buttons,
                "bgcolor": "#ffffff",
                "bordercolor": "#cccccc",
                "borderwidth": 1,
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
                    for progress in progress_range
                ],
                "active": slider_start_index,  # Start at 200% if extensions selected, 100% otherwise
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
        legend=dict(font=dict(size=16), x=0.98, xanchor="right", y=0.98, yanchor="top"),
        template="plotly_white",
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": -0.05,
                "xanchor": "left",
                "y": -0.25,
                "yanchor": "top",
                "font": {"size": 14, "family": "DejaVu Sans"},
                "buttons": updatemenus_buttons,
                "bgcolor": "#ffffff",
                "bordercolor": "#cccccc",
                "borderwidth": 1,
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
                    for progress in progress_range
                ],
                "active": slider_start_index,  # Start at 200% if extensions selected, 100% otherwise
                "currentvalue": {"prefix": "Training Progress: "},
                "len": 0.9,
                "x": 0.1,
                "xanchor": "left",
                "y": -0.15,
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
st.markdown("---")
st.markdown("&nbsp;")
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

st.markdown(
    '<div style="font-size:smaller; color: #888; margin-top:2em;"><sup>1</sup> We directly append the steering vector '
    "to layer 24 of the residual stream, thus it is literally an activation addition.</div>",
    unsafe_allow_html=True,
)
