from pathlib import Path

import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import streamlit as st

from background_utils import get_background_css, get_narrow_content_css
from data.emergent_misalignment.em_utils import get_model_colors
from data.emergent_misalignment.get_pca import get_pca_plot_df
from data.emergent_misalignment.model_dict import MODELS

st.markdown(get_background_css(), unsafe_allow_html=True)
st.markdown(get_narrow_content_css(), unsafe_allow_html=True)

st.title("😈 Emergent Misalignment")

st.markdown(
    """

### Corresponding Work
- **[Narrow Misalignment is Hard, Emergent Misalignment is Easy](https://arxiv.org/pdf/2506.11613)**


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
weights being equivalent to activations<sup>1</sup>. This allows us to visualise highly
interpretable training traces.


The below lets you pick different KL penalisation training runs. As discussed in the
[corresponding work](https://arxiv.org/pdf/2506.11613), the KL penalisation directly controls
learning the generally or narrowly misaligned solution<sup>2</sup>. The animation shows the training
trajectory of the steering vector in the latent space.


The visualisation has a few steps:

1. **Pick the KL weights**: This lets you pick which KL weights to show.
2. **Pick the extensions**: This lets you pick which extensions to show (none, 0 KL, 1e6 KL or both).
3. **Pick the PC axes**: This lets you pick the PC axes to show (try out the 3D option!).
4. **Play/Pause**: This lets you play/pause the animation (you can also drag the slider to change the progress).


""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="font-size:smaller; color: #888; margin-top:2em;"><sup>1</sup> We directly append the steering vector '
    "to layer 24 of the residual stream, thus it is literally an activation addition.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="font-size:smaller; color: #888; margin-top:1em;"><sup>2</sup> We find KL ≥ 5e5 is where narrow '
    "misalignemnt is learnt over general.</div>",
    unsafe_allow_html=True,
)


# --- PCA Trajectory Plot ---
st.markdown("---")

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
st.markdown(
    '<div style="margin-top: -10px;"><span style="color: #888888; font-style: italic; font-size: 14.5px;">'
    "(Reg: regular training, + KL: include second run with 1e6 KL penalty<sup> 3</sup>, "
    "0 KL: include second run without regularising, Both: both second runs)</span></div>",
    unsafe_allow_html=True,
)

num_weights = len(all_kl_weights)
cols = st.columns(num_weights)

for i, kl_weight in enumerate(all_kl_weights):
    with cols[i]:
        # Single multi-selector for each weight
        default_value = "Reg" if kl_weight in default_kl_weights else "Off"

        option = st.selectbox(
            f"{kl_weight}",
            ["Off", "Reg", "+ KL", "0 KL", "Both"],
            key=f"weight_{kl_weight}",
            index=["Off", "Reg", "+ KL", "0 KL", "Both"].index(default_value),
        )

        # Handle the selection
        if option != "Off":
            selected_kl_weights.append(kl_weight)

            if option == "+ KL":
                selected_full_kl_ext[kl_weight] = True
            elif option == "0 KL":
                selected_0kl_ext[kl_weight] = True
            elif option == "Both":
                selected_full_kl_ext[kl_weight] = True
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

# Camera tracking for 3D plots
if z_pc != "None":
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

    # Initialize camera position in session state
    if "current_camera" not in st.session_state:
        st.session_state.current_camera = dict(x=1.5, y=1.5, z=1.5)

    # Set up camera presets for compatibility
    camera_presets = {"Current View": st.session_state.current_camera}
    camera_angle = "Current View"


def pc_label(pc):
    var = pc_var_dict.get(pc, 0)
    return f"{pc} ({var:.1f}% var)"


def get_extension_label(is_no_kl, is_full_kl):
    """Generate extension label based on boolean flags"""
    if is_no_kl and not is_full_kl:
        return 'Ext: <span style="color: red">0 KL</span>'
    elif is_full_kl and not is_no_kl:
        return 'Ext: <span style="color: green">+ KL</span>'
    elif is_no_kl and is_full_kl:
        return 'Ext: <span style="color: red">0 KL</span>, <span style="color: green">+ KL</span>'
    return ""


# Ensure plot_df is a DataFrame
if not isinstance(plot_df, pd.DataFrame):
    plot_df = pd.DataFrame(plot_df)

# Get color palette from utils
model_colors = get_model_colors()

# --- Checkpoint Freeze Slider ---
col1, col2 = st.columns([3, 1])
with col1:
    st.header("Inside the Training")
with col2:
    # Only show button for 3D plots
    if z_pc != "None":
        st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

        # Camera capture button
        if st.button("📸 Use Current View", key="capture_camera", help="Capture the current 3D view for animation"):
            # Get live camera position from URL parameters (continuously updated by tracking JavaScript)
            live_camera = st.session_state.current_camera.copy()  # Default fallback
            try:
                if hasattr(st, "query_params"):
                    params = st.query_params
                    if "live_cam_x" in params and "live_cam_y" in params and "live_cam_z" in params:
                        live_camera = {
                            "x": float(params["live_cam_x"]),
                            "y": float(params["live_cam_y"]),
                            "z": float(params["live_cam_z"]),
                        }
            except Exception:
                pass

            # Store the live camera position
            st.session_state.current_camera = live_camera
            st.session_state.show_success_message = True
            st.session_state.success_message = (
                f"✅ Camera view captured! Position: "
                f"({live_camera['x']:.2f}, {live_camera['y']:.2f}, {live_camera['z']:.2f})"
            )


# Calculate the maximum checkpoint across all selected models
max_checkpoint = plot_df["checkpoint"].max() if not plot_df.empty else 100

# Calculate axis ranges from the full dataset (100%) for consistent scaling
full_df = plot_df.copy()
if not isinstance(full_df, pd.DataFrame):
    full_df = pd.DataFrame(full_df)

# Get the full range for axes with padding
x_min, x_max = full_df[x_pc].min(), full_df[x_pc].max()
y_min, y_max = full_df[y_pc].min(), full_df[y_pc].max()

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
else:
    # For 2D plots, apply the SAME logic as 3D to make it exactly square
    # Calculate the maximum range across both axes for square aspect ratio
    x_range_size = x_max - x_min
    y_range_size = y_max - y_min
    max_range_size = max(x_range_size, y_range_size)

    # Add 10% padding to ensure all points are visible
    padded_range_size = max_range_size * 1.1

    # Center each axis around its midpoint with the maximum range
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Apply the padded maximum range to both axes for square aspect ratio
    x_range = [x_center - padded_range_size / 2, x_center + padded_range_size / 2]
    y_range = [y_center - padded_range_size / 2, y_center + padded_range_size / 2]


# Animation speed (fixed at 5 FPS)
animation_speed = 5

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
            line_color = model_colors.get(model, "#1f77b4")  # fallback to default blue
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


# Layout will be configured later with animation controls to avoid conflicts


# Create frames for animation
frames = []

# Determine progress range based on whether extended runs are included
# Use larger steps for better performance
if extended_models and any(extended_models):
    progress_range = range(0, 201, 5)  # Go to 200% for extended runs, 5% steps
else:
    progress_range = range(0, 101, 5)  # Go to 100% for regular runs, 5% steps

for progress in progress_range:  # Use 5% steps for better performance
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
                line_color = model_colors.get(model, "#1f77b4")  # fallback to default blue
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

    # Create frame with layout to preserve camera position for 3D plots
    if is_3d and st.session_state.current_camera is not None:
        # Set camera position in frame to prevent animation from resetting it
        frame_layout = dict(scene=dict(camera=dict(eye=st.session_state.current_camera, center=dict(x=0, y=0, z=0))))
        frames.append(go.Frame(data=frame_traces, layout=frame_layout, name=str(progress)))
    else:
        frames.append(go.Frame(data=frame_traces, name=str(progress)))

fig.frames = frames

# Determine starting position for slider based on whether extension runs are selected
if extended_models and any(extended_models):
    slider_start_index = 40  # Start at 200% when extension runs are selected (40 * 5% = 200%)
else:
    slider_start_index = 20  # Start at 100% when only main runs are selected (20 * 5% = 100%)

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
    # Configure 3D scene based on captured camera position
    if st.session_state.current_camera is not None:
        # Use the captured camera position
        scene_config = dict(
            xaxis_title=pc_label(x_pc),
            yaxis_title=pc_label(y_pc),
            zaxis_title=pc_label(z_pc),
            xaxis=dict(range=x_range, showgrid=True, gridwidth=2),
            yaxis=dict(range=y_range, showgrid=True, gridwidth=2),
            zaxis=dict(range=z_range, showgrid=True, gridwidth=2),
            aspectmode="cube",  # Force cubic aspect ratio
            aspectratio=dict(x=1, y=1, z=1),  # Ensure equal scaling
            camera=dict(eye=st.session_state.current_camera, center=dict(x=0, y=0, z=0)),
        )
    else:
        # Interactive mode - no fixed camera position
        scene_config = dict(
            xaxis_title=pc_label(x_pc),
            yaxis_title=pc_label(y_pc),
            zaxis_title=pc_label(z_pc),
            xaxis=dict(range=x_range, showgrid=True, gridwidth=2),
            yaxis=dict(range=y_range, showgrid=True, gridwidth=2),
            zaxis=dict(range=z_range, showgrid=True, gridwidth=2),
            aspectmode="cube",  # Force cubic aspect ratio
            aspectratio=dict(x=1, y=1, z=1),  # Ensure equal scaling
        )

    fig.update_layout(
        height=800,
        scene=scene_config,
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
        xaxis=dict(range=x_range, showgrid=True, gridwidth=2, minor=dict(showgrid=True, gridwidth=1)),
        yaxis=dict(
            range=y_range,
            showgrid=True,
            gridwidth=2,
            minor=dict(showgrid=True, gridwidth=1),
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor=plot_bg,
        paper_bgcolor=plot_bg,
        margin=dict(t=20, b=80, l=60, r=20),
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

# No additional JavaScript needed - using live coordinates from tracking

# Display the animated plot
st.plotly_chart(fig, use_container_width=True)

if z_pc != "None":
    st.markdown('<div style="margin-top: 10px; margin-bottom: 10px;"></div>', unsafe_allow_html=True)

    # JavaScript bridge for camera tracking
    import streamlit.components.v1 as components

    # JavaScript component that tracks camera position silently
    camera_tracker_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <script>
        let trackingInterval;
        let currentCameraPosition = null;

        function startTracking() {
            // Access the parent window (Streamlit)
            const parentWindow = window.parent;

            function findPlotlyPlot() {
                try {
                    const plots = parentWindow.document.querySelectorAll('.js-plotly-plot');
                    return plots.length > 0 ? plots[plots.length - 1] : null;
                } catch (e) {
                    console.log('Cannot access parent plots:', e);
                    return null;
                }
            }

            function getCameraPosition() {
                const plot = findPlotlyPlot();
                if (plot) {
                    // Try multiple ways to get current camera position
                    let camera = null;

                    // Method 1: Try _fullLayout (Plotly's internal state)
                    if (plot._fullLayout && plot._fullLayout.scene && plot._fullLayout.scene.camera) {
                        camera = plot._fullLayout.scene.camera.eye;
                    }

                    // Method 2: Try accessing the current camera state directly
                    if (!camera && plot.layout && plot.layout.scene && plot.layout.scene.camera) {
                        camera = plot.layout.scene.camera.eye;
                    }

                    return camera;
                }
                return null;
            }

            // Set up Plotly event listeners for camera changes
            function setupPlotlyEvents() {
                const plot = findPlotlyPlot();
                if (plot) {
                    // Listen for camera changes
                    plot.on('plotly_relayout', function(eventData) {
                        if (eventData && eventData['scene.camera']) {
                            setTimeout(updateCamera, 100);
                        }
                    });

                    console.log('Camera tracking active');
                }
            }

            function updateCamera() {
                const camera = getCameraPosition();
                if (camera && camera.x !== undefined && camera.y !== undefined && camera.z !== undefined) {
                    currentCameraPosition = {
                        x: parseFloat(camera.x.toFixed(3)),
                        y: parseFloat(camera.y.toFixed(3)),
                        z: parseFloat(camera.z.toFixed(3))
                    };

                    // Store in sessionStorage for capture button
                    parentWindow.sessionStorage.setItem(
                        'current_camera_position',
                        JSON.stringify(currentCameraPosition)
                    );

                    // ALSO update URL parameters for real-time display
                    try {
                        const url = new URL(parentWindow.location.href);
                        url.searchParams.set('live_cam_x', currentCameraPosition.x);
                        url.searchParams.set('live_cam_y', currentCameraPosition.y);
                        url.searchParams.set('live_cam_z', currentCameraPosition.z);
                        parentWindow.history.replaceState({}, '', url);
                    } catch (e) {
                        console.log('Cannot update URL for live display:', e);
                    }
                }
            }

            // Start monitoring with events + polling
            setupPlotlyEvents();

            if (trackingInterval) clearInterval(trackingInterval);
            trackingInterval = setInterval(updateCamera, 1000);

            console.log('Camera tracking started');
        }

        // Start after a short delay to ensure everything is loaded
        setTimeout(startTracking, 2000);
        </script>
    </head>
    <body>
        <div style="display: none;">Camera tracking active</div>
    </body>
    </html>
    """

    # Render the tracking component (hidden)
    components.html(camera_tracker_html, height=0)

    # Read live camera position from URL parameters
    live_camera = st.session_state.current_camera.copy()  # Default to session state
    try:
        if hasattr(st, "query_params"):
            params = st.query_params
            if "live_cam_x" in params and "live_cam_y" in params and "live_cam_z" in params:
                live_camera = {
                    "x": float(params["live_cam_x"]),
                    "y": float(params["live_cam_y"]),
                    "z": float(params["live_cam_z"]),
                }
    except Exception:
        pass

# Animation logic (auto-advance if playing)
if st.session_state.get("animation_playing", False):
    import time

    st.session_state["animation_progress"] = min(100, st.session_state["animation_progress"] + 2)
    time.sleep(0.125)  # 8 FPS
    st.rerun()


st.markdown(
    '<div style="font-size:smaller; color: #888; margin-top:1em;"><sup>3</sup> We use a 1e6 KL divergence penalty for '
    "the second run since we find this to be the optimal penalisation to induce narrow misalignment.</div>",
    unsafe_allow_html=True,
)

# --- Training Details ---
st.markdown("---")
st.markdown(
    """
### Learning Rate Schedule

The steering vector training uses a **warmup followed by linear decay** schedule:

- **Warmup Steps**: 5 steps (starting from 0, reaching peak at step 5)
- **Peak Learning Rate**: 1e-4 (reached at step 5)
- **Final Learning Rate**: 0 (reached at step 800)
- **Decay Schedule**: Linear decay from peak to zero after warmup
- **Total Steps**: Varies by model (typically around 800 steps)

The learning rate starts at 0, warms up linearly to 1e-4 over the first 5 steps, then decreases linearly to 0
over the remaining training steps.This gradual reduction is standard and helps the model converge more stably.
This is why you observe the change in speed in visualizations above.

### Additional Training Configuration

Several other aspects of the training configuration are noteworthy:

#### Checkpointing & Data
- **Save Steps**: 5 (checkpoints saved every 5 steps)
- **Epochs**: 2 (short training duration with frequent monitoring)
- **Training Dataset Size**: 6000 (all examples are bad medical advice)
- **Regularization Dataset Size**: 1000 (all examples are non-medical domain, mixture of good and bad advice)

The extremely frequent checkpointing (every 5 steps) enables the detailed trajectory analysis you see in the
visualizations above. This granular tracking is essential for understanding how steering vectors evolve during training.

#### Optimization Details
- **Optimizer**: AdamW 8-bit (memory-efficient optimization)
- **Weight Decay**: 0.0 (no L2 regularization)
- **Batch Configuration**:
  - Per-device train batch size: 2
  - Gradient accumulation steps: 8
  - Effective batch size: 16

"""
)

# Contact information
st.markdown("---")
st.markdown(
    """
### Links

For associated models and code:
- **Hugging Face**: [@EdwardTurner](https://huggingface.co/EdwardTurner)
- **GitHub**: [@edwardbturner](https://github.com/edwardbturner)

For any feedback, bugs, or questions about this visualization:
- **Email**: [edward.turner01@outlook.com](mailto:edward.turner01@outlook.com)
- **Twitter**: [@EdTurner42](https://twitter.com/EdTurner42)
""",
    unsafe_allow_html=True,
)
