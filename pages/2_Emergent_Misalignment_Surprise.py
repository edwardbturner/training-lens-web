from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import orjson
import plotly.graph_objects as go
import streamlit as st  # type: ignore

from background_utils import get_background_css, get_narrow_content_css

st.set_page_config(page_title="Emergent Misalignment Surprise", layout="wide")
st.markdown(get_background_css(), unsafe_allow_html=True)
st.markdown(get_narrow_content_css(), unsafe_allow_html=True)

st.title("😲 Emergent Misalignment Surprise")
st.markdown(
    """
This page visualizes *surprise* (in nats, -log(prob)) for each model type (base, narrow, general) on the combined dataset.\
Choose between summary statistics or individual data visualizations.
"""
)
st.markdown("---")

DATA_PATH = Path("data/emergent_misalignment_surprise/combined_probabilities.json")
SAMPLE_SIZE_OPTIONS = [10_000, 50_000]
BASE_PERCENTILE_OPTIONS = [25, 50, 75, 100]

with open(DATA_PATH, "rb") as f:
    raw = orjson.loads(f.read())
    total_records = raw["metadata"]["num_records"]
    all_data = raw["data"]
    model_types = raw["metadata"]["model_types"]

# Precompute all combinations of (sample size, base percentile)
precomputed_data = {}
for sample_size in SAMPLE_SIZE_OPTIONS:
    # Sample without replacement
    if len(all_data) > sample_size:
        idx = np.random.choice(len(all_data), sample_size, replace=False)
        sample = [all_data[i] for i in idx]
    else:
        sample = all_data
    base_probs = np.array([rec.get("base_prob", 0) for rec in sample if rec.get("base_prob") is not None])
    for base_percentile in BASE_PERCENTILE_OPTIONS:
        base_thresh = np.percentile(base_probs, base_percentile)
        filtered = [rec for rec in sample if rec.get("base_prob") is not None and rec["base_prob"] <= base_thresh]
        precomputed_data[(sample_size, base_percentile)] = filtered

# UI: Dropdowns for sample size and base percentile
sample_size = st.selectbox(
    "Sample size (number of records to use for plotting)", SAMPLE_SIZE_OPTIONS, index=0, format_func=lambda x: f"{x:,}"
)
base_percentile = st.selectbox(
    "Base percentile (filter tokens with base probability ≤ this percentile)",
    BASE_PERCENTILE_OPTIONS,
    index=3,
)

filtered_data = precomputed_data[(sample_size, base_percentile)]
percent_used = 100 * sample_size / total_records
st.caption(f"Using {sample_size:,} out of {total_records:,} records ({percent_used:.2f}%) for all plots.")


# --- User Mode Selection ---
mode = st.radio("Choose visualization mode:", ["Summary", "Individual"], horizontal=True)


# --- Helper: Extract nats for each model type ---
def extract_nats(data: List[Dict[str, Any]], model_types: List[str]) -> Dict[str, List[float]]:
    nats = {mt: [] for mt in model_types}
    for rec in data:
        for mt in model_types:
            prob = rec.get(f"{mt}_prob")
            if prob is not None and prob > 0:
                nats[mt].append(-np.log(prob))
    return nats


# --- SUMMARY MODE ---
if mode == "Summary":
    st.header("Summary: Mean Surprise per Model Type")
    nats = extract_nats(filtered_data, model_types)
    means = [np.mean(nats[mt]) for mt in model_types]
    stds = [np.std(nats[mt]) for mt in model_types]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=model_types,
            y=means,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color=["#e41a1c", "#377eb8", "#4daf4a"],
            opacity=0.8,
            name="Mean nats",
        )
    )
    fig.update_layout(
        yaxis_title="Mean nats (-log(prob))",
        xaxis_title="Model Type",
        title="Mean Surprise (nats) per Model Type",
        plot_bgcolor="#dbe7f0",
        paper_bgcolor="#dbe7f0",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Error bars show ±1 std. Only a sample of data is used for speed.")

# --- INDIVIDUAL MODE ---
if mode == "Individual":
    indiv_mode = st.radio("Individual plot type:", ["Histogram", "Scatterplot"], horizontal=True)
    nats = extract_nats(filtered_data, model_types)
    if indiv_mode == "Histogram":
        st.header("Histogram: Surprise Distribution per Model Type")
        fig = go.Figure()
        colors = {"base": "#e41a1c", "general": "#377eb8", "narrow": "#4daf4a"}
        bin_count = 50
        for mt in model_types:
            nats_arr = np.array(nats[mt])
            if len(nats_arr) == 0:
                continue
            counts, bin_edges = np.histogram(nats_arr, bins=bin_count)
            bin_examples = [[] for _ in range(bin_count)]
            for idx, rec in enumerate(filtered_data):
                prob = rec.get(f"{mt}_prob")
                if prob is not None and prob > 0:
                    nat = -np.log(prob)
                    bin_idx = np.searchsorted(bin_edges, nat, side="right") - 1
                    if 0 <= bin_idx < bin_count and len(bin_examples[bin_idx]) < 10:
                        pre = rec.get("pre_context", [])
                        pred = rec.get("predicted_token", "")
                        pre_str = "..." + "".join(pre[-10:])
                        bin_examples[bin_idx].append(f"(pre-context: {pre_str}, token: {pred})")
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            customdata = []
            for i in range(bin_count):
                exs = bin_examples[i]
                if exs:
                    joined = "<br>".join(exs)
                    customdata.append(f"<b>Examples:</b><br>{joined}")
                else:
                    customdata.append("No examples in this bin.")
            bin_indices = np.clip(np.searchsorted(bin_edges, nats_arr, side="right") - 1, 0, bin_count - 1)
            fig.add_trace(
                go.Histogram(
                    x=nats_arr,
                    name=mt,
                    opacity=0.5,
                    marker_color=colors.get(mt, None),
                    nbinsx=bin_count,
                    customdata=np.array(customdata)[bin_indices][:, None],
                    hovertemplate="Count: %{y}<br>Surprise (nats): %{x:.2f}<br>%{customdata[0]}",
                )
            )
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Surprise (nats)",
            yaxis_title="Count",
            title="Surprise Distribution per Model Type",
            plot_bgcolor="#dbe7f0",
            paper_bgcolor="#dbe7f0",
            height=500,
        )
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Histogram overlays for all model types. Hover a bin to see up to 10 example contexts.")
    elif indiv_mode == "Scatterplot":
        # --- SCATTERPLOT LOGIC (as previously implemented, robust) ---
        # --- Dropdowns for axis selection ---
        axis_options = ["None"] + model_types
        col1, col2, col3 = st.columns(3)
        x_axis = col1.selectbox("X Axis", axis_options, index=1)
        y_axis = col2.selectbox("Y Axis", axis_options, index=2)
        z_axis = col3.selectbox("Z Axis", axis_options, index=3)

        show_ref = st.checkbox("Show y=x line (or x=y=z plane in 3D)", value=True)

        axes = [(x_axis, "x"), (y_axis, "y"), (z_axis, "z")]
        selected = [(m, ax) for m, ax in axes if m != "None"]

        if len(selected) < 2:
            st.info("Please select at least two axes (model types) to plot.")
            st.stop()

        # --- Prepare data for plotting ---
        def get_nats(record, model):
            prob = record.get(f"{model}_prob")
            if prob is not None and prob > 0:
                return -np.log(prob)
            return None

        def get_color_and_size(narrow, general):
            diff = narrow - general if (narrow is not None and general is not None) else 0.0
            color_val = np.sign(diff) * np.sqrt(abs(diff)) if diff is not None else 0.0
            size = 6 * np.sqrt(abs(diff)) if diff != 0 else 4
            return diff, size, color_val

        plot_data = {ax: [] for _, ax in selected}
        color_vals = []
        size_vals = []
        color_sqrt_vals = []
        hover_texts = []
        for rec in filtered_data:
            vals = [get_nats(rec, m) for m, ax in selected]
            narrow_nats = get_nats(rec, "narrow")
            general_nats = get_nats(rec, "general")
            base_nats = get_nats(rec, "base")
            if all(v is not None for v in vals) and (narrow_nats is not None and general_nats is not None):
                for (m, ax), v in zip(selected, vals):
                    plot_data[ax].append(v)
                diff, size, color_val = get_color_and_size(narrow_nats, general_nats)
                color_vals.append(diff)
                size_vals.append(size)
                color_sqrt_vals.append(color_val)
                hover_texts.append(
                    f"sample_index: {rec.get('sample_index', '')}<br>"
                    f"position: {rec.get('position', '')}<br>"
                    f"predicted_token: {rec.get('predicted_token', '')}<br>"
                    f"nats(base): {base_nats:.3f}<br>"
                    f"nats(general): {general_nats:.3f}<br>"
                    f"nats(narrow): {narrow_nats:.3f}"
                )

        # --- Plotting ---
        plot_bg = "#dbe7f0"
        if len(selected) == 2:
            if not plot_data["x"] or not plot_data["y"]:
                st.info("No data to display for the selected axes/models.")
                st.stop()
            x_vals = np.array(plot_data["x"])
            y_vals = np.array(plot_data["y"])
            minv = min(x_vals.min(), y_vals.min())
            maxv = max(x_vals.max(), y_vals.max())
            cmin = -max(abs(np.min(color_sqrt_vals)), abs(np.max(color_sqrt_vals)))
            cmax = max(abs(np.min(color_sqrt_vals)), abs(np.max(color_sqrt_vals)))
            tickvals = [cmin, 0, cmax]
            fig = go.Figure(
                data=go.Scattergl(
                    x=plot_data["x"],
                    y=plot_data["y"],
                    mode="markers",
                    marker=dict(
                        size=size_vals,
                        color=color_sqrt_vals,
                        colorscale="RdBu",
                        cmin=cmin,
                        cmax=cmax,
                        colorbar=dict(
                            title="sign(nats(narrow)-nats(general)) * sqrt(|nats(narrow)-nats(general)|)",
                            tickvals=tickvals,
                            ticks="outside",
                            orientation="h",
                            x=0.5,
                            y=-0.25,
                            xanchor="center",
                            yanchor="bottom",
                            len=0.6,
                            thickness=18,
                            tickfont=dict(size=14),
                        ),
                        showscale=True,
                        opacity=1.0,
                        line=dict(width=0),
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                )
            )
            if show_ref:
                fig.add_trace(
                    go.Scattergl(
                        x=[minv, maxv],
                        y=[minv, maxv],
                        mode="lines",
                        line=dict(color="black", dash="dash"),
                        name="y = x",
                        showlegend=True,
                    )
                )
            fig.update_layout(
                xaxis_title=f"{selected[0][0]} surprise (nats)",
                yaxis_title=f"{selected[1][0]} surprise (nats)",
                title="2D Surprise Scatterplot",
                height=600,
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
                xaxis=dict(
                    scaleanchor="y",
                    scaleratio=1,
                    range=[minv, maxv],
                ),
                yaxis=dict(
                    scaleanchor="x",
                    scaleratio=1,
                    range=[minv, maxv],
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
        elif len(selected) == 3:
            if not plot_data["x"] or not plot_data["y"] or not plot_data["z"]:
                st.info("No data to display for the selected axes/models.")
                st.stop()
            x_vals = np.array(plot_data["x"])
            y_vals = np.array(plot_data["y"])
            z_vals = np.array(plot_data["z"])
            minv = min(x_vals.min(), y_vals.min(), z_vals.min())
            maxv = max(x_vals.max(), y_vals.max(), z_vals.max())
            cmin = -max(abs(np.min(color_sqrt_vals)), abs(np.max(color_sqrt_vals)))
            cmax = max(abs(np.min(color_sqrt_vals)), abs(np.max(color_sqrt_vals)))
            tickvals = [cmin, 0, cmax]
            fig = go.Figure(
                data=go.Scatter3d(
                    x=plot_data["x"],
                    y=plot_data["y"],
                    z=plot_data["z"],
                    mode="markers",
                    marker=dict(
                        size=[max(2, s * 0.7) for s in size_vals],
                        color=color_sqrt_vals,
                        colorscale="RdBu",
                        cmin=cmin,
                        cmax=cmax,
                        colorbar=dict(
                            title="sign(nats(narrow)-nats(general)) * sqrt(|nats(narrow)-nats(general)|)",
                            tickvals=tickvals,
                            ticks="outside",
                            orientation="h",
                            x=0.5,
                            y=-0.25,
                            xanchor="center",
                            yanchor="bottom",
                            len=0.6,
                            thickness=18,
                            tickfont=dict(size=14),
                        ),
                        showscale=True,
                        opacity=1.0,
                        line=dict(width=0),
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                )
            )
            if show_ref:
                # Add y=x, y=z, x=z planes
                minv = min(np.min(plot_data["x"]), np.min(plot_data["y"]), np.min(plot_data["z"]))
                maxv = max(np.max(plot_data["x"]), np.max(plot_data["y"]), np.max(plot_data["z"]))
                plane_vals = np.linspace(minv, maxv, 20)
                # y = x plane (base = general)
                X, Z = np.meshgrid(plane_vals, plane_vals)
                fig.add_trace(
                    go.Surface(
                        x=X,
                        y=X,
                        z=Z,
                        showscale=False,
                        opacity=0.15,
                        colorscale=[[0, "red"], [1, "red"]],
                        name="base = general (y = x)",
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )
                # y = z plane (general = narrow)
                X, Z = np.meshgrid(plane_vals, plane_vals)
                fig.add_trace(
                    go.Surface(
                        x=X,
                        y=Z,
                        z=Z,
                        showscale=False,
                        opacity=0.15,
                        colorscale=[[0, "green"], [1, "green"]],
                        name="general = narrow (y = z)",
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )
                # x = z plane (base = narrow)
                X, Y = np.meshgrid(plane_vals, plane_vals)
                fig.add_trace(
                    go.Surface(
                        x=X,
                        y=Y,
                        z=X,
                        showscale=False,
                        opacity=0.15,
                        colorscale=[[0, "blue"], [1, "blue"]],
                        name="base = narrow (x = z)",
                        hoverinfo="skip",
                        showlegend=True,
                    )
                )
            fig.update_layout(
                scene=dict(
                    xaxis_title=f"{selected[0][0]} surprise (nats)",
                    yaxis_title=f"{selected[1][0]} surprise (nats)",
                    zaxis_title=f"{selected[2][0]} surprise (nats)",
                    aspectmode="cube",
                    xaxis=dict(range=[minv, maxv]),
                    yaxis=dict(range=[minv, maxv]),
                    zaxis=dict(range=[minv, maxv]),
                ),
                title="3D Surprise Scatterplot",
                height=700,
                plot_bgcolor=plot_bg,
                paper_bgcolor=plot_bg,
            )
            st.plotly_chart(fig, use_container_width=True)
            # Custom legend for plane colors
            st.markdown(
                '<div style="display: flex; gap: 2em; align-items: center; margin-top: 0.5em; margin-bottom: 1em;">'
                '  <span style="display: flex; align-items: center;">'
                '    <span style="background: red; width: 1.5em; height: 1.5em; display: inline-block; '
                'margin-right: 0.5em; border-radius: 0.2em;"></span>'
                "    base = general (y = x)"
                "  </span>"
                '  <span style="display: flex; align-items: center;">'
                '    <span style="background: green; width: 1.5em; height: 1.5em; display: inline-block; '
                'margin-right: 0.5em; border-radius: 0.2em;"></span>'
                "    general = narrow (y = z)"
                "  </span>"
                '  <span style="display: flex; align-items: center;">'
                '    <span style="background: blue; width: 1.5em; height: 1.5em; display: inline-block; '
                'margin-right: 0.5em; border-radius: 0.2em;"></span>'
                "    base = narrow (x = z)"
                "  </span>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Please select at least two axes (model types) to plot.")
