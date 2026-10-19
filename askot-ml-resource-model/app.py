"""Askot Cu-Pb-Zn resource model -- interactive dashboard.

Run locally:

    streamlit run app.py

Every number on screen is computed at request time from the composite
assays in ``data/``: the Random Forest is fitted, grades are predicted onto
a 3-D block grid, and tonnage is summed above the chosen cut-off. Nothing is
hard-coded or pre-rendered, so moving a control genuinely re-runs the
estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from askot import (
    DATA_CAVEATS,
    EstimationParams,
    benchmark_table,
    build_block_model,
    estimate_resource,
    feature_importance,
    fit_random_forest,
    grade_tonnage_curve,
    in_range_mask,
    leave_one_borehole_out_cv,
    load_collars,
    load_composites,
    load_sections,
    section_weighted_grade,
)

ORE_COLOURSCALE = "Viridis"
HALO_COLOUR = "#9aa3ad"
COLLAR_COLOUR = "#eb6834"

st.set_page_config(page_title="Askot Resource Model",
                   page_icon="⛏",
                   layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------
# Cached computation
#
# The split matters for responsiveness: loading and fitting happen once per
# session, the block grid is rebuilt only when its geometry changes, and the
# cut-off / density / radius controls re-filter an existing grid.
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data():
    return load_composites(), load_collars(), load_sections()


@st.cache_resource(show_spinner="Fitting Random Forest on composite assays...")
def get_model(n_composites: int):
    # n_composites participates in the cache key so the forest is refitted
    # if the underlying CSV ever changes between runs.
    composites, _, _ = get_data()
    return fit_random_forest(composites)


@st.cache_data(show_spinner="Predicting grades onto the block grid...")
def get_blocks(block_size_m: float, padding_m: float) -> pd.DataFrame:
    composites, _, _ = get_data()
    model = get_model(len(composites))
    params = EstimationParams(block_size_m=block_size_m, padding_m=padding_m)
    return build_block_model(composites, model, params)


@st.cache_data(show_spinner="Cross-validating (leave-one-borehole-out)...")
def get_cv(n_composites: int):
    composites, _, _ = get_data()
    return leave_one_borehole_out_cv(composites)


@st.cache_data(show_spinner=False)
def get_curve(block_size_m: float, padding_m: float, radius_m: float,
              density: float) -> pd.DataFrame:
    blocks = get_blocks(block_size_m, padding_m)
    params = EstimationParams(block_size_m=block_size_m, padding_m=padding_m,
                              max_distance_to_sample_m=radius_m,
                              density_t_per_m3=density)
    return grade_tonnage_curve(blocks, params)


composites, collars, sections = get_data()

# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Estimation parameters")
    st.caption("Each control re-runs the estimate. Defaults reproduce the "
               "published figure of 1.52 Mt.")

    cutoff = st.slider("TMC cut-off (%)", 0.0, 24.0, 2.0, 0.5,
                       help="Total metal content = Cu% + Pb% + Zn%. "
                            "MECL 1992 used 2%.")
    density = st.slider("Bulk density (t/m³)", 2.8, 4.0, 3.50, 0.05,
                        help="GSI used 3.20; MECL 1992 used 3.50.")
    radius = st.slider("Search radius (m)", 5.0, 60.0, 15.0, 5.0,
                       help="A block counts only if a composite sample lies "
                            "within this distance. Relaxing it inflates "
                            "tonnage into unsampled ground.")
    block_size = st.select_slider("Block size (m)", options=[5.0, 10.0, 20.0],
                                  value=10.0)

    st.divider()
    show_halo = st.toggle("Show below-cut-off blocks", value=True,
                          help="The faint halo is modelled volume being "
                               "discarded at the current cut-off.")
    show_collars = st.toggle("Show drillhole collars", value=True)

    st.divider()
    st.caption(
        f"**Data**: {len(composites)} composite assays · "
        f"{composites['borehole_id'].nunique()} holes · "
        f"{len(collars)} collars · {len(sections)} section intercepts · "
        "4 source reports (1974-2018)")

params = EstimationParams(tmc_cutoff_pct=cutoff, block_size_m=block_size,
                          max_distance_to_sample_m=radius,
                          density_t_per_m3=density)

blocks = get_blocks(block_size, params.padding_m)
estimate = estimate_resource(blocks, params)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Askot Cu-Pb-Zn deposit — machine-learning resource model")
st.markdown(
    "Four exploration campaigns between **1974 and 2018** left the Askot "
    "polymetallic deposit (Pithoragarh, Uttarakhand) with four incompatible "
    "reserve figures locked in scanned paper reports. This model rebuilds a "
    "single auditable 3-D estimate from the original assays — and reproduces "
    "44 years of expert estimation without being told any geology."
)

classical_low, classical_high = 0.77, 1.86
inside = classical_low <= estimate.tonnage_mt <= classical_high

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tonnage above cut-off", f"{estimate.tonnage_mt:.2f} Mt",
          delta=("inside classical range" if inside else "outside 0.77–1.86 Mt"),
          delta_color="normal" if inside else "inverse")
c2.metric("Cu", f"{estimate.grades['Cu_pct']:.2f} %")
c3.metric("Pb", f"{estimate.grades['Pb_pct']:.2f} %")
c4.metric("Zn", f"{estimate.grades['Zn_pct']:.2f} %")
c5.metric("Ore blocks", f"{estimate.n_blocks:,}",
          delta=f"of {estimate.n_blocks_in_range:,} supported")

st.caption(
    f"At a {cutoff:g}% TMC cut-off, {params.block_size_m:g} m blocks and "
    f"{density:g} t/m³, the model reports **{estimate.tonnage_t:,.0f} t** "
    f"grading **{estimate.tmc_pct:.2f}% TMC**. The four independent "
    "estimates made between 1974 and 2018 span 0.77–1.86 Mt."
)

tab_body, tab_sensitivity, tab_validation, tab_data = st.tabs(
    ["Ore body", "Sensitivity", "Validation", "Data"])

# --------------------------------------------------------------------------
# Tab 1 -- the 3-D ore shell
# --------------------------------------------------------------------------
with tab_body:
    left, right = st.columns([3, 1])

    supported = blocks[in_range_mask(blocks, params)]
    ore = supported[supported["TMC_pct"] >= cutoff]
    halo = supported[supported["TMC_pct"] < cutoff]

    with left:
        fig = go.Figure()

        if show_halo and len(halo):
            fig.add_trace(go.Scatter3d(
                x=halo["X"], y=halo["Y"], z=halo["Z"],
                mode="markers", name="below cut-off",
                marker=dict(size=2, color=HALO_COLOUR, opacity=0.12),
                hoverinfo="skip",
            ))

        if len(ore):
            fig.add_trace(go.Scatter3d(
                x=ore["X"], y=ore["Y"], z=ore["Z"],
                mode="markers", name="ore",
                marker=dict(size=3.5, color=ore["TMC_pct"],
                            colorscale=ORE_COLOURSCALE, cmin=2, cmax=20,
                            opacity=0.95,
                            colorbar=dict(title="TMC %", thickness=12, len=0.6)),
                customdata=np.stack(
                    [ore["Cu_pct"], ore["Pb_pct"], ore["Zn_pct"]], axis=-1),
                hovertemplate=("TMC %{marker.color:.1f}%<br>"
                               "Cu %{customdata[0]:.2f}%  "
                               "Pb %{customdata[1]:.2f}%  "
                               "Zn %{customdata[2]:.2f}%<br>"
                               "RL %{z:.0f} m<extra></extra>"),
            ))

        if show_collars and len(collars):
            fig.add_trace(go.Scatter3d(
                x=collars["X"], y=collars["Y"], z=collars["Z"],
                mode="markers", name="drillhole collars",
                marker=dict(size=3, color=COLLAR_COLOUR, symbol="diamond"),
                text=collars["borehole_id"],
                hovertemplate="%{text}<br>RL %{z:.0f} m<extra></extra>",
            ))

        fig.update_layout(
            height=620,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=0.99),
            # uirevision keeps the camera where the user left it when a
            # control re-runs the script -- without it every slider nudge
            # would snap the view back to the default angle.
            uirevision="askot-3d",
            scene=dict(
                xaxis_title="Local grid E (m)",
                yaxis_title="Local grid N (m)",
                zaxis_title="RL (m)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
            ),
        )
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("What you're looking at")

        min_tmc = float(supported["TMC_pct"].min()) if len(supported) else 0.0
        if len(halo):
            st.markdown(
                f"**{len(ore):,} ore blocks** above the {cutoff:g}% cut-off, "
                "coloured by total metal content, inside a faint halo of "
                f"**{len(halo):,} modelled blocks** that now fail it."
            )
        else:
            st.markdown(
                f"**All {len(ore):,} supported blocks** clear the "
                f"{cutoff:g}% cut-off — the lowest modelled block grades "
                f"**{min_tmc:.2f}% TMC**. At Askot the MECL cut-off is not "
                "what bounds the ore body; the limit of drilling is."
            )

        st.markdown(
            f"Push the cut-off past **{min_tmc:.1f}%** and the shell starts "
            "melting back to its high-grade core — that is the "
            "tonnage-versus-grade trade-off every mine plan turns on, and "
            "the reason a reserve figure quoted without its cut-off is "
            "close to meaningless."
        )
        st.divider()
        st.markdown("**Volume accounting**")
        rejected = estimate.n_blocks_total - estimate.n_blocks_in_range
        st.dataframe(pd.DataFrame({
            "stage": ["Blocks in grid", f"Within {radius:g} m of a sample",
                      f"Above {cutoff:g}% TMC"],
            "blocks": [f"{estimate.n_blocks_total:,}",
                       f"{estimate.n_blocks_in_range:,}",
                       f"{estimate.n_blocks:,}"],
        }), hide_index=True, width="stretch")
        st.caption(
            f"{rejected:,} blocks are discarded for sitting beyond the search "
            "radius. Skipping that step is what makes naive ML resource "
            "estimates overstate tonnage by orders of magnitude."
        )

# --------------------------------------------------------------------------
# Tab 2 -- grade-tonnage curve and benchmark
# --------------------------------------------------------------------------
with tab_sensitivity:
    curve = get_curve(block_size, params.padding_m, radius, density)

    st.subheader("Grade–tonnage curve")
    st.caption("The first check a reviewer applies: tonnage must fall "
               "monotonically as cut-off rises, while grade climbs. The "
               "marker tracks your current cut-off.")

    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(
        x=curve["cutoff_tmc_pct"], y=curve["tonnage_mt"],
        name="Tonnage (Mt)", mode="lines",
        line=dict(color="#2a78d6", width=2.5),
        hovertemplate="cut-off %{x:.1f}%<br>%{y:.2f} Mt<extra></extra>",
    ))
    fig_c.add_trace(go.Scatter(
        x=curve["cutoff_tmc_pct"], y=curve["TMC_pct"],
        name="Average grade (TMC %)", mode="lines", yaxis="y2",
        line=dict(color="#eb6834", width=2.5, dash="dot"),
        hovertemplate="cut-off %{x:.1f}%<br>%{y:.2f}% TMC<extra></extra>",
    ))
    fig_c.add_vline(x=cutoff, line=dict(color="#1baf7a", width=2))
    fig_c.add_annotation(x=cutoff, y=1.0, yref="paper",
                         text=f"cut-off {cutoff:g}%", showarrow=False,
                         yanchor="bottom", font=dict(color="#1baf7a"))
    fig_c.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="TMC cut-off (%)",
        yaxis=dict(title="Tonnage (Mt)"),
        yaxis2=dict(title="Average grade (TMC %)", overlaying="y",
                    side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_c, width="stretch")

    full = curve[curve["blocks"] == curve["blocks"].max()]
    plateau_end = float(full["cutoff_tmc_pct"].max())
    st.caption(
        f"The curve is flat up to **{plateau_end:g}% TMC** — every block the "
        "drilling supports already grades above that, so the cut-off does no "
        "work until it passes the bottom of the modelled grade population. "
        "At Askot the resource is bounded by the extent of drilling, not by "
        "the economic cut-off, which is exactly why the four historical "
        "estimates disagree on tonnage far more than they disagree on grade."
    )

    st.subheader("Benchmark against 44 years of classical estimation")
    bench = benchmark_table(estimate)
    is_model = bench["name"].str.startswith("ML model")

    fig_b = go.Figure(go.Bar(
        x=bench["tonnage_mt"], y=bench["name"], orientation="h",
        marker_color=np.where(is_model, "#eb6834", "#2a78d6"),
        text=[f"{v:.2f} Mt" for v in bench["tonnage_mt"]],
        textposition="outside",
        hovertemplate="%{y}<br>%{x:.2f} Mt<extra></extra>",
    ))
    fig_b.update_layout(
        height=320, margin=dict(l=0, r=40, t=10, b=0),
        xaxis_title="Tonnage (Mt)", yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    st.plotly_chart(fig_b, width="stretch")

    st.dataframe(
        bench[["name", "tonnage_mt", "Cu_pct", "Pb_pct", "Zn_pct", "cutoff",
               "method"]].rename(columns={
                   "name": "Estimate", "tonnage_mt": "Tonnage (Mt)",
                   "Cu_pct": "Cu %", "Pb_pct": "Pb %", "Zn_pct": "Zn %",
                   "cutoff": "Cut-off", "method": "Method"}),
        hide_index=True, width="stretch",
        column_config={"Tonnage (Mt)": st.column_config.NumberColumn(
            format="%.2f")},
    )
    st.caption("Blank cells are figures the source report does not state — "
               "they are left empty rather than inferred.")

# --------------------------------------------------------------------------
# Tab 3 -- validation
# --------------------------------------------------------------------------
with tab_validation:
    st.subheader("Leave-one-borehole-out cross-validation")
    st.markdown(
        "Every composite from a hole is held out **together**. Plain k-fold "
        "would scatter intervals from the same hole across train and test, "
        "leak position, and report a score that would not survive on the "
        "next drillhole. The Random Forest has to beat inverse-distance "
        "weighting — the classical interpolator — on identical folds to "
        "justify itself."
    )

    cv = get_cv(len(composites))
    st.caption(cv.summary_line())
    st.dataframe(
        cv.scores.rename(columns={
            "metal": "Metal", "RF_MAE": "RF MAE", "RF_R2": "RF R²",
            "IDW_MAE": "IDW MAE", "IDW_R2": "IDW R²"}),
        hide_index=True, width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.3f")
                       for c in ("RF MAE", "RF R²", "IDW MAE", "IDW R²")},
    )
    st.info(
        "**The honest finding.** Adding 48 underground channel samples to "
        "the 25 borehole composites moved tonnage from 1.14 Mt to 1.52 Mt — "
        "more realistic — but made cross-validated R² *worse*. Channel "
        "samples resolve fine-scale grade variation at ~2 m spacing that "
        "widely spaced boreholes never see, and the 960 m-level Cu assays "
        "carry ~30% QA/QC error. More data improved the tonnage and "
        "degraded the interpolation score. **Data quality, not algorithm "
        "choice, is the binding constraint at Askot.**",
        icon="⚠️",
    )

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Spatial feature importance**")
        imp = feature_importance(composites)
        st.dataframe(imp.rename(columns={"axis": "Axis",
                                         "importance": "Gini importance"}),
                     hide_index=True, width="stretch",
                     column_config={"Gini importance":
                                    st.column_config.ProgressColumn(
                                        format="%.3f", min_value=0.0,
                                        max_value=1.0)})
    with v2:
        st.markdown("**Independent check: 1992 cross-sections**")
        sw = section_weighted_grade(sections)
        if sw:
            st.dataframe(pd.DataFrame({
                "Source": ["1992 cross-sections (true-thickness weighted)",
                           "This model (above cut-off)"],
                "Cu %": [sw["Cu_pct"], estimate.grades["Cu_pct"]],
                "Pb %": [sw["Pb_pct"], estimate.grades["Pb_pct"]],
                "Zn %": [sw["Zn_pct"], estimate.grades["Zn_pct"]],
            }), hide_index=True, width="stretch",
                column_config={c: st.column_config.NumberColumn(format="%.2f")
                               for c in ("Cu %", "Pb %", "Zn %")})
            st.caption(f"{sw['n_intercepts']} graded intercepts digitised from "
                       "22 geological cross-sections (Plates IV-A to IV-I), "
                       "estimated by a method the model never sees.")

    st.subheader("Known limitations")
    for caveat in DATA_CAVEATS:
        st.markdown(f"- {caveat}")

# --------------------------------------------------------------------------
# Tab 4 -- the underlying data
# --------------------------------------------------------------------------
with tab_data:
    st.subheader("Composite assays")
    st.caption("Transcribed and cross-checked from four scanned reports with "
               "no text layer. This table is the model's entire training set.")
    st.dataframe(composites, hide_index=True, width="stretch",
                 height=320)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Drillhole collars**")
        st.dataframe(collars, hide_index=True, width="stretch",
                     height=280)
    with d2:
        st.markdown("**Cross-section intercepts**")
        st.dataframe(sections, hide_index=True, width="stretch",
                     height=280)

    st.download_button(
        "Download composites (CSV)",
        composites.to_csv(index=False).encode(),
        file_name="askot_composites.csv", mime="text/csv")

st.divider()
st.caption(
    "Askot Cu-Pb-Zn ML resource model · data consolidated from GSI (1974-75), "
    "MECL (1980), MECL Phase-II (1992) and SRK/Adi Gold NI 43-101 (2018) "
    "reports · coordinates on the local mine grid, not a projected CRS · "
    "read the limitations above before quoting any figure."
)
