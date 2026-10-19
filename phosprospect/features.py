"""Geochemical feature engineering for sedimentary phosphorite prospectivity.

Every feature is dimensionless (a ratio, anomaly or enrichment factor) so that samples
from different basins, detrital backgrounds and laboratories remain comparable.
"""

import numpy as np
import pandas as pd

from .reference import (
    CHONDRITIC_Y_HO,
    FRANCOLITE_CAO_P2O5,
    OXIDE_TO_ELEMENT,
    PAAS_AL_PPM,
    PAAS_REE_PPM,
    PAAS_TRACE_PPM,
    REE_ALL,
)


def _safe_div(a, b):
    """Element-wise division returning NaN where the denominator is zero or missing."""
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.where((b != 0) & b.notna())


def _col(df, name):
    """Return a column as floats, or an all-NaN series when absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def to_element_ppm(df):
    """Convert major-oxide wt% columns to element ppm, passing trace columns through."""
    out = df.copy()
    for oxide, (element, factor) in OXIDE_TO_ELEMENT.items():
        if oxide in df.columns:
            out[element] = pd.to_numeric(df[oxide], errors="coerce") * factor * 10_000.0
    return out


def paas_normalise_ree(df):
    """Return REE concentrations divided by PAAS (subscript N values)."""
    el = to_element_ppm(df)
    out = pd.DataFrame(index=df.index)
    for ree, paas in PAAS_REE_PPM.items():
        if ree in el.columns:
            out[f"{ree}_N"] = _col(el, ree) / paas
    return out


def ree_systematics(df):
    """REE shape descriptors: anomalies, fractionation slopes, MREE bulge, Y/Ho."""
    el = to_element_ppm(df)
    n = paas_normalise_ree(df)
    f = pd.DataFrame(index=df.index)

    present = [r for r in REE_ALL if r in el.columns]
    if present:
        f["sum_REE"] = el[present].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)

    def N(x):
        return n[f"{x}_N"] if f"{x}_N" in n.columns else pd.Series(np.nan, index=df.index)

    # Ce/Ce* after Bau & Dulski (1996): a negative anomaly records oxic bottom water,
    # because Ce(III) oxidises to insoluble Ce(IV) and is stripped from seawater.
    f["Ce_anomaly"] = _safe_div(N("Ce"), 0.5 * N("La") + 0.5 * N("Pr"))
    f["Eu_anomaly"] = _safe_div(N("Eu"), np.sqrt(N("Sm") * N("Gd")))
    f["La_Yb_N"] = _safe_div(N("La"), N("Yb"))
    f["La_Sm_N"] = _safe_div(N("La"), N("Sm"))
    f["Gd_Yb_N"] = _safe_div(N("Gd"), N("Yb"))

    # Authigenic apatite grows a middle-REE "bulge"; detrital-dominated samples stay flat.
    mree = (N("Sm") + N("Gd") + N("Dy")) / 3.0
    ends = (N("La") + N("Yb")) / 2.0
    f["MREE_bulge"] = _safe_div(mree, ends)

    # Seawater Y/Ho is superchondritic; detrital input pulls it back toward ~28.
    f["Y_Ho"] = _safe_div(_col(el, "Y"), _col(el, "Ho"))
    f["Y_Ho_excess"] = f["Y_Ho"] / CHONDRITIC_Y_HO
    return f


def redox_proxies(df):
    """Trace-element ratios recording bottom-water and porewater oxygenation."""
    el = to_element_ppm(df)
    f = pd.DataFrame(index=df.index)
    f["V_Cr"] = _safe_div(_col(el, "V"), _col(el, "Cr"))
    f["U_Th"] = _safe_div(_col(el, "U"), _col(el, "Th"))
    f["Ni_Co"] = _safe_div(_col(el, "Ni"), _col(el, "Co"))
    f["V_V_Ni"] = _safe_div(_col(el, "V"), _col(el, "V") + _col(el, "Ni"))

    # Authigenic uranium, Wignall & Myers (1988): total U minus the detrital share
    # implied by Th. Positive values mean U was fixed from porewater under reducing
    # conditions, which is where phosphogenesis operates.
    f["U_authigenic"] = _col(el, "U") - _col(el, "Th") / 3.0
    return f


def enrichment_factors(df, elements=None):
    """Al-normalised enrichment against PAAS; EF > 1 means addition beyond detritus."""
    el = to_element_ppm(df)
    al = _col(el, "Al")
    if elements is None:
        elements = [e for e in PAAS_TRACE_PPM if e in el.columns]
    f = pd.DataFrame(index=df.index)
    for e in elements:
        sample_ratio = _safe_div(_col(el, e), al)
        paas_ratio = PAAS_TRACE_PPM[e] / PAAS_AL_PPM
        f[f"EF_{e}"] = sample_ratio / paas_ratio
    return f


def productivity_proxies(df):
    """Proxies for the organic-carbon flux that drives phosphogenesis."""
    el = to_element_ppm(df)
    al = _col(el, "Al")
    f = pd.DataFrame(index=df.index)
    f["Ba_Al"] = _safe_div(_col(el, "Ba"), al)
    f["Cd_Al"] = _safe_div(_col(el, "Cd"), al)
    f["Cu_Al"] = _safe_div(_col(el, "Cu"), al)
    f["Ni_Al"] = _safe_div(_col(el, "Ni"), al)

    # Biogenic (excess) barium: total Ba less the fraction explicable as detrital.
    f["Ba_excess"] = _col(el, "Ba") - al * (PAAS_TRACE_PPM["Ba"] / PAAS_AL_PPM)
    return f


def apatite_chemistry(df):
    """Francolite stoichiometry and substitution indicators."""
    el = to_element_ppm(df)
    f = pd.DataFrame(index=df.index)
    f["CaO_P2O5"] = _safe_div(_col(df, "CaO"), _col(df, "P2O5"))

    # Departure above francolite stoichiometry implies calcite dilution rather than
    # higher apatite content, so grade and this ratio must be read together.
    f["carbonate_excess"] = f["CaO_P2O5"] - FRANCOLITE_CAO_P2O5
    f["Sr_Ca"] = _safe_div(_col(el, "Sr"), _col(el, "Ca"))
    f["F_P2O5"] = _safe_div(_col(df, "F"), _col(df, "P2O5"))
    f["P_Al"] = _safe_div(_col(el, "P"), _col(el, "Al"))
    return f


def detrital_proxies(df):
    """Indicators of clastic dilution and provenance."""
    el = to_element_ppm(df)
    al = _col(el, "Al")
    f = pd.DataFrame(index=df.index)
    f["Si_Al"] = _safe_div(_col(el, "Si"), al)
    f["Ti_Al"] = _safe_div(_col(el, "Ti"), al)
    f["Zr_Al"] = _safe_div(_col(el, "Zr"), al)
    f["K_Al"] = _safe_div(_col(el, "K"), al)
    return f


def clr_transform(df, columns=None, pseudocount=1e-6):
    """Centred log-ratio transform to remove closure before modelling."""
    cols = columns or list(df.columns)
    sub = df[cols].apply(pd.to_numeric, errors="coerce").clip(lower=pseudocount)
    log_sub = np.log(sub)
    return log_sub.sub(log_sub.mean(axis=1), axis=0).add_suffix("_clr")


FEATURE_BLOCKS = {
    "ree": ree_systematics,
    "redox": redox_proxies,
    "productivity": productivity_proxies,
    "apatite": apatite_chemistry,
    "detrital": detrital_proxies,
    "enrichment": enrichment_factors,
}


def build_features(df, blocks=None):
    """Assemble the full dimensionless feature table from raw assay data."""
    chosen = blocks or list(FEATURE_BLOCKS)
    parts = [FEATURE_BLOCKS[b](df) for b in chosen]
    out = pd.concat(parts, axis=1)
    return out.replace([np.inf, -np.inf], np.nan)
