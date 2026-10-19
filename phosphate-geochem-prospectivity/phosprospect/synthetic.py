"""Synthetic drillhole assays for a fictitious sedimentary phosphorite basin.

Nothing here is real data. Each sample is a linear mix of three end-members
(authigenic francolite, detrital shale, marine limestone) whose proportions follow
smooth spatial fields, so neighbouring holes look alike - the property that makes
random cross-validation over-optimistic. Compositions are loosely based on published
ranges for marine phosphorites and PAAS-like shale; they are illustrative only.
"""

import numpy as np
import pandas as pd

from .reference import REE_ALL

MAJORS = ["P2O5", "CaO", "SiO2", "Al2O3", "Fe2O3", "TiO2", "K2O", "MgO", "F"]
TRACES = ["V", "Cr", "Ni", "Co", "Cu", "Zn", "Mo", "U", "Th", "Ba", "Sr", "Zr",
          "Cd", "Y"]
ANALYTES = MAJORS + TRACES + REE_ALL

# End-member compositions: majors in wt%, everything else in ppm.
_FRANCOLITE = {
    "P2O5": 34.0, "CaO": 50.5, "SiO2": 2.0, "Al2O3": 0.3, "Fe2O3": 0.5, "TiO2": 0.02,
    "K2O": 0.05, "MgO": 0.3, "F": 3.6,
    "V": 260, "Cr": 60, "Ni": 70, "Co": 4, "Cu": 60, "Zn": 180, "Mo": 22, "U": 85,
    "Th": 1.5, "Ba": 250, "Sr": 2300, "Zr": 15, "Cd": 12, "Y": 290,
    # MREE-enriched, negative Ce anomaly, superchondritic Y/Ho (seawater inheritance).
    "La": 140, "Ce": 95, "Pr": 31, "Nd": 140, "Sm": 34, "Eu": 8.0, "Gd": 42, "Tb": 6.0,
    "Dy": 36, "Ho": 6.4, "Er": 17, "Tm": 2.2, "Yb": 12.5, "Lu": 1.8,
}
_SHALE = {
    "P2O5": 0.16, "CaO": 1.3, "SiO2": 62.0, "Al2O3": 18.9, "Fe2O3": 6.5, "TiO2": 1.0,
    "K2O": 3.7, "MgO": 2.2, "F": 0.05,
    "V": 150, "Cr": 110, "Ni": 55, "Co": 23, "Cu": 50, "Zn": 85, "Mo": 1.0, "U": 3.1,
    "Th": 14.6, "Ba": 650, "Sr": 200, "Zr": 210, "Cd": 0.1, "Y": 27,
    "La": 38.2, "Ce": 79.6, "Pr": 8.83, "Nd": 33.9, "Sm": 5.55, "Eu": 1.08, "Gd": 4.66,
    "Tb": 0.774, "Dy": 4.68, "Ho": 0.991, "Er": 2.85, "Tm": 0.405, "Yb": 2.82, "Lu": 0.433,
}
_LIMESTONE = {
    "P2O5": 0.08, "CaO": 53.0, "SiO2": 2.5, "Al2O3": 0.5, "Fe2O3": 0.3, "TiO2": 0.03,
    "K2O": 0.1, "MgO": 1.2, "F": 0.03,
    "V": 12, "Cr": 8, "Ni": 10, "Co": 2, "Cu": 6, "Zn": 20, "Mo": 0.4, "U": 2.0,
    "Th": 0.6, "Ba": 40, "Sr": 600, "Zr": 8, "Cd": 0.2, "Y": 6,
    "La": 4.5, "Ce": 5.0, "Pr": 0.9, "Nd": 3.8, "Sm": 0.75, "Eu": 0.16, "Gd": 0.85,
    "Tb": 0.13, "Dy": 0.8, "Ho": 0.18, "Er": 0.5, "Tm": 0.07, "Yb": 0.45, "Lu": 0.07,
}
END_MEMBERS = pd.DataFrame([_FRANCOLITE, _SHALE, _LIMESTONE],
                           index=["francolite", "shale", "limestone"])[ANALYTES]


def _smooth_field(x, y, rng, n_modes=12, wavelength=(1500.0, 5000.0)):
    """Cheap stationary random field: a sum of random plane waves, scaled to unit sd."""
    z = np.zeros_like(x, dtype=float)
    for _ in range(n_modes):
        lam = rng.uniform(*wavelength)
        theta = rng.uniform(0, np.pi)
        phase = rng.uniform(0, 2 * np.pi)
        k = 2 * np.pi / lam
        z += np.cos(k * (x * np.cos(theta) + y * np.sin(theta)) + phase)
    return (z - z.mean()) / z.std()


def _log_lithology(f_phos, f_carb, cutoff, rng, mislog_rate=0.05):
    """Turn true end-member fractions into a geologist's lithology log, with errors.

    ``cutoff`` is the francolite fraction at which the logger calls a rock phosphatic.
    It varies between areas, as it does when different geologists log different blocks.
    """
    names = []
    for fp, fc, cut in zip(f_phos, f_carb, cutoff):
        host = "limestone" if fc > 0.5 else "carbonaceous shale"
        if fp >= 0.40:
            name = "Phosphorite, pelletal"
        elif fp >= cut:
            name = f"Phosphatic {host}"
        else:
            name = host.capitalize()
        # Logging error: phosphate pellets missed in dark shale, or over-called.
        if rng.random() < mislog_rate:
            name = host.capitalize() if "hosph" in name else f"Phosphatic {host}"
        names.append(name)
    return names


def make_basin(nx=24, ny=16, spacing=250.0, samples_per_hole=4, seed=42):
    """Generate a synthetic drillhole assay table.

    Returns one row per sample with hole_id, easting/northing (local metres), depth,
    logged lithology, the true francolite fraction (for checking only, never a model
    input) and assays for every analyte in ``ANALYTES``.
    """
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(nx) * spacing, np.arange(ny) * spacing)
    hx = gx.ravel() + rng.normal(0, spacing * 0.15, gx.size)
    hy = gy.ravel() + rng.normal(0, spacing * 0.15, gy.size)

    # Phosphogenesis favours a shoaling belt running SW-NE plus local highs.
    width = nx * spacing
    belt = np.exp(-((hy - 0.35 * hx - 0.25 * width * 0.6) / 900.0) ** 2)
    phos_field = 2.2 * belt + 0.8 * _smooth_field(hx, hy, rng) - 2.3
    carb_field = 0.9 * _smooth_field(hx, hy, rng, wavelength=(3000.0, 8000.0))
    # Provenance drift in the detrital end-member (heavier minerals to the east).
    prov_field = 0.25 * _smooth_field(hx, hy, rng, wavelength=(4000.0, 9000.0))
    # Sub-basin differences in apatite chemistry (REE, U and Sr uptake).
    apatite_field = 0.30 * _smooth_field(hx, hy, rng, wavelength=(2500.0, 6000.0))
    # Logging convention drifts between areas: phosphatic called at 10-30 % francolite.
    logger_cut = 0.20 + 0.07 * np.tanh(_smooth_field(hx, hy, rng, wavelength=(2000.0, 5000.0)))

    rows = []
    for h in range(hx.size):
        for s in range(samples_per_hole):
            depth = 20.0 + s * 6.0 + rng.uniform(0, 3)
            fp = 1 / (1 + np.exp(-(phos_field[h] + rng.normal(0, 0.7))))
            fp = float(np.clip(fp * 0.85, 0.0, 0.85))
            fc_share = 1 / (1 + np.exp(-(carb_field[h] + rng.normal(0, 0.6))))
            fc = (1 - fp) * fc_share
            fs = 1 - fp - fc
            shale = END_MEMBERS.loc["shale"].copy()
            shale[["Zr", "TiO2", "Th"]] *= np.exp(prov_field[h])
            apatite = END_MEMBERS.loc["francolite"].copy()
            apatite[REE_ALL + ["Y", "U", "Sr"]] *= np.exp(apatite_field[h])
            mix = (fp * apatite + fs * shale
                   + fc * END_MEMBERS.loc["limestone"])
            # Lognormal analytical + geological scatter.
            assay = mix * np.exp(rng.normal(0, 0.08, len(ANALYTES)))
            rows.append({"hole_id": f"SYN{h:03d}", "easting": hx[h], "northing": hy[h],
                         "depth_m": round(depth, 1), "logger_cutoff": logger_cut[h],
                         "f_francolite_true": fp,
                         "f_carbonate_true": fc, **assay.to_dict()})

    df = pd.DataFrame(rows)
    df.insert(4, "lithology", _log_lithology(df["f_francolite_true"],
                                             df["f_carbonate_true"],
                                             df.pop("logger_cutoff"), rng))
    df.insert(0, "sample_id", [f"S{i:05d}" for i in range(len(df))])
    return df
