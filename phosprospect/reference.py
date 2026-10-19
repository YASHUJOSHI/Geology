"""Reference compositions and unit conversions for sedimentary phosphorite geochemistry."""

# Post-Archean Australian Shale, Taylor & McLennan (1985); REE after McLennan (1989).
# Used as the detrital baseline: a sample's departure from PAAS is the authigenic signal.
PAAS_REE_PPM = {
    "La": 38.2, "Ce": 79.6, "Pr": 8.83, "Nd": 33.9, "Sm": 5.55,
    "Eu": 1.08, "Gd": 4.66, "Tb": 0.774, "Dy": 4.68, "Ho": 0.991,
    "Er": 2.85, "Tm": 0.405, "Yb": 2.82, "Lu": 0.433,
}

PAAS_TRACE_PPM = {
    "V": 150.0, "Cr": 110.0, "Ni": 55.0, "Co": 23.0, "Cu": 50.0,
    "Zn": 85.0, "Mo": 1.0, "U": 3.1, "Th": 14.6, "Ba": 650.0,
    "Sr": 200.0, "Y": 27.0, "Zr": 210.0, "Sc": 16.0, "Cd": 0.1,
}

# PAAS Al content in ppm (18.9 wt% Al2O3 -> 10.0 wt% Al).
PAAS_AL_PPM = 100_000.0

REE_LIGHT = ["La", "Ce", "Pr", "Nd", "Sm"]
REE_MIDDLE = ["Eu", "Gd", "Tb", "Dy"]
REE_HEAVY = ["Ho", "Er", "Tm", "Yb", "Lu"]
REE_ALL = REE_LIGHT + REE_MIDDLE + REE_HEAVY

# Oxide weight fraction -> element weight fraction.
OXIDE_TO_ELEMENT = {
    "SiO2": ("Si", 0.4674), "Al2O3": ("Al", 0.5293), "Fe2O3": ("Fe", 0.6994),
    "FeO": ("Fe", 0.7773), "CaO": ("Ca", 0.7147), "MgO": ("Mg", 0.6030),
    "Na2O": ("Na", 0.7419), "K2O": ("K", 0.8301), "TiO2": ("Ti", 0.5995),
    "P2O5": ("P", 0.4364), "MnO": ("Mn", 0.7745),
}

MAJOR_OXIDES = list(OXIDE_TO_ELEMENT)

# Chondritic Y/Ho. Seawater runs 44-74 because Ho is scavenged preferentially,
# so apatite precipitated from seawater inherits a superchondritic ratio.
CHONDRITIC_Y_HO = 28.0

# Stoichiometric CaO/P2O5 of carbonate-fluorapatite (francolite).
FRANCOLITE_CAO_P2O5 = 1.32


def oxide_pct_to_element_ppm(value_pct, oxide):
    """Convert a major-oxide wt% value to element ppm."""
    _, factor = OXIDE_TO_ELEMENT[oxide]
    return value_pct * factor * 10_000.0

# CI chondrite, Sun & McDonough (1989), ppm. Used for igneous-style REE spider plots;
# PAAS above is the better baseline for sedimentary rocks, so both are provided.
CHONDRITE_REE_PPM = {
    "La": 0.237, "Ce": 0.613, "Pr": 0.0928, "Nd": 0.457, "Sm": 0.148,
    "Eu": 0.0563, "Gd": 0.199, "Tb": 0.0361, "Dy": 0.246, "Ho": 0.0546,
    "Er": 0.160, "Tm": 0.0247, "Yb": 0.161, "Lu": 0.0246,
}
