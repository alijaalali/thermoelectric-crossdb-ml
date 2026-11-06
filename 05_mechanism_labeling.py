# Derives a band-like vs. small-polaron-hopping transport-mechanism label for each ESTM formula from its own multi-temperature conductivity/Seebeck curve shape (a heuristic classification, not a confirmed microscopic mechanism).
import json
import numpy as np
import pandas as pd
from scipy import stats

KB_EV = 8.617333262e-5  # eV/K

df = pd.read_excel("data/SIMD/dataset/estm.xlsx")
df = df.rename(columns={
    "temperature(K)": "T",
    "seebeck_coefficient(μV/K)": "S",
    "electrical_conductivity(S/m)": "sigma",
    "thermal_conductivity(W/mK)": "kappa",
    "power_factor(W/mK2)": "PF",
})


def fit_formula(g, min_points):
    g = g.sort_values("T")
    n = len(g)
    if n < min_points:
        return None
    T = g["T"].to_numpy()
    sigma = g["sigma"].to_numpy()
    S = g["S"].to_numpy()
    if np.any(sigma <= 0) or len(np.unique(T)) < min_points:
        return None

    # Primary signal: monotonic trend of sigma vs T (Spearman rank correlation, insensitive to
    # non-Arrhenius curvature)
    rho, p_rho = stats.spearmanr(T, sigma)

    # Quantitative signal: Arrhenius fit of ln(sigma) vs 1/T
    invT = 1.0 / T
    lnsig = np.log(sigma)
    slope, intercept, r_arr, p_arr, se = stats.linregress(invT, lnsig)
    Ea_eV = -slope * KB_EV  # eV; positive Ea => hopping-consistent
    r2_arr = r_arr ** 2

    # Secondary signal: Seebeck |S| trend vs T (weaker/plateau expected for
    # Heikes-type hopping; stronger near-linear increase for band transport)
    rho_S, p_S = stats.spearmanr(T, np.abs(S))

    return {
        "n_points": int(n),
        "T_min": float(T.min()),
        "T_max": float(T.max()),
        "spearman_rho_sigma_T": float(rho),
        "spearman_p_sigma_T": float(p_rho),
        "arrhenius_slope": float(slope),
        "arrhenius_Ea_eV": float(Ea_eV),
        "arrhenius_r2": float(r2_arr),
        "spearman_rho_absS_T": float(rho_S),
        "spearman_p_absS_T": float(p_S),
    }


def label_from_fit(fit, rho_thresh, p_thresh, ea_min, ea_max):
    if fit is None:
        return "excluded_insufficient_data"
    rho = fit["spearman_rho_sigma_T"]
    p = fit["spearman_p_sigma_T"]
    ea = fit["arrhenius_Ea_eV"]
    if p > p_thresh:
        return "ambiguous_no_significant_trend"
    if rho >= rho_thresh and ea_min <= ea <= ea_max:
        return "hopping_like"
    if rho <= -rho_thresh:
        return "band_like"
    return "ambiguous_mixed_signal"


# Primary (default) thresholds -- sensitivity sweep over these is done in
# scripts/13_threshold_sweep.py
MIN_POINTS = 4
RHO_THRESH = 0.6
P_THRESH = 0.05
EA_MIN, EA_MAX = 0.0, 2.0

records = []
for formula, g in df.groupby("Formula"):
    fit = fit_formula(g, MIN_POINTS)
    label = label_from_fit(fit, RHO_THRESH, P_THRESH, EA_MIN, EA_MAX)
    rec = {"formula": formula, "label": label}
    if fit is not None:
        rec.update(fit)
    records.append(rec)

out = pd.DataFrame(records)
counts = out["label"].value_counts().to_dict()
print("Label counts (default thresholds: min_points=%d, rho=%.2f, p=%.2f, Ea in [%.2f,%.2f]):"
      % (MIN_POINTS, RHO_THRESH, P_THRESH, EA_MIN, EA_MAX))
print(counts)

out.to_json("data/estm_mechanism_labels.json", orient="records", indent=2)

summary = {
    "method": "Spearman(sigma,T) sign/significance + Arrhenius Ea plausibility range; "
              "heuristic label derived from each formula's own multi-T data, "
              "not a confirmed microscopic mechanism",
    "default_thresholds": {
        "min_points": MIN_POINTS, "rho_thresh": RHO_THRESH,
        "p_thresh": P_THRESH, "ea_min_eV": EA_MIN, "ea_max_eV": EA_MAX,
    },
    "n_total_formulas": int(df["Formula"].nunique()),
    "label_counts": counts,
}
with open("data/phase2_mechanism_labeling_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
