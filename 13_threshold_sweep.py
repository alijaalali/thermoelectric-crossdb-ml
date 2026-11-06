# Threshold sensitivity sweep for the mechanism-labeling procedure in 05.
import json
import numpy as np
import pandas as pd
from scipy import stats

KB_EV = 8.617333262e-5

df = pd.read_excel("data/SIMD/dataset/estm.xlsx")
df = df.rename(columns={
    "temperature(K)": "T", "seebeck_coefficient(μV/K)": "S",
    "electrical_conductivity(S/m)": "sigma",
})


def fit_formula(g, min_points):
    g = g.sort_values("T")
    n = len(g)
    if n < min_points:
        return None
    T = g["T"].to_numpy()
    sigma = g["sigma"].to_numpy()
    if np.any(sigma <= 0) or len(np.unique(T)) < min_points:
        return None
    rho, p_rho = stats.spearmanr(T, sigma)
    invT = 1.0 / T
    lnsig = np.log(sigma)
    slope, intercept, r_arr, p_arr, se = stats.linregress(invT, lnsig)
    Ea_eV = -slope * KB_EV
    return {"rho": rho, "p": p_rho, "Ea_eV": Ea_eV}


def label_all(min_points, rho_thresh, p_thresh=0.05, ea_min=0.0, ea_max=2.0):
    out = {}
    for formula, g in df.groupby("Formula"):
        fit = fit_formula(g, min_points)
        if fit is None:
            out[formula] = "excluded_insufficient_data"
            continue
        if fit["p"] > p_thresh:
            out[formula] = "ambiguous_no_significant_trend"
        elif fit["rho"] >= rho_thresh and ea_min <= fit["Ea_eV"] <= ea_max:
            out[formula] = "hopping_like"
        elif fit["rho"] <= -rho_thresh:
            out[formula] = "band_like"
        else:
            out[formula] = "ambiguous_mixed_signal"
    return out


import re

def elemset(f):
    return set(re.findall(r"[A-Z][a-z]?", re.sub(r"[0-9().,]", "", str(f))))

estm_elem = df.assign(elemset=df["Formula"].apply(elemset))
hopping_groups = [{"La", "Co", "O"}, {"Ca", "Co", "O"}, {"Fe", "O"}, {"Ti", "O"}]
band_groups = [{"Bi", "Te"}, {"Pb", "Te"}, {"Si", "Ge"}, {"Co", "Sb"}]
hopping_formulas = set()
for req in hopping_groups:
    hopping_formulas |= set(estm_elem.loc[estm_elem["elemset"].apply(lambda s: req.issubset(s)), "Formula"].unique())
band_formulas = set()
for req in band_groups:
    band_formulas |= set(estm_elem.loc[estm_elem["elemset"].apply(lambda s: req.issubset(s)), "Formula"].unique())
overlap = hopping_formulas & band_formulas
hopping_formulas -= overlap
band_formulas -= overlap


def agreement(labels, formulas, expected):
    n_agree = sum(1 for f in formulas if labels.get(f) == expected)
    opposite = "band_like" if expected == "hopping_like" else "hopping_like"
    n_opp = sum(1 for f in formulas if labels.get(f) == opposite)
    denom = n_agree + n_opp
    return n_agree, n_opp, (n_agree / denom if denom else None)


sweep_results = []
for min_points in [3, 4, 5]:
    for rho_thresh in [0.5, 0.6, 0.7]:
        labels = label_all(min_points, rho_thresh)
        counts = pd.Series(list(labels.values())).value_counts().to_dict()
        h_agree, h_opp, h_rate = agreement(labels, hopping_formulas, "hopping_like")
        b_agree, b_opp, b_rate = agreement(labels, band_formulas, "band_like")
        sweep_results.append({
            "min_points": min_points, "rho_thresh": rho_thresh,
            "label_counts": counts,
            "hopping_validation_agree": h_agree, "hopping_validation_opposite": h_opp,
            "hopping_validation_rate": h_rate,
            "band_validation_agree": b_agree, "band_validation_opposite": b_opp,
            "band_validation_rate": b_rate,
        })
        print(f"min_points={min_points} rho={rho_thresh}: "
              f"hopping={counts.get('hopping_like',0)} band={counts.get('band_like',0)} "
              f"hop_val_rate={h_rate} band_val_rate={b_rate}")

with open("data/phase2_threshold_sweep.json", "w") as f:
    json.dump(sweep_results, f, indent=2, default=str)
print("DONE_SWEEP")
