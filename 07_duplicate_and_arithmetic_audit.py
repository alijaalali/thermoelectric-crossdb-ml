# Duplicate-record audit and per-entry arithmetic consistency check (power factor = Seebeck^2 * conductivity) for both datasets.
import json
import numpy as np
import pandas as pd

results = {}

# --- (a) JARVIS duplicate audit ---
with open("data/jarvis_te_raw.json") as f:
    jarvis = pd.DataFrame(json.load(f))

jid_dupes = jarvis["jid"].duplicated().sum()
exact_row_dupes = jarvis.drop(columns=["atoms"]).duplicated().sum()
formula_counts = jarvis["formula"].value_counts()
results["jarvis"] = {
    "n_rows": int(len(jarvis)),
    "n_unique_jid": int(jarvis["jid"].nunique()),
    "jid_duplicates": int(jid_dupes),
    "exact_full_row_duplicates_excl_atoms": int(exact_row_dupes),
    "n_unique_formula": int(jarvis["formula"].nunique()),
    "max_rows_sharing_one_formula": int(formula_counts.max()),
}

# --- (a) ESTM duplicate audit ---
estm = pd.read_excel("data/SIMD/dataset/estm.xlsx")
exact_dupes_estm = estm.duplicated().sum()
formula_T_dupes = estm.duplicated(subset=["Formula", "temperature(K)"]).sum()
# look closer at the (Formula, T) duplicate pairs: same source or different?
dup_mask = estm.duplicated(subset=["Formula", "temperature(K)"], keep=False)
dup_rows = estm[dup_mask].sort_values(["Formula", "temperature(K)"])
same_ref_count = 0
diff_ref_count = 0
for (formula, T), g in dup_rows.groupby(["Formula", "temperature(K)"]):
    if g["reference"].nunique() == 1:
        same_ref_count += 1
    else:
        diff_ref_count += 1

results["estm"] = {
    "n_rows": int(len(estm)),
    "exact_full_row_duplicates": int(exact_dupes_estm),
    "formula_temperature_duplicate_rows": int(formula_T_dupes),
    "formula_temperature_duplicate_groups_same_reference": int(same_ref_count),
    "formula_temperature_duplicate_groups_different_reference": int(diff_ref_count),
    "interpretation": "formula+T duplicates with the same reference are likely accidental "
                       "re-entries in the ESTM curation; duplicates with different references "
                       "are independent literature re-measurements of the same compound/T and "
                       "are legitimate.",
}

# --- (b) Arithmetic consistency: ESTM ---
S_V = estm["seebeck_coefficient(μV/K)"].to_numpy() * 1e-6  # V/K
sigma = estm["electrical_conductivity(S/m)"].to_numpy()  # S/m
PF_reported = estm["power_factor(W/mK2)"].to_numpy()
PF_computed = (S_V ** 2) * sigma  # W/(m K^2)
rel_err = np.abs(PF_computed - PF_reported) / np.maximum(np.abs(PF_reported), 1e-30)
results["estm_arithmetic_check"] = {
    "formula": "PF_computed = Seebeck[V/K]^2 * sigma[S/m]  (expected units W/(m K^2))",
    "median_relative_error": float(np.median(rel_err)),
    "fraction_within_1pct": float(np.mean(rel_err < 0.01)),
    "fraction_within_5pct": float(np.mean(rel_err < 0.05)),
    "n_above_10pct_error": int(np.sum(rel_err > 0.10)),
    "max_relative_error": float(np.max(rel_err)),
}

# --- (b) Arithmetic consistency: JARVIS (empirically determine unit convention) ---
nS = jarvis["n-Seebeck"].to_numpy() * 1e-6  # assume muV/K -> V/K
ncond = jarvis["ncond"].to_numpy()  # assume 1/(Ohm m) = S/m
n_pf_reported = jarvis["n-powerfact"].to_numpy()
pf_computed_SI = (nS ** 2) * ncond  # W/(m K^2) if assumptions hold
# Empirically determined (not assumed): reported/computed ratio has a tight
# median of ~1.0e6 across 23,204 compounds, i.e. JARVIS n-powerfact is
# reported in microW/(m K^2), not the microW/(cm K^2) initially assumed.
# 1 W/(m K^2) = 1e6 microW/(m K^2).
pf_computed_jarvis_units = pf_computed_SI * 1e6
mask = (n_pf_reported != 0) & np.isfinite(n_pf_reported) & np.isfinite(pf_computed_jarvis_units)
rel_err_j = np.abs(pf_computed_jarvis_units[mask] - n_pf_reported[mask]) / np.maximum(np.abs(n_pf_reported[mask]), 1e-30)
results["jarvis_arithmetic_check"] = {
    "formula": "n-powerfact[muW/(m K^2)] vs (n-Seebeck[V/K])^2 * ncond[S/m] converted W/(m K^2) -> muW/(m K^2) (x1e6; unit empirically determined from the reported/computed ratio, median~1.0e6 across 23,204 compounds)",
    "n_checked": int(mask.sum()),
    "median_relative_error": float(np.median(rel_err_j)) if mask.sum() else None,
    "fraction_within_5pct": float(np.mean(rel_err_j < 0.05)) if mask.sum() else None,
    "fraction_within_20pct": float(np.mean(rel_err_j < 0.20)) if mask.sum() else None,
}

with open("data/phase2_duplicate_arithmetic_audit.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2, default=str))
