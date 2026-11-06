# Extends target tables with power factor (unit-harmonized to JARVIS's muW/(m K^2) convention), ESTM thermal conductivity/ZT, and a JARVIS electronic-kappa / electronic-ZT-proxy pair (labeled "electronic" to distinguish them from total thermal conductivity and ZT).
import json
import re
import numpy as np
import pandas as pd

T_JARVIS_K = 300.0  # BoltzTraP screening convention (Choudhary2020TE)


def elemset(formula):
    return frozenset(re.findall(r"[A-Z][a-z]?", re.sub(r"[0-9().,]", "", str(formula))))


# ---------------- JARVIS ----------------
with open("data/jarvis_te_raw.json") as f:
    jarvis = pd.DataFrame(json.load(f))

jarvis_rows = []
for _, r in jarvis.iterrows():
    kappa_n = r["nkappa"]
    kappa_p = r["pkappa"]
    S_n_V = r["n-Seebeck"] * 1e-6  # muV/K -> V/K
    S_p_V = r["p-Seebeck"] * 1e-6
    zt_elec_n = (S_n_V ** 2) * r["ncond"] * T_JARVIS_K / kappa_n if kappa_n and kappa_n > 0 else np.nan
    zt_elec_p = (S_p_V ** 2) * r["pcond"] * T_JARVIS_K / kappa_p if kappa_p and kappa_p > 0 else np.nan
    jarvis_rows.append({
        "jid": r["jid"], "formula": r["formula"], "doping_type": "n",
        "Seebeck_uVK": r["n-Seebeck"], "log10_cond": np.log10(max(r["ncond"], 1e-12)),
        "powerfact": r["n-powerfact"],
        "kappa_elec_Wm1K1": kappa_n,
        "zt_elec_proxy": zt_elec_n,
        "elemset": elemset(r["formula"]),
    })
    jarvis_rows.append({
        "jid": r["jid"], "formula": r["formula"], "doping_type": "p",
        "Seebeck_uVK": r["p-Seebeck"], "log10_cond": np.log10(max(r["pcond"], 1e-12)),
        "powerfact": r["p-powerfact"],
        "kappa_elec_Wm1K1": kappa_p,
        "zt_elec_proxy": zt_elec_p,
        "elemset": elemset(r["formula"]),
    })
jarvis_target = pd.DataFrame(jarvis_rows)
jarvis_target["elemset"] = jarvis_target["elemset"].apply(lambda s: "|".join(sorted(s)))

# filter kappa_elec/zt_elec_proxy: require positive finite values
jarvis_target["log10_kappa_elec"] = np.where(
    jarvis_target["kappa_elec_Wm1K1"] > 0,
    np.log10(jarvis_target["kappa_elec_Wm1K1"].clip(lower=1e-12)),
    np.nan,
)

# ---------------- ESTM ----------------
estm = pd.read_excel("data/SIMD/dataset/estm.xlsx")
estm = estm.rename(columns={
    "temperature(K)": "T", "seebeck_coefficient(μV/K)": "S",
    "electrical_conductivity(S/m)": "sigma", "power_factor(W/mK2)": "PF_SI",
    "thermal_conductivity(W/mK)": "kappa_total", "ZT": "ZT_reported",
})
estm["PF"] = estm["PF_SI"] * 1e6  # W/(m K^2) -> muW/(m K^2), harmonized to JARVIS units
estm["dT300"] = (estm["T"] - 300).abs()
idx = estm.groupby("Formula")["dT300"].idxmin()
estm_300 = estm.loc[idx].copy()
estm_300["doping_type"] = np.where(estm_300["S"] < 0, "n", "p")
estm_300["log10_cond"] = np.log10(estm_300["sigma"].clip(lower=1e-12))
estm_300["log10_kappa_total"] = np.log10(estm_300["kappa_total"].clip(lower=1e-12))
estm_300["elemset"] = estm_300["Formula"].apply(lambda f: "|".join(sorted(elemset(f))))
estm_300 = estm_300.rename(columns={"Formula": "formula", "S": "Seebeck_uVK"})
estm_300 = estm_300.rename(columns={"PF": "powerfact"})
estm_target = estm_300[[
    "formula", "doping_type", "Seebeck_uVK", "log10_cond", "powerfact", "T",
    "kappa_total", "log10_kappa_total", "ZT_reported", "elemset",
]]

jarvis_target.to_json("data/jarvis_target_table_ext.json", orient="records", indent=2)
estm_target.to_json("data/estm_target_table_ext.json", orient="records", indent=2)

# ---------------- ZT arithmetic-consistency check (ESTM's own stated ZT) ----------------
check = estm_300.copy()
check["ZT_computed"] = (check["Seebeck_uVK"] * 1e-6) ** 2 * check["sigma"] * check["T"] / check["kappa_total"]
valid = check[(check["ZT_reported"] > 0) & np.isfinite(check["ZT_computed"]) & (check["ZT_computed"] > 0)].copy()
valid["rel_err"] = (valid["ZT_computed"] - valid["ZT_reported"]).abs() / valid["ZT_reported"]
zt_check_summary = {
    "n_checked": int(len(valid)),
    "n_total_estm_rows_with_zt": int((check["ZT_reported"] > 0).sum()),
    "median_rel_err": float(valid["rel_err"].median()),
    "within_1pct": float((valid["rel_err"] < 0.01).mean()),
    "within_10pct": float((valid["rel_err"] < 0.10).mean()),
    "within_20pct": float((valid["rel_err"] < 0.20).mean()),
    "n_exceeding_50pct": int((valid["rel_err"] > 0.50).sum()),
    "note": ("Note: this check uses the same representative 300K-closest row selected for "
             "modeling, not every (Formula,T) measurement in the raw file, so n_checked "
             "reflects the 880-formula representative set, not all 5205 raw rows."),
}

# ---------------- JARVIS unit check: compare PF columns pre/post harmonization ----------------
unit_check = {
    "jarvis_powerfact_median_native_units": float(jarvis_target["powerfact"].median()),
    "estm_PF_median_native_W_m_K2": float(estm_300["PF_SI"].median()),
    "estm_PF_median_harmonized_muW_m_K2": float(estm_300["powerfact"].median()),
    "note": ("JARVIS n/p-powerfact already in muW/(m K^2) (see 07_duplicate_and_arithmetic_audit.py); ESTM's "
             "power_factor(W/mK2) column multiplied by 1e6 to match. After harmonization "
             "both medians are the same order of magnitude (JARVIS ~578-870, ESTM ~420), "
             "consistent with the conversion direction used."),
}

summary = {
    "jarvis_rows": len(jarvis_target),
    "jarvis_unique_formulas": jarvis_target["formula"].nunique(),
    "jarvis_rows_with_positive_kappa_elec": int((jarvis_target["kappa_elec_Wm1K1"] > 0).sum()),
    "jarvis_rows_with_finite_zt_elec_proxy": int(jarvis_target["zt_elec_proxy"].notna().sum()
                                                   & (jarvis_target["zt_elec_proxy"] > 0).sum()),
    "estm_rows": len(estm_target),
    "estm_unique_formulas": estm_target["formula"].nunique(),
    "estm_rows_with_kappa_total": int(estm_300["kappa_total"].notna().sum()),
    "estm_rows_with_zt_reported": int((estm_300["ZT_reported"] > 0).sum()),
    "power_factor_unit_check": unit_check,
    "zt_arithmetic_consistency_check": zt_check_summary,
}
with open("data/phase4_target_table_ext_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
