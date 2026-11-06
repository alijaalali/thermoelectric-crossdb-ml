# Builds per-compound target tables for property-prediction modeling from both databases.
import json
import re
import numpy as np
import pandas as pd


def elemset(formula):
    return frozenset(re.findall(r"[A-Z][a-z]?", re.sub(r"[0-9().,]", "", str(formula))))


# ---- JARVIS ----
with open("data/jarvis_te_raw.json") as f:
    jarvis = pd.DataFrame(json.load(f))

jarvis_rows = []
for _, r in jarvis.iterrows():
    jarvis_rows.append({
        "jid": r["jid"], "formula": r["formula"], "doping_type": "n",
        "Seebeck_uVK": r["n-Seebeck"], "log10_cond": np.log10(max(r["ncond"], 1e-12)),
        "powerfact": r["n-powerfact"],
        "elemset": elemset(r["formula"]),
    })
    jarvis_rows.append({
        "jid": r["jid"], "formula": r["formula"], "doping_type": "p",
        "Seebeck_uVK": r["p-Seebeck"], "log10_cond": np.log10(max(r["pcond"], 1e-12)),
        "powerfact": r["p-powerfact"],
        "elemset": elemset(r["formula"]),
    })
jarvis_target = pd.DataFrame(jarvis_rows)
jarvis_target["elemset"] = jarvis_target["elemset"].apply(lambda s: "|".join(sorted(s)))

# ---- ESTM ----
estm = pd.read_excel("data/SIMD/dataset/estm.xlsx")
estm = estm.rename(columns={
    "temperature(K)": "T", "seebeck_coefficient(μV/K)": "S",
    "electrical_conductivity(S/m)": "sigma", "power_factor(W/mK2)": "PF",
})
estm["dT300"] = (estm["T"] - 300).abs()
idx = estm.groupby("Formula")["dT300"].idxmin()
estm_300 = estm.loc[idx].copy()
estm_300["doping_type"] = np.where(estm_300["S"] < 0, "n", "p")
estm_300["log10_cond"] = np.log10(estm_300["sigma"].clip(lower=1e-12))
estm_300["elemset"] = estm_300["Formula"].apply(lambda f: "|".join(sorted(elemset(f))))
estm_300 = estm_300.rename(columns={"Formula": "formula", "S": "Seebeck_uVK"})
estm_target = estm_300[["formula", "doping_type", "Seebeck_uVK", "log10_cond", "PF", "T", "elemset"]]

jarvis_target.to_json("data/jarvis_target_table.json", orient="records", indent=2)
estm_target.to_json("data/estm_target_table.json", orient="records", indent=2)

summary = {
    "jarvis_target_rows": len(jarvis_target),
    "jarvis_unique_formulas": jarvis_target["formula"].nunique(),
    "estm_target_rows": len(estm_target),
    "estm_unique_formulas": estm_target["formula"].nunique(),
    "estm_T_distance_from_300K_median": float(estm_300["dT300"].median()),
    "estm_T_distance_from_300K_max": float(estm_300["dT300"].max()),
    "estm_doping_type_counts": estm_target["doping_type"].value_counts().to_dict(),
    "jarvis_doping_type_counts": jarvis_target["doping_type"].value_counts().to_dict(),
}
with open("data/phase2_target_table_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
