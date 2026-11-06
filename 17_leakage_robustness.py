# Leakage-robustness check: reruns cross-database transfer excluding exact-formula overlaps between the two databases.
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

magpie_j = pd.read_json("data/jarvis_magpie_features.json")
magpie_e = pd.read_json("data/estm_magpie_features.json")
tgt_j = pd.read_json("data/jarvis_target_table.json")
tgt_e = pd.read_json("data/estm_target_table.json")
feat_cols = [c for c in magpie_j.columns if c != "formula"]

j = tgt_j.merge(magpie_j, on="formula", how="inner")
e = tgt_e.merge(magpie_e, on="formula", how="inner")
j["doping_n"] = (j["doping_type"] == "n").astype(int)
e["doping_n"] = (e["doping_type"] == "n").astype(int)
X_cols = feat_cols + ["doping_n"]
for df in (j, e):
    df[X_cols] = df[X_cols].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=X_cols, inplace=True)

exact_overlap = set(j["formula"]).intersection(set(e["formula"]))
print(f"Exact-formula overlap: {len(exact_overlap)} formulas")

j_noleak = j[~j["formula"].isin(exact_overlap)]
e_noleak = e[~e["formula"].isin(exact_overlap)]


def transfer_r2(train_df, test_df, target):
    Xtr, ytr = train_df[X_cols].to_numpy(), train_df[target].to_numpy()
    Xte, yte = test_df[X_cols].to_numpy(), test_df[target].to_numpy()
    # Same RF settings as 10_property_prediction.py, so the with-overlap baseline
    # reproduces the RandomForest cross-database estimate reported there.
    model = RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=4, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {"r2": float(r2_score(yte, pred)), "mae": float(mean_absolute_error(yte, pred)), "n_test": int(len(yte))}


results = {"n_exact_overlap_excluded": len(exact_overlap)}
for target in ["Seebeck_uVK", "log10_cond"]:
    results[f"JARVIStoESTM_WITH_overlap__{target}"] = transfer_r2(j, e, target)
    results[f"JARVIStoESTM_NO_overlap__{target}"] = transfer_r2(j_noleak, e_noleak, target)
    results[f"ESTMtoJARVIS_WITH_overlap__{target}"] = transfer_r2(e, j, target)
    results[f"ESTMtoJARVIS_NO_overlap__{target}"] = transfer_r2(e_noleak, j_noleak, target)
    print(target, "JARVIS->ESTM with/without overlap:",
          results[f"JARVIStoESTM_WITH_overlap__{target}"]["r2"],
          results[f"JARVIStoESTM_NO_overlap__{target}"]["r2"])

with open("data/phase2_leakage_robustness.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE_LEAKAGE")
