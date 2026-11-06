# SHAP feature attribution for the property regression models (Seebeck, log10 conductivity,
# power factor), analogous to the SHAP analysis of the mechanism classifier in
# 12_mechanism_classifier.py. Thermal conductivity and ZT are excluded because the JARVIS
# kappa column is electronic-only and does not match ESTM's total thermal conductivity.
# LightGBM (best or near-best model for all three targets in the benchmark) is trained on the
# full JARVIS table, the larger database, for interpretation only.
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

RNG = 42

magpie_j = pd.read_json("data/jarvis_magpie_features.json")
tgt_j = pd.read_json("data/jarvis_target_table_ext.json")
feat_cols = [c for c in magpie_j.columns if c != "formula"]

j = tgt_j.merge(magpie_j, on="formula", how="inner")
j["doping_n"] = (j["doping_type"] == "n").astype(int)
X_cols = feat_cols + ["doping_n"]
j[X_cols] = j[X_cols].apply(pd.to_numeric, errors="coerce")
j.dropna(subset=X_cols, inplace=True)

out = {"n_rows": len(j), "model": "LightGBM(n_estimators=300, max_depth=-1, learning_rate=0.08), trained on full JARVIS table for interpretability only", "targets": {}}

for target, label in [("Seebeck_uVK", "Seebeck coefficient"), ("log10_cond", "log10(electrical conductivity)"), ("powerfact", "power factor")]:
    X = j[X_cols].to_numpy()
    y = j[target].to_numpy()
    model = lgb.LGBMRegressor(n_estimators=300, max_depth=-1, learning_rate=0.08, random_state=RNG, verbosity=-1, n_jobs=4)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(-mean_abs_shap)[:15]
    top_features = [{"feature": X_cols[i], "mean_abs_shap": float(mean_abs_shap[i])} for i in top_idx]
    out["targets"][target] = {"label": label, "top15_shap_features": top_features}
    print(label, "top5:", [t["feature"] for t in top_features[:5]], flush=True)

with open("data/phase5_regression_shap.json", "w") as f:
    json.dump(out, f, indent=2)
print("done")
