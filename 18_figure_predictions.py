# Lightweight single-fit re-runs (not a grid search) to get real prediction arrays for the parity plots and ROC curve.
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

def load_join(feat_path, target_path, target_cols):
    feats = json.load(open(feat_path))
    targets = json.load(open(target_path))
    fdf = pd.DataFrame(feats).drop_duplicates(subset="formula").set_index("formula")
    tdf = pd.DataFrame(targets)
    feat_cols = [c for c in fdf.columns if c != "formula"]
    merged = tdf.join(fdf, on="formula", how="inner")
    merged["doping_n"] = (merged["doping_type"] == "n").astype(int)
    feat_cols = feat_cols + ["doping_n"]
    merged = merged.dropna(subset=feat_cols + target_cols)
    return merged, feat_cols

out = {}

# --- JARVIS in-database Seebeck, RF, held-out group fold ---
jdf, jfeat = load_join("data/jarvis_magpie_features.json", "data/jarvis_target_table.json", ["Seebeck_uVK"])
groups = jdf["elemset"].values
gkf = GroupKFold(n_splits=3)
tr_idx, te_idx = next(gkf.split(jdf, groups=groups))
Xtr, Xte = jdf.iloc[tr_idx][jfeat].values, jdf.iloc[te_idx][jfeat].values
ytr, yte = jdf.iloc[tr_idx]["Seebeck_uVK"].values, jdf.iloc[te_idx]["Seebeck_uVK"].values
rf = RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=0, max_depth=20)
rf.fit(Xtr, ytr)
pred = rf.predict(Xte)
sub = np.random.choice(len(yte), size=min(3000, len(yte)), replace=False)
out["jarvis_indb_seebeck_parity"] = {
    "y_true": yte[sub].tolist(), "y_pred": pred[sub].tolist(),
    "r2_full_fold": float(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)),
}
print("JARVIS in-db Seebeck fold R2:", out["jarvis_indb_seebeck_parity"]["r2_full_fold"])

# --- JARVIS -> ESTM cross-database Seebeck, Ridge (best point-estimate model) ---
edf, efeat_cols_full = load_join("data/estm_magpie_features.json", "data/estm_target_table.json", ["Seebeck_uVK"])
common_feat = [c for c in jfeat if c in edf.columns]
scaler = StandardScaler().fit(jdf[common_feat].values)
ridge = Ridge(alpha=1.0)
ridge.fit(scaler.transform(jdf[common_feat].values), jdf["Seebeck_uVK"].values)
Xe = scaler.transform(edf[common_feat].values)
ye = edf["Seebeck_uVK"].values
pred_e = ridge.predict(Xe)
out["jarvis_to_estm_seebeck_parity"] = {
    "y_true": ye.tolist(), "y_pred": pred_e.tolist(),
    "r2": float(1 - np.sum((ye - pred_e) ** 2) / np.sum((ye - ye.mean()) ** 2)),
}
print("JARVIS->ESTM Seebeck R2:", out["jarvis_to_estm_seebeck_parity"]["r2"])

# --- Mechanism classifier ROC: RF, grouped out-of-fold predictions ---
labels = json.load(open("data/phase2_mechanism_labeling_summary.json"))
# reload full per-compound labels from the validation-adjacent source: mechanism labeling script output
mech = json.load(open("data/estm_mechanism_labels.json"))
mdf = pd.DataFrame(mech)
mdf = mdf[mdf["label"].isin(["hopping_like", "band_like"])].copy()
mdf["y"] = (mdf["label"] == "hopping_like").astype(int)
mdf = mdf.merge(pd.DataFrame(json.load(open("data/estm_magpie_features.json"))).drop_duplicates("formula"),
                 on="formula", how="inner")
comp_feat = [c for c in jfeat if c != "doping_n"]
Xm = mdf[comp_feat].values
ym = mdf["y"].values
groups_m = mdf["formula"].values
gkf5 = GroupKFold(n_splits=5)
rfc = RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=0, class_weight="balanced", max_depth=12)
oof_proba = cross_val_predict(rfc, Xm, ym, cv=gkf5, groups=groups_m, method="predict_proba")[:, 1]
out["mechanism_classifier_roc"] = {"y_true": ym.tolist(), "y_score": oof_proba.tolist()}
print("n mechanism-labeled used:", len(ym), "n positive:", int(ym.sum()))

with open("data/phase3_figure_predictions.json", "w") as f:
    json.dump(out, f)
print("saved data/phase3_figure_predictions.json")
