# Mechanism-classification models predicting the derived band-like/hopping-like label from composition descriptors alone, with composition-grouped CV and SHAP feature importance.
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import shap

labels = pd.read_json("data/estm_mechanism_labels.json")
magpie_e = pd.read_json("data/estm_magpie_features.json")

df = labels.merge(magpie_e, on="formula", how="inner")
df = df[df["label"].isin(["band_like", "hopping_like"])].copy()
df["y"] = (df["label"] == "hopping_like").astype(int)

feat_cols = [c for c in magpie_e.columns if c != "formula"]
df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
df.dropna(subset=feat_cols, inplace=True)

import re
df["elemset"] = df["formula"].apply(lambda f: "|".join(sorted(set(re.findall(r"[A-Z][a-z]?", re.sub(r"[0-9().,]", "", str(f)))))))

print(f"Classifier dataset: {len(df)} formulas ({df['y'].sum()} hopping_like, {(~df['y'].astype(bool)).sum()} band_like)")

X = df[feat_cols].to_numpy()
y = df["y"].to_numpy()
groups = df["elemset"].to_numpy()

n_groups = len(set(groups))
k = min(5, n_groups)
gkf = GroupKFold(n_splits=k)


def make_clfs():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=4),
        "XGBoostClassifier": xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                                scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
                                                n_jobs=4, random_state=42, verbosity=0, eval_metric="logloss"),
    }


metrics = {name: {"accuracy": [], "f1": [], "auc": []} for name in make_clfs()}
for tr_idx, te_idx in gkf.split(X, y, groups):
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    for name, clf in make_clfs().items():
        Xtr_use, Xte_use = (Xtr_s, Xte_s) if name == "LogisticRegression" else (Xtr, Xte)
        clf.fit(Xtr_use, ytr)
        pred = clf.predict(Xte_use)
        try:
            proba = clf.predict_proba(Xte_use)[:, 1]
            auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else np.nan
        except Exception:
            auc = np.nan
        metrics[name]["accuracy"].append(accuracy_score(yte, pred))
        metrics[name]["f1"].append(f1_score(yte, pred, zero_division=0))
        metrics[name]["auc"].append(auc)

summary = {}
for name, m in metrics.items():
    summary[name] = {
        "accuracy_mean": float(np.mean(m["accuracy"])), "accuracy_std": float(np.std(m["accuracy"])),
        "f1_mean": float(np.mean(m["f1"])), "f1_std": float(np.std(m["f1"])),
        "auc_mean": float(np.nanmean(m["auc"])), "auc_std": float(np.nanstd(m["auc"])),
        "n_folds": len(m["accuracy"]),
    }
print(json.dumps(summary, indent=2))

# SHAP on an XGBoost model trained on all data (for interpretability, not for CV metrics)
clf_full = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                              scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
                              n_jobs=4, random_state=42, verbosity=0, eval_metric="logloss")
clf_full.fit(X, y)
explainer = shap.TreeExplainer(clf_full)
shap_values = explainer.shap_values(X)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(-mean_abs_shap)[:15]
top_features = [{"feature": feat_cols[i], "mean_abs_shap": float(mean_abs_shap[i])} for i in top_idx]

result = {
    "n_formulas_used": int(len(df)),
    "n_hopping_like": int(df["y"].sum()),
    "n_band_like": int((~df["y"].astype(bool)).sum()),
    "cv_metrics": summary,
    "top15_shap_features_full_model": top_features,
}
with open("data/phase2_mechanism_classification.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
