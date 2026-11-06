# Grouped-bootstrap 95% CIs for the headline cross-database transfer R^2 numbers and the mechanism-classification AUC.
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
import re

RNG = np.random.default_rng(42)
N_BOOT = 500

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


def grouped_bootstrap_r2(train_df, test_df, target, n_boot=N_BOOT):
    Xtr = train_df[X_cols].to_numpy()
    ytr = train_df[target].to_numpy()
    model = RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=42)
    model.fit(Xtr, ytr)

    test_groups = test_df["elemset"].unique()
    boot_r2 = []
    Xte_full = test_df[X_cols].to_numpy()
    yte_full = test_df[target].to_numpy()
    pred_full = model.predict(Xte_full)
    test_df = test_df.reset_index(drop=True)
    pred_series = pd.Series(pred_full, index=test_df.index)
    y_series = pd.Series(yte_full, index=test_df.index)
    group_to_idx = test_df.groupby("elemset").apply(lambda g: g.index.to_numpy())
    for _ in range(n_boot):
        sampled_groups = RNG.choice(test_groups, size=len(test_groups), replace=True)
        idxs = np.concatenate([group_to_idx[g] for g in sampled_groups])
        yb = y_series.loc[idxs].to_numpy()
        pb = pred_series.loc[idxs].to_numpy()
        if len(set(yb)) > 1:
            boot_r2.append(r2_score(yb, pb))
    point_r2 = r2_score(yte_full, pred_full)
    return {
        "point_r2": float(point_r2),
        "boot_ci_lower_2.5": float(np.percentile(boot_r2, 2.5)),
        "boot_ci_upper_97.5": float(np.percentile(boot_r2, 97.5)),
        "n_boot_valid": len(boot_r2),
    }


results = {}
for target in ["Seebeck_uVK", "log10_cond"]:
    results[f"JARVIStoESTM__{target}"] = grouped_bootstrap_r2(j, e, target)
    results[f"ESTMtoJARVIS__{target}"] = grouped_bootstrap_r2(e, j, target)
    print(target, results[f"JARVIStoESTM__{target}"], results[f"ESTMtoJARVIS__{target}"])

# Mechanism classification AUC grouped bootstrap
labels = pd.read_json("data/estm_mechanism_labels.json")
dfc = labels.merge(magpie_e, on="formula", how="inner")
dfc = dfc[dfc["label"].isin(["band_like", "hopping_like"])].copy()
dfc[feat_cols] = dfc[feat_cols].apply(pd.to_numeric, errors="coerce")
dfc.dropna(subset=feat_cols, inplace=True)
dfc["elemset"] = dfc["formula"].apply(lambda f: "|".join(sorted(set(re.findall(r"[A-Z][a-z]?", re.sub(r"[0-9().,]", "", str(f)))))))
dfc = dfc.reset_index(drop=True)
Xc = dfc[feat_cols].to_numpy()
yc = (dfc["label"] == "hopping_like").astype(int).to_numpy()

from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
oof_pred = np.full(len(dfc), np.nan)
for tr_idx, te_idx in gkf.split(Xc, yc, dfc["elemset"].to_numpy()):
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=4)
    clf.fit(Xc[tr_idx], yc[tr_idx])
    oof_pred[te_idx] = clf.predict_proba(Xc[te_idx])[:, 1]

valid = ~np.isnan(oof_pred)
point_auc = roc_auc_score(yc[valid], oof_pred[valid])
groups_c = dfc.loc[valid, "elemset"].to_numpy()
y_valid = yc[valid]
pred_valid = oof_pred[valid]
unique_groups_c = np.unique(groups_c)
boot_auc = []
group_idx_map = {g: np.where(groups_c == g)[0] for g in unique_groups_c}
for _ in range(N_BOOT):
    sampled = RNG.choice(unique_groups_c, size=len(unique_groups_c), replace=True)
    idxs = np.concatenate([group_idx_map[g] for g in sampled])
    yb, pb = y_valid[idxs], pred_valid[idxs]
    if len(set(yb)) > 1:
        boot_auc.append(roc_auc_score(yb, pb))

results["mechanism_classification_AUC_grouped_oof"] = {
    "point_auc": float(point_auc),
    "boot_ci_lower_2.5": float(np.percentile(boot_auc, 2.5)),
    "boot_ci_upper_97.5": float(np.percentile(boot_auc, 97.5)),
    "n_boot_valid": len(boot_auc),
}
print("mechanism classification AUC:", results["mechanism_classification_AUC_grouped_oof"])

with open("data/phase2_bootstrap_ci.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE_BOOTSTRAP")
