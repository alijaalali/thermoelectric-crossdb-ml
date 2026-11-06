# Extended multi-model property-prediction benchmark (same model roster and composition-grouped CV as 10) for power factor, ESTM thermal conductivity/ZT, and JARVIS electronic kappa/ZT proxy (internal-only, not cross-database-compared).
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

RNG = 42

magpie_j = pd.read_json("data/jarvis_magpie_features.json")
magpie_e = pd.read_json("data/estm_magpie_features.json")
tgt_j = pd.read_json("data/jarvis_target_table_ext.json")
tgt_e = pd.read_json("data/estm_target_table_ext.json")

feat_cols = [c for c in magpie_j.columns if c != "formula"]

j = tgt_j.merge(magpie_j, on="formula", how="inner")
e = tgt_e.merge(magpie_e, on="formula", how="inner")
j["doping_n"] = (j["doping_type"] == "n").astype(int)
e["doping_n"] = (e["doping_type"] == "n").astype(int)

X_cols = feat_cols + ["doping_n"]

for df in (j, e):
    df[X_cols] = df[X_cols].apply(pd.to_numeric, errors="coerce")

print("JARVIS rows (pre target-specific filtering):", len(j), " ESTM rows:", len(e))


def make_models(include_gp=False):
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=RNG),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=20, n_jobs=4, random_state=RNG),
        "XGBoost": xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, n_jobs=4, random_state=RNG, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=300, max_depth=-1, learning_rate=0.08, random_state=RNG, verbosity=-1, n_jobs=4),
        "MLP": MLPRegressor(hidden_layer_sizes=(32,), max_iter=150, random_state=RNG, early_stopping=True, n_iter_no_change=8),
    }
    if include_gp:
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        models["GaussianProcess"] = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=RNG)
    return models


def clean_for_target(df, target):
    out = df.dropna(subset=X_cols + [target]).copy()
    out = out[np.isfinite(out[target])]
    return out


def grouped_cv_scores(df, target, n_splits=5, include_gp=False):
    X = df[X_cols].to_numpy()
    y = df[target].to_numpy()
    groups = df["elemset"].to_numpy()
    n_groups = len(set(groups))
    k = min(n_splits, n_groups)
    if k < 2:
        return {"error": f"too few groups ({n_groups}) for CV"}
    gkf = GroupKFold(n_splits=k)
    results = {m: {"r2": [], "mae": []} for m in make_models(include_gp)}
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        models = make_models(include_gp)
        for name, model in models.items():
            print(f"    fold {fold_i+1}/{k} model={name} target={target} (n_train={len(tr_idx)}, n_test={len(te_idx)}) ...", flush=True)
            try:
                if name in ("Ridge", "MLP", "GaussianProcess"):
                    model.fit(Xtr_s, ytr)
                    pred = model.predict(Xte_s)
                else:
                    model.fit(Xtr, ytr)
                    pred = model.predict(Xte)
                results[name]["r2"].append(r2_score(yte, pred))
                results[name]["mae"].append(mean_absolute_error(yte, pred))
            except Exception as exc:
                results[name].setdefault("errors", []).append(str(exc))
    summary = {}
    for name, r in results.items():
        if r["r2"]:
            summary[name] = {
                "r2_mean": float(np.mean(r["r2"])), "r2_std": float(np.std(r["r2"])),
                "mae_mean": float(np.mean(r["mae"])), "mae_std": float(np.std(r["mae"])),
                "n_folds": len(r["r2"]),
            }
        else:
            summary[name] = {"error": r.get("errors", ["unknown"])}
    return summary


def cross_db_transfer(train_df, test_df, target):
    Xtr = train_df[X_cols].to_numpy()
    ytr = train_df[target].to_numpy()
    Xte = test_df[X_cols].to_numpy()
    yte = test_df[target].to_numpy()
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    models = make_models(include_gp=False)
    out = {}
    for name, model in models.items():
        try:
            if name in ("Ridge", "MLP"):
                model.fit(Xtr_s, ytr)
                pred = model.predict(Xte_s)
            else:
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
            out[name] = {
                "r2": float(r2_score(yte, pred)),
                "mae": float(mean_absolute_error(yte, pred)),
                "n_test": int(len(yte)),
            }
        except Exception as exc:
            out[name] = {"error": str(exc)}
    return out


results = {"in_database": {}, "cross_database": {}, "n_rows_used": {}}

# ---------------- Power factor: full cross-database treatment ----------------
target = "powerfact"
j_pf = clean_for_target(j, target)
e_pf = clean_for_target(e, target)
results["n_rows_used"][f"JARVIS__{target}"] = len(j_pf)
results["n_rows_used"][f"ESTM__{target}"] = len(e_pf)

print(f"=== In-database CV: JARVIS, target={target} (n={len(j_pf)}) ===")
results["in_database"][f"JARVIS__{target}"] = grouped_cv_scores(j_pf, target, n_splits=3, include_gp=False)
print(results["in_database"][f"JARVIS__{target}"])

print(f"=== In-database CV: ESTM, target={target} (n={len(e_pf)}) ===")
results["in_database"][f"ESTM__{target}"] = grouped_cv_scores(e_pf, target, n_splits=5, include_gp=True)
print(results["in_database"][f"ESTM__{target}"])

print(f"=== Cross-database: JARVIS -> ESTM, target={target} ===")
results["cross_database"][f"JARVIStoESTM__{target}"] = cross_db_transfer(j_pf, e_pf, target)
print(results["cross_database"][f"JARVIStoESTM__{target}"])

print(f"=== Cross-database: ESTM -> JARVIS, target={target} ===")
results["cross_database"][f"ESTMtoJARVIS__{target}"] = cross_db_transfer(e_pf, j_pf, target)
print(results["cross_database"][f"ESTMtoJARVIS__{target}"])

# ---------------- ESTM thermal conductivity (internal only, real experimental target) ----------------
target = "log10_kappa_total"
e_kappa = clean_for_target(e, target)
results["n_rows_used"][f"ESTM__{target}"] = len(e_kappa)
print(f"=== In-database CV: ESTM, target={target} (n={len(e_kappa)}) ===")
results["in_database"][f"ESTM__{target}"] = grouped_cv_scores(e_kappa, target, n_splits=5, include_gp=True)
print(results["in_database"][f"ESTM__{target}"])

# ---------------- ESTM ZT (internal only, real experimental target) ----------------
target = "ZT_reported"
e_zt = clean_for_target(e, target)
e_zt = e_zt[e_zt[target] > 0]
results["n_rows_used"][f"ESTM__{target}"] = len(e_zt)
print(f"=== In-database CV: ESTM, target={target} (n={len(e_zt)}) ===")
results["in_database"][f"ESTM__{target}"] = grouped_cv_scores(e_zt, target, n_splits=5, include_gp=True)
print(results["in_database"][f"ESTM__{target}"])

# ---------------- JARVIS electronic kappa (internal only, labeled "electronic") ----------------
target = "log10_kappa_elec"
j_kappa = clean_for_target(j, target)
results["n_rows_used"][f"JARVIS__{target}"] = len(j_kappa)
print(f"=== In-database CV: JARVIS, target={target} (n={len(j_kappa)}) ===")
results["in_database"][f"JARVIS__{target}"] = grouped_cv_scores(j_kappa, target, n_splits=3, include_gp=False)
print(results["in_database"][f"JARVIS__{target}"])

# ---------------- JARVIS electronic-ZT proxy (internal only, labeled "proxy") ----------------
target = "zt_elec_proxy"
j_zt = clean_for_target(j, target)
j_zt = j_zt[(j_zt[target] > 0) & (j_zt[target] < 1e4)]  # drop pathological divide-by-near-zero-kappa outliers
results["n_rows_used"][f"JARVIS__{target}"] = len(j_zt)
print(f"=== In-database CV: JARVIS, target={target} (n={len(j_zt)}) ===")
results["in_database"][f"JARVIS__{target}"] = grouped_cv_scores(j_zt, target, n_splits=3, include_gp=False)
print(results["in_database"][f"JARVIS__{target}"])

# ---------------- Cross-database context check, not like-for-like ----------------
# JARVIS electronic kappa -> ESTM total kappa: included only to show that transfer is
# poor because the two quantities are defined differently. It is not a primary result.
j_kappa_cdb = j_kappa.rename(columns={"log10_kappa_elec": "log10_kappa_crossdb_context"})
e_kappa_cdb = e_kappa.rename(columns={"log10_kappa_total": "log10_kappa_crossdb_context"})
target = "log10_kappa_crossdb_context"
print("=== not-like-for-like context check: JARVIS(electronic) -> ESTM(total) ===")
results["cross_database"]["JARVIStoESTM__log10_kappa_not_like_for_like"] = cross_db_transfer(j_kappa_cdb, e_kappa_cdb, target)
print(results["cross_database"]["JARVIStoESTM__log10_kappa_not_like_for_like"])

with open("data/phase4_extended_property_prediction_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE")
