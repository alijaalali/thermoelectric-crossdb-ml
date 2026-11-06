# Core multi-model property-prediction benchmark: in-database composition-grouped CV and cross-database zero-shot transfer for Seebeck and log10(conductivity).
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

print("JARVIS usable rows:", len(j), " ESTM usable rows:", len(e))


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


def grouped_cv_scores(df, target, n_splits=5, include_gp=False):
    X = df[X_cols].to_numpy()
    y = df[target].to_numpy()
    groups = df["elemset"].to_numpy()
    n_groups = len(set(groups))
    k = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=k)
    results = {m: {"r2": [], "mae": []} for m in make_models(include_gp)}
    for fold_i, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        models = make_models(include_gp)
        for name, model in models.items():
            print(f"    fold {fold_i+1}/{k} model={name} (n_train={len(tr_idx)}, n_test={len(te_idx)}) ...", flush=True)
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


results = {"in_database": {}, "cross_database": {}}

targets = ["Seebeck_uVK", "log10_cond"]
for target in targets:
    print(f"=== In-database CV: JARVIS, target={target} ===")
    results["in_database"][f"JARVIS__{target}"] = grouped_cv_scores(j, target, n_splits=3, include_gp=False)
    print(results["in_database"][f"JARVIS__{target}"])

    print(f"=== In-database CV: ESTM, target={target} ===")
    results["in_database"][f"ESTM__{target}"] = grouped_cv_scores(e, target, n_splits=5, include_gp=True)
    print(results["in_database"][f"ESTM__{target}"])

    print(f"=== Cross-database: JARVIS -> ESTM, target={target} ===")
    results["cross_database"][f"JARVIStoESTM__{target}"] = cross_db_transfer(j, e, target)
    print(results["cross_database"][f"JARVIStoESTM__{target}"])

    print(f"=== Cross-database: ESTM -> JARVIS, target={target} ===")
    results["cross_database"][f"ESTMtoJARVIS__{target}"] = cross_db_transfer(e, j, target)
    print(results["cross_database"][f"ESTMtoJARVIS__{target}"])

with open("data/phase2_property_prediction_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("DONE")
