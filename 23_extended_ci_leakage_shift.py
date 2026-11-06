# Grouped-bootstrap CIs, leakage-robustness, and target-shift diagnosis for the extended target set (power factor, thermal conductivity, ZT).
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy import stats

RNG = np.random.default_rng(42)
N_BOOT = 500

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


def clean(df, target):
    out = df.dropna(subset=X_cols + [target]).copy()
    out = out[np.isfinite(out[target])]
    return out


def grouped_bootstrap_r2(train_df, test_df, target, n_boot=N_BOOT):
    Xtr, ytr = train_df[X_cols].to_numpy(), train_df[target].to_numpy()
    model = RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=42)
    model.fit(Xtr, ytr)
    test_df = test_df.reset_index(drop=True)
    Xte_full = test_df[X_cols].to_numpy()
    yte_full = test_df[target].to_numpy()
    pred_full = model.predict(Xte_full)
    test_groups = test_df["elemset"].unique()
    group_to_idx = test_df.groupby("elemset").apply(lambda g: g.index.to_numpy())
    boot_r2 = []
    for _ in range(n_boot):
        sampled_groups = RNG.choice(test_groups, size=len(test_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        try:
            boot_r2.append(r2_score(yte_full[idx], pred_full[idx]))
        except Exception:
            continue
    boot_r2 = np.array(boot_r2)
    point_r2 = r2_score(yte_full, pred_full)
    return {
        "point_r2": float(point_r2),
        "ci_lower": float(np.percentile(boot_r2, 2.5)),
        "ci_upper": float(np.percentile(boot_r2, 97.5)),
        "n_boot_valid": int(len(boot_r2)),
    }


def grouped_bootstrap_r2_internal(df, target, n_boot=N_BOOT, test_frac=0.3):
    # internal (in-database) version: fit on 70% of groups, bootstrap-CI the R2 on held-out 30%
    df = df.reset_index(drop=True)
    groups = df["elemset"].unique()
    RNG2 = np.random.default_rng(7)
    RNG2.shuffle(groups)
    n_test_groups = max(1, int(len(groups) * test_frac))
    test_groups_fixed = set(groups[:n_test_groups])
    train_mask = ~df["elemset"].isin(test_groups_fixed)
    test_mask = df["elemset"].isin(test_groups_fixed)
    return grouped_bootstrap_r2(df[train_mask], df[test_mask], target, n_boot)


results = {}

# ---------------- Power factor: cross-database bootstrap CI (RF, fast representative) ----------------
target = "powerfact"
j_pf, e_pf = clean(j, target), clean(e, target)
print("Power factor JARVIS->ESTM bootstrap...")
results["powerfact_JARVIStoESTM"] = grouped_bootstrap_r2(j_pf, e_pf, target)
print(results["powerfact_JARVIStoESTM"])
print("Power factor ESTM->JARVIS bootstrap...")
results["powerfact_ESTMtoJARVIS"] = grouped_bootstrap_r2(e_pf, j_pf, target)
print(results["powerfact_ESTMtoJARVIS"])

# ---------------- ESTM thermal conductivity / ZT: internal bootstrap CI ----------------
target = "log10_kappa_total"
e_kappa = clean(e, target)
print("ESTM log10(kappa_total) internal bootstrap...")
results["ESTM_log10_kappa_total_internal"] = grouped_bootstrap_r2_internal(e_kappa, target)
print(results["ESTM_log10_kappa_total_internal"])

target = "ZT_reported"
e_zt = clean(e, target)
e_zt = e_zt[e_zt[target] > 0]
print("ESTM ZT internal bootstrap...")
results["ESTM_ZT_internal"] = grouped_bootstrap_r2_internal(e_zt, target)
print(results["ESTM_ZT_internal"])

# ---------------- Leakage robustness: power factor, excluding the 33 exact-formula overlaps ----------------
exact_overlap = set(j_pf["formula"]).intersection(set(e_pf["formula"]))
j_pf_noleak = j_pf[~j_pf["formula"].isin(exact_overlap)]
e_pf_noleak = e_pf[~e_pf["formula"].isin(exact_overlap)]


def transfer_r2(train_df, test_df, target):
    Xtr, ytr = train_df[X_cols].to_numpy(), train_df[target].to_numpy()
    Xte, yte = test_df[X_cols].to_numpy(), test_df[target].to_numpy()
    model = RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return float(r2_score(yte, pred))


results["leakage_robustness_powerfact"] = {
    "n_overlap_excluded": len(exact_overlap),
    "JARVIStoESTM_with_overlap": transfer_r2(j_pf, e_pf, "powerfact"),
    "JARVIStoESTM_no_overlap": transfer_r2(j_pf_noleak, e_pf_noleak, "powerfact"),
    "ESTMtoJARVIS_with_overlap": transfer_r2(e_pf, j_pf, "powerfact"),
    "ESTMtoJARVIS_no_overlap": transfer_r2(e_pf_noleak, j_pf_noleak, "powerfact"),
}
print(results["leakage_robustness_powerfact"])

# ---------------- Target-shift diagnosis: power factor distribution comparison ----------------
ks = stats.ks_2samp(j_pf["powerfact"].to_numpy(), e_pf["powerfact"].to_numpy())
results["shift_diagnosis_powerfact"] = {
    "jarvis_median": float(j_pf["powerfact"].median()),
    "jarvis_mean": float(j_pf["powerfact"].mean()),
    "estm_median": float(e_pf["powerfact"].median()),
    "estm_mean": float(e_pf["powerfact"].mean()),
    "ks_statistic": float(ks.statistic),
    "ks_pvalue": float(ks.pvalue),
}
print(results["shift_diagnosis_powerfact"])

with open("data/phase4_extended_ci_leakage_shift.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("DONE")
