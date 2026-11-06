# Target-shift and feature-shift diagnosis for cross-database transfer (KS test on target distributions; nearest-neighbor feature-space distance check).
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

tgt_j = pd.read_json("data/jarvis_target_table.json")
tgt_e = pd.read_json("data/estm_target_table.json")
magpie_j = pd.read_json("data/jarvis_magpie_features.json")
magpie_e = pd.read_json("data/estm_magpie_features.json")

feat_cols = [c for c in magpie_j.columns if c != "formula"]

j = tgt_j.merge(magpie_j, on="formula", how="inner")
e = tgt_e.merge(magpie_e, on="formula", how="inner")
for df in (j, e):
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=feat_cols, inplace=True)

results = {"target_shift": {}, "feature_shift": {}}

for target in ["Seebeck_uVK", "log10_cond"]:
    xj = j[target].dropna().to_numpy()
    xe = e[target].dropna().to_numpy()
    ks = stats.ks_2samp(xj, xe)
    results["target_shift"][target] = {
        "jarvis_mean": float(np.mean(xj)), "jarvis_std": float(np.std(xj)),
        "jarvis_median": float(np.median(xj)),
        "estm_mean": float(np.mean(xe)), "estm_std": float(np.std(xe)),
        "estm_median": float(np.median(xe)),
        "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
    }

# Feature-space shift: standardize on JARVIS, transform both
scaler = StandardScaler().fit(j[feat_cols].to_numpy())
Xj = scaler.transform(j[feat_cols].to_numpy())
Xe = scaler.transform(e[feat_cols].to_numpy())

per_feature_mean_shift = np.abs(Xe.mean(axis=0) - Xj.mean(axis=0))
results["feature_shift"]["n_features"] = len(feat_cols)
results["feature_shift"]["mean_abs_standardized_mean_shift"] = float(per_feature_mean_shift.mean())
results["feature_shift"]["top10_shifted_features"] = [
    feat_cols[i] for i in np.argsort(-per_feature_mean_shift)[:10]
]
results["feature_shift"]["top10_shift_values"] = [
    float(per_feature_mean_shift[i]) for i in np.argsort(-per_feature_mean_shift)[:10]
]

# Nearest-neighbor distance: ESTM -> nearest JARVIS, vs JARVIS internal baseline
nn_j = NearestNeighbors(n_neighbors=2).fit(Xj)  # 2 so self-match excluded for the baseline case
dist_e_to_j, _ = NearestNeighbors(n_neighbors=1).fit(Xj).kneighbors(Xe)
dist_j_to_j, _ = nn_j.kneighbors(Xj)  # first neighbor is self (dist 0), take 2nd column
dist_j_to_j_baseline = dist_j_to_j[:, 1]

results["feature_shift"]["estm_to_jarvis_nn_distance_median"] = float(np.median(dist_e_to_j))
results["feature_shift"]["estm_to_jarvis_nn_distance_mean"] = float(np.mean(dist_e_to_j))
results["feature_shift"]["jarvis_internal_nn_distance_median_baseline"] = float(np.median(dist_j_to_j_baseline))
results["feature_shift"]["jarvis_internal_nn_distance_mean_baseline"] = float(np.mean(dist_j_to_j_baseline))
results["feature_shift"]["ratio_estm_to_jarvis_vs_internal_baseline"] = float(
    np.median(dist_e_to_j) / np.median(dist_j_to_j_baseline)
)

with open("data/phase2_shift_diagnosis.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
