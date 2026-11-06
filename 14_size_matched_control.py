# Size-matched control checking whether the mechanism classifier's apparent class separation survives downsampling the majority class.
import json
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

RNG = np.random.default_rng(42)

labels = pd.read_json("data/estm_mechanism_labels.json")
magpie_e = pd.read_json("data/estm_magpie_features.json")
df = labels.merge(magpie_e, on="formula", how="inner")
df = df[df["label"].isin(["band_like", "hopping_like"])].copy()
feat_cols = [c for c in magpie_e.columns if c != "formula"]
df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
df.dropna(subset=feat_cols, inplace=True)
df["elemset"] = df["formula"].apply(lambda f: "|".join(sorted(set(re.findall(r"[A-Z][a-z]?", re.sub(r"[0-9().,]", "", str(f)))))))

hopping = df[df["label"] == "hopping_like"]
band = df[df["label"] == "band_like"]
n_hop = len(hopping)
n_band = len(band)
print(f"hopping_like={n_hop}, band_like={n_band}")


def eval_once(sub_df):
    X = sub_df[feat_cols].to_numpy()
    y = (sub_df["label"] == "hopping_like").astype(int).to_numpy()
    groups = sub_df["elemset"].to_numpy()
    k = min(5, len(set(groups)))
    if k < 2:
        return None
    gkf = GroupKFold(n_splits=k)
    accs, f1s, aucs = [], [], []
    for tr_idx, te_idx in gkf.split(X, y, groups):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        # Same RF settings as 12_mechanism_classifier.py, so the full-data baseline
        # reproduces the AUC reported there.
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=4)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]
        accs.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, zero_division=0))
        aucs.append(roc_auc_score(yte, proba))
    if not accs:
        return None
    return {"accuracy": float(np.mean(accs)), "f1": float(np.mean(f1s)), "auc": float(np.mean(aucs))}


# Full-imbalanced-data baseline
full_result = eval_once(df)
print("Full (imbalanced) RF grouped-CV:", full_result)

# Size-matched control: downsample band_like to n_hop, repeat 20x
n_repeats = 20
matched_results = []
for i in range(n_repeats):
    band_sample = band.sample(n=n_hop, random_state=int(RNG.integers(0, 1_000_000)))
    sub = pd.concat([hopping, band_sample], ignore_index=True)
    r = eval_once(sub)
    if r is not None:
        matched_results.append(r)

accs = [r["accuracy"] for r in matched_results]
f1s = [r["f1"] for r in matched_results]
aucs = [r["auc"] for r in matched_results]

summary = {
    "n_hopping_like": int(n_hop),
    "n_band_like": int(n_band),
    "full_imbalanced_result": full_result,
    "size_matched_n_repeats": len(matched_results),
    "size_matched_accuracy_median": float(np.median(accs)) if accs else None,
    "size_matched_accuracy_5_95pct": [float(np.percentile(accs, 5)), float(np.percentile(accs, 95))] if accs else None,
    "size_matched_f1_median": float(np.median(f1s)) if f1s else None,
    "size_matched_auc_median": float(np.median(aucs)) if aucs else None,
    "size_matched_auc_5_95pct": [float(np.percentile(aucs, 5)), float(np.percentile(aucs, 95))] if aucs else None,
    "interpretation": "If size-matched AUC/accuracy remain well above chance (0.5), the "
                       "classifier's separation of hopping_like vs band_like compounds is "
                       "not an artifact of class-size imbalance.",
}
with open("data/phase2_size_matched_control.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
