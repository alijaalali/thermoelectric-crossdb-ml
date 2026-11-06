# Leave-one-chemical-family-out robustness check: holds out every formula containing a given element from training, to test whether the mechanism classifier generalizes beyond memorizing known chemical families.
import json
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

labels = pd.read_json("data/estm_mechanism_labels.json")
magpie_e = pd.read_json("data/estm_magpie_features.json")

df = labels.merge(magpie_e, on="formula", how="inner")
df = df[df["label"].isin(["band_like", "hopping_like"])].copy()
df["y"] = (df["label"] == "hopping_like").astype(int)

feat_cols = [c for c in magpie_e.columns if c != "formula"]
df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
df.dropna(subset=feat_cols, inplace=True)

elem_re = re.compile(r"[A-Z][a-z]?")


def formula_elements(f):
    return set(elem_re.findall(re.sub(r"[0-9().,]", "", str(f))))


df["elements"] = df["formula"].apply(formula_elements)

print(f"Family-holdout dataset: {len(df)} formulas "
      f"({df['y'].sum()} hopping_like, {(~df['y'].astype(bool)).sum()} band_like)")

# Candidate held-out elements: present with reasonable counts in both classes,
# chemically meaningful for polaron-hopping chemistry (transition metals /
# alkaline-earth common in cobaltite/titanate-type hopping thermoelectrics).
candidates = ["Co", "Ca", "O", "Ni", "Mg"]

results = {}
for elem in candidates:
    mask_holdout = df["elements"].apply(lambda es: elem in es)
    n_holdout = int(mask_holdout.sum())
    n_holdout_hop = int(df.loc[mask_holdout, "y"].sum())
    n_holdout_band = int(n_holdout - n_holdout_hop)
    n_train = int((~mask_holdout).sum())

    if n_holdout < 5 or n_holdout_hop == 0 or n_holdout_band == 0:
        results[elem] = {
            "skipped": True,
            "reason": "insufficient held-out formulas or only one class present",
            "n_holdout": n_holdout, "n_holdout_hopping_like": n_holdout_hop,
            "n_holdout_band_like": n_holdout_band,
        }
        print(f"[{elem}] SKIPPED: n_holdout={n_holdout}, hop={n_holdout_hop}, band={n_holdout_band}")
        continue

    Xtr = df.loc[~mask_holdout, feat_cols].to_numpy()
    ytr = df.loc[~mask_holdout, "y"].to_numpy()
    Xte = df.loc[mask_holdout, feat_cols].to_numpy()
    yte = df.loc[mask_holdout, "y"].to_numpy()

    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  random_state=42, n_jobs=4)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]

    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, zero_division=0))
    auc = float(roc_auc_score(yte, proba)) if len(set(yte)) > 1 else None

    results[elem] = {
        "skipped": False,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "n_holdout_hopping_like": n_holdout_hop,
        "n_holdout_band_like": n_holdout_band,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
    }
    print(f"[{elem}] train n={n_train}, holdout n={n_holdout} "
          f"({n_holdout_hop} hop / {n_holdout_band} band) -> "
          f"accuracy={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}" if auc is not None else
          f"[{elem}] train n={n_train}, holdout n={n_holdout} -> accuracy={acc:.3f}, f1={f1:.3f}, auc=NA")

valid = {k: v for k, v in results.items() if not v.get("skipped", False)}
if valid:
    mean_auc = float(np.mean([v["auc"] for v in valid.values() if v["auc"] is not None]))
    mean_acc = float(np.mean([v["accuracy"] for v in valid.values()]))
else:
    mean_auc = None
    mean_acc = None

out = {
    "method": "leave-one-chemical-family-out: remove all formulas containing "
               "element E from training, train RandomForestClassifier (same "
               "hyperparameters as main benchmark, scripts/12) on the "
               "remainder, test only on held-out E-containing formulas",
    "candidate_elements": candidates,
    "per_element": results,
    "mean_auc_across_valid_elements": mean_auc,
    "mean_accuracy_across_valid_elements": mean_acc,
    "n_valid_elements": len(valid),
    "interpretation": "If AUC remains well above chance (0.5) on elements/"
                       "families held out entirely from training, the "
                       "classifier is not simply memorizing which chemical "
                       "families were labeled hopping_like during training.",
}
with open("data/phase3_family_holdout_check.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
