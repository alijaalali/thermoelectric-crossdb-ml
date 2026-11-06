# Screening demonstration: use the trained composition-only mechanism classifier to
# flag candidate small-polaron-hopping thermoelectric compounds among JARVIS-DFT
# formulas that have no ESTM entry, i.e. compounds outside the labeled training set.
import json
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

labels = pd.read_json("data/estm_mechanism_labels.json")
magpie_e = pd.read_json("data/estm_magpie_features.json")
magpie_j = pd.read_json("data/jarvis_magpie_features.json")

df = labels.merge(magpie_e, on="formula", how="inner")
df = df[df["label"].isin(["band_like", "hopping_like"])].copy()
feat_cols = [c for c in magpie_e.columns if c != "formula"]
df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")
df.dropna(subset=feat_cols, inplace=True)

X = df[feat_cols].to_numpy()
y = (df["label"] == "hopping_like").astype(int).to_numpy()

clf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=4)
clf.fit(X, y)

# Candidate pool: JARVIS formulas with no ESTM entry (no temperature-dependent
# measurement in this experimental database). A positive prediction can be checked
# against the literature and is a candidate for experimental characterization.
estm_formulas = set(pd.read_json("data/estm_target_table.json")["formula"].unique())
candidates = magpie_j[~magpie_j["formula"].isin(estm_formulas)].copy()
candidates[feat_cols] = candidates[feat_cols].apply(pd.to_numeric, errors="coerce")
candidates.dropna(subset=feat_cols, inplace=True)
print(f"Candidate pool (JARVIS formulas with no ESTM entry): {len(candidates)}")

proba = clf.predict_proba(candidates[feat_cols].to_numpy())[:, 1]
candidates["hopping_proba"] = proba
candidates_sorted = candidates.sort_values("hopping_proba", ascending=False)

# Exclude formulas with the same element set as the hopping families in the training
# set (Ca-Co-O, La-Co-O, Na-Co-O cobaltites; La/Ca-Ti-O), to avoid near-duplicates of
# the training compounds.
known_family_elems = {frozenset(["Ca", "Co", "O"]), frozenset(["La", "Co", "O"]),
                       frozenset(["Na", "Co", "O"]), frozenset(["Ca", "Ti", "O"]),
                       frozenset(["La", "Ca", "Ti", "O"])}


def elemset(f):
    return frozenset(re.findall(r"[A-Z][a-z]?", re.sub(r"[0-9().,]", "", str(f))))


candidates_sorted["elemset"] = candidates_sorted["formula"].apply(elemset)
novel = candidates_sorted[~candidates_sorted["elemset"].isin(known_family_elems)]

top20 = novel.head(20)[["formula", "hopping_proba"]].to_dict("records")
print("Top 20 novel candidates:")
for r in top20:
    print(f"  {r['formula']}: p={r['hopping_proba']:.3f}")

out = {
    "n_candidate_pool": int(len(candidates)),
    "n_novel_after_known_family_exclusion": int(len(novel)),
    "top20_candidates": top20,
}
with open("data/phase5_screening_demo.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE")
