# Validates the derived mechanism labels from 05 against known literature compound families (agreement-rate check, not a training signal).
import json
import re
import pandas as pd

estm = pd.read_excel("data/SIMD/dataset/estm.xlsx")
labels = pd.read_json("data/estm_mechanism_labels.json")


def elems(f):
    return set(re.findall(r"[A-Z][a-z]?", re.sub(r"[0-9().,]", "", str(f))))


hopping_groups = {
    "La-Co-O": {"La", "Co", "O"},
    "Ca-Co-O": {"Ca", "Co", "O"},
    "Fe-O": {"Fe", "O"},
    "Ti-O": {"Ti", "O"},
}
band_groups = {
    "Bi-Te": {"Bi", "Te"},
    "Pb-Te": {"Pb", "Te"},
    "Si-Ge": {"Si", "Ge"},
    "Co-Sb": {"Co", "Sb"},
}

estm["elemset"] = estm["Formula"].apply(elems)

hopping_formulas = set()
for req in hopping_groups.values():
    hopping_formulas |= set(estm.loc[estm["elemset"].apply(lambda s: req.issubset(s)), "Formula"].unique())

band_formulas = set()
for req in band_groups.values():
    band_formulas |= set(estm.loc[estm["elemset"].apply(lambda s: req.issubset(s)), "Formula"].unique())

# a formula could in principle match both element-set families; remove any such overlap
overlap = hopping_formulas & band_formulas
hopping_formulas -= overlap
band_formulas -= overlap

label_map = dict(zip(labels["formula"], labels["label"]))


def eval_group(name, formulas, expected):
    rows = []
    for f in sorted(formulas):
        lab = label_map.get(f, "not_in_labels")
        rows.append({"formula": f, "derived_label": lab})
    n_total = len(rows)
    n_agree = sum(1 for r in rows if r["derived_label"] == expected)
    n_disagree_opposite = sum(1 for r in rows if r["derived_label"] == ("band_like" if expected == "hopping_like" else "hopping_like"))
    n_ambiguous_or_excluded = n_total - n_agree - n_disagree_opposite
    return {
        "expected": expected,
        "n_formulas": n_total,
        "n_agree": n_agree,
        "n_disagree_opposite_label": n_disagree_opposite,
        "n_ambiguous_or_excluded": n_ambiguous_or_excluded,
        "agreement_rate_among_labeled": (n_agree / (n_agree + n_disagree_opposite)) if (n_agree + n_disagree_opposite) > 0 else None,
        "detail": rows,
    }


result = {
    "hopping_side": eval_group("hopping", hopping_formulas, "hopping_like"),
    "band_side": eval_group("band", band_formulas, "band_like"),
    "note": "overlap formulas appearing in both element-set families were excluded from this check "
            f"({len(overlap)} formulas): {sorted(overlap)}",
}

with open("data/phase2_mechanism_validation.json", "w") as f:
    json.dump(result, f, indent=2)

for side in ["hopping_side", "band_side"]:
    r = result[side]
    print(f"{side}: n={r['n_formulas']}, agree={r['n_agree']}, "
          f"opposite={r['n_disagree_opposite_label']}, "
          f"ambiguous/excluded={r['n_ambiguous_or_excluded']}, "
          f"agreement_rate_among_labeled={r['agreement_rate_among_labeled']}")
print("overlap excluded:", len(overlap))
