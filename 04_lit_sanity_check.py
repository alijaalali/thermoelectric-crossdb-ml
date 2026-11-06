import pandas as pd
import re

estm = pd.read_excel("data/SIMD/dataset/estm.xlsx")

def elems(f):
    return set(re.findall(r"[A-Z][a-z]?", re.sub(r"[0-9().,]", "", str(f))))

estm["elemset"] = estm["Formula"].apply(elems)

groups = {
    "La-Mn-O manganite (known polaron-hopping)": {"La", "Mn", "O"},
    "La-Co-O cobaltite (known polaron-hopping)": {"La", "Co", "O"},
    "Ca-Co-O cobaltite (known polaron-hopping)": {"Ca", "Co", "O"},
    "Na-Co-O cobaltite (known polaron-hopping)": {"Na", "Co", "O"},
    "Fe-O (magnetite family, known polaron-hopping)": {"Fe", "O"},
    "Ti-O (titanate, polaron-relevant)": {"Ti", "O"},
    "Sr-Ti-O (polaron-relevant)": {"Sr", "Ti", "O"},
    "Bi-Te (known band transport)": {"Bi", "Te"},
    "Pb-Te (known band transport)": {"Pb", "Te"},
    "Si-Ge (known band transport)": {"Si", "Ge"},
    "Co-Sb skutterudite (known band transport)": {"Co", "Sb"},
}

results = {}
for name, req in groups.items():
    mask = estm["elemset"].apply(lambda s: req.issubset(s))
    sub = estm[mask]
    n_rows, n_formulas = len(sub), sub["Formula"].nunique()
    results[name] = {"rows": int(n_rows), "unique_formulas": int(n_formulas)}
    print(f"{name}: {n_rows} rows, {n_formulas} unique formulas")
    if n_formulas > 0 and n_formulas <= 8:
        print("   formulas:", sorted(sub["Formula"].unique().tolist()))

import json
with open("data/phase1_lit_sanity_check.json", "w") as f:
    json.dump(results, f, indent=2)
