import json
import re
import pandas as pd
import numpy as np

estm = pd.read_excel("data/SIMD/dataset/estm.xlsx")
jarvis = pd.DataFrame(json.load(open("data/jarvis_te_raw.json")))

print("=== ESTM (experimental) ===")
print("rows:", len(estm))
print("unique formulas (raw string):", estm["Formula"].nunique())
print("duplicate exact rows (all columns):", estm.duplicated().sum())
print("duplicate (Formula, temperature) pairs:", estm.duplicated(subset=["Formula", "temperature(K)"]).sum())
print("missing values per column:\n", estm.isna().sum())
print("temperature range:", estm["temperature(K)"].min(), "-", estm["temperature(K)"].max())
print("Seebeck range (uV/K):", estm["seebeck_coefficient(μV/K)"].min(), "-", estm["seebeck_coefficient(μV/K)"].max())
print("conductivity range (S/m):", estm["electrical_conductivity(S/m)"].min(), "-", estm["electrical_conductivity(S/m)"].max())
print("ZT range:", estm["ZT"].min(), "-", estm["ZT"].max())
print("n unique references:", estm["reference"].nunique())

print("\n=== JARVIS-DFT (computational) ===")
print("rows:", len(jarvis))
print("unique formulas:", jarvis["formula"].nunique())
print("duplicate exact jid:", jarvis["jid"].duplicated().sum())
print("duplicate formula rows:", jarvis["formula"].duplicated().sum())

def normalize_formula(f):
    # strip whitespace, doping fractions in parentheses are kept as-is for an exact-match check;
    # also build a reduced-element-set version for a looser check
    return re.sub(r"\s+", "", str(f))

estm["formula_norm"] = estm["Formula"].apply(normalize_formula)
jarvis["formula_norm"] = jarvis["formula"].apply(normalize_formula)

estm_formulas = set(estm["formula_norm"])
jarvis_formulas = set(jarvis["formula_norm"])
exact_overlap = estm_formulas & jarvis_formulas
print("\n=== Composition overlap (exact normalized formula string) ===")
print("ESTM unique normalized formulas:", len(estm_formulas))
print("JARVIS unique normalized formulas:", len(jarvis_formulas))
print("exact overlap:", len(exact_overlap))
print("sample overlap:", list(exact_overlap)[:10])

# element-set based overlap (looser): same set of elements regardless of stoichiometry/doping
def element_set(f):
    # crude element extraction, strips digits/parens/dots/fractions
    els = re.findall(r"[A-Z][a-z]?", re.sub(r"[\d().,]", "", str(f)))
    return frozenset(els)

estm["elemset"] = estm["Formula"].apply(element_set)
jarvis["elemset"] = jarvis["formula"].apply(element_set)
estm_elemsets = set(estm["elemset"])
jarvis_elemsets = set(jarvis["elemset"])
elem_overlap = estm_elemsets & jarvis_elemsets
print("\n=== Element-set overlap (same elements, any stoichiometry) ===")
print("ESTM unique element-sets:", len(estm_elemsets))
print("JARVIS unique element-sets:", len(jarvis_elemsets))
print("overlap:", len(elem_overlap))

# literature reference compounds
known_hopping = ["LaMnO3", "La0.7Sr0.3MnO3", "LaCoO3", "La0.8Sr0.2CoO3", "Fe3O4", "TiO2",
                  "Ca3Co4O9", "NaCo2O4", "SrTiO3", "La2CuO4", "LaNiO3"]
known_band = ["Bi2Te3", "PbTe", "SiGe", "Si", "Ge", "CoSb3", "SnSe", "Mg2Si", "ZnO", "Half-Heusler"]

print("\n=== Literature reference compounds in ESTM (substring match on formula) ===")
for name in known_hopping + known_band:
    key = re.sub(r"[\d.]", "", name).replace("La", "La")  # keep as-is, just substring search
    hits = estm[estm["Formula"].astype(str).str.contains(re.escape(name.split("0")[0].split(".")[0]) if any(c.isdigit() for c in name) else re.escape(name), case=False, na=False, regex=True)]
    print(f"  {name}: {len(hits)} rows (substring match)")

print("\n=== element-based detection: Mn, Co (hopping-relevant 3d) vs classic band TE (Bi2Te3-family, PbTe-family) ===")
mn_co_containing = estm[estm["Formula"].astype(str).str.contains("Mn|Co", case=False, na=False, regex=True)]
print("rows containing Mn or Co:", len(mn_co_containing), "unique formulas:", mn_co_containing["Formula"].nunique())
bite_pbte = estm[estm["Formula"].astype(str).str.contains("Bi.*Te|Pb.*Te|SiGe", case=False, na=False, regex=True)]
print("rows containing Bi-Te / Pb-Te / SiGe family:", len(bite_pbte), "unique formulas:", bite_pbte["Formula"].nunique())

out = {
    "estm_rows": int(len(estm)), "estm_unique_formulas": int(estm["Formula"].nunique()),
    "estm_dup_rows": int(estm.duplicated().sum()),
    "jarvis_rows": int(len(jarvis)), "jarvis_unique_formulas": int(jarvis["formula"].nunique()),
    "exact_formula_overlap": int(len(exact_overlap)),
    "element_set_overlap": int(len(elem_overlap)),
    "mn_co_rows": int(len(mn_co_containing)), "bite_pbte_rows": int(len(bite_pbte)),
}
with open("data/phase1_eda_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved data/phase1_eda_summary.json")
