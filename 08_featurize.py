# Magpie composition features for every unique formula in both databases, plus simple structure descriptors for JARVIS-DFT only.
import json
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure, Lattice
from matminer.featurizers.composition import ElementProperty

magpie = ElementProperty.from_preset("magpie")


def safe_comp(formula):
    try:
        return Composition(formula)
    except Exception:
        return None


def featurize_formulas(formulas):
    rows = []
    failed = []
    for f in formulas:
        c = safe_comp(f)
        if c is None or len(c.elements) == 0:
            failed.append(f)
            continue
        try:
            feats = magpie.featurize(c)
        except Exception:
            failed.append(f)
            continue
        row = dict(zip(magpie.feature_labels(), feats))
        row["formula"] = f
        rows.append(row)
    return pd.DataFrame(rows), failed


# ---- ESTM (composition only) ----
estm = pd.read_excel("data/SIMD/dataset/estm.xlsx")
estm_formulas = sorted(estm["Formula"].unique())
estm_feats, estm_failed = featurize_formulas(estm_formulas)
estm_feats.to_json("data/estm_magpie_features.json", orient="records", indent=2)
print(f"ESTM: {len(estm_formulas)} unique formulas, "
      f"{len(estm_feats)} featurized, {len(estm_failed)} failed: {estm_failed}")

# ---- JARVIS (composition + structure) ----
with open("data/jarvis_te_raw.json") as f:
    jarvis = json.load(f)
jarvis_df = pd.DataFrame(jarvis)

unique_jarvis_formulas = sorted(jarvis_df["formula"].unique())
jarvis_comp_feats, jarvis_failed = featurize_formulas(unique_jarvis_formulas)
jarvis_comp_feats.to_json("data/jarvis_magpie_features.json", orient="records", indent=2)
print(f"JARVIS: {len(unique_jarvis_formulas)} unique formulas, "
      f"{len(jarvis_comp_feats)} featurized, {len(jarvis_failed)} failed")

# Structure-based Tier-B descriptors, per JARVIS row (jid-level, since
# different polymorphs of the same formula have different structures)
struct_rows = []
n_struct_failed = 0
for rec in jarvis:
    jid = rec["jid"]
    atoms = rec["atoms"]
    try:
        lat = Lattice(atoms["lattice_mat"])
        struct = Structure(lat, atoms["elements"], atoms["coords"], coords_are_cartesian=atoms.get("cartesian", True))
        density = struct.density
        vol_per_atom = struct.volume / len(struct)
        n_sites = len(struct)
    except Exception:
        density = np.nan
        vol_per_atom = np.nan
        n_sites = len(atoms.get("elements", []))
        n_struct_failed += 1
    struct_rows.append({
        "jid": jid,
        "spg_number": rec.get("spg_number"),
        "dimensionality": rec.get("dimensionality"),
        "density_g_cm3": density,
        "volume_per_atom_A3": vol_per_atom,
        "n_sites": n_sites,
    })

struct_df = pd.DataFrame(struct_rows)
struct_df.to_json("data/jarvis_structure_features.json", orient="records", indent=2)
print(f"JARVIS structure features: {len(struct_df)} rows, {n_struct_failed} structure-build failures "
      f"(fell back to element-count n_sites for those)")

summary = {
    "estm_unique_formulas": len(estm_formulas),
    "estm_featurized": len(estm_feats),
    "estm_failed": estm_failed,
    "jarvis_unique_formulas": len(unique_jarvis_formulas),
    "jarvis_comp_featurized": len(jarvis_comp_feats),
    "jarvis_comp_failed": jarvis_failed,
    "jarvis_structure_rows": len(struct_df),
    "jarvis_structure_build_failures": n_struct_failed,
    "magpie_n_features": len(magpie.feature_labels()),
}
with open("data/phase2_featurization_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
