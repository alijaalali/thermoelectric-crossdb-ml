"""Filter JARVIS dft_3d to real, non-missing thermoelectric records and save."""
import json
from jarvis.db.figshare import data

jv = data("dft_3d")
print("TOTAL:", len(jv))

def valid(x):
    return x is not None and x != "na" and x != ""

fields = ["n-Seebeck", "p-Seebeck", "n-powerfact", "p-powerfact", "ncond", "pcond", "nkappa", "pkappa"]
counts = {f: sum(1 for d in jv if valid(d.get(f))) for f in fields}
print("field non-missing counts:", counts)

both_seebeck = sum(1 for d in jv if valid(d.get("n-Seebeck")) and valid(d.get("p-Seebeck")))
print("both n- and p-Seebeck present:", both_seebeck)

either_seebeck = [d for d in jv if valid(d.get("n-Seebeck")) or valid(d.get("p-Seebeck"))]
print("either n- or p-Seebeck present:", len(either_seebeck))

# save the filtered raw records (either n- or p-Seebeck present) for downstream use
out = []
for d in either_seebeck:
    out.append({
        "jid": d["jid"], "formula": d["formula"], "atoms": d["atoms"],
        "spg_number": d.get("spg_number"), "spg_symbol": d.get("spg_symbol"),
        "optb88vdw_bandgap": d.get("optb88vdw_bandgap"),
        "mbj_bandgap": d.get("mbj_bandgap"),
        "formation_energy_peratom": d.get("formation_energy_peratom"),
        "ehull": d.get("ehull"),
        "n-Seebeck": d.get("n-Seebeck"), "p-Seebeck": d.get("p-Seebeck"),
        "n-powerfact": d.get("n-powerfact"), "p-powerfact": d.get("p-powerfact"),
        "ncond": d.get("ncond"), "pcond": d.get("pcond"),
        "nkappa": d.get("nkappa"), "pkappa": d.get("pkappa"),
        "avg_elec_mass": d.get("avg_elec_mass"), "avg_hole_mass": d.get("avg_hole_mass"),
        "dimensionality": d.get("dimensionality"),
    })

with open("data/jarvis_te_raw.json", "w") as f:
    json.dump(out, f)
print("saved", len(out), "records to data/jarvis_te_raw.json")
