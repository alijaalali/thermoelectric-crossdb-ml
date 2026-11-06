"""Pull real JARVIS-DFT dft_3d data and inspect thermoelectric-relevant fields."""
import json
from jarvis.db.figshare import data

jv = data("dft_3d")
print("TOTAL dft_3d entries:", len(jv))

keys = set()
for d in jv[:50]:
    keys.update(d.keys())
print("sample keys:", sorted(keys))

# Look for thermoelectric-relevant fields
te_keys = [k for k in keys if any(s in k.lower() for s in
           ["seebeck", "boltztrap", "powerfact", "n-seebeck", "p-seebeck",
            "kappa", "thermal_cond", "epsx", "max_seebeck", "max_powerfact"])]
print("TE-relevant key candidates:", te_keys)
