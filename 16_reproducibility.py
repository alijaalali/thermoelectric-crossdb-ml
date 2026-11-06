"""Record the software versions used to run the pipeline."""
import json
import importlib.metadata as m
import sys

pkgs = ["scikit-learn", "xgboost", "lightgbm", "shap", "matminer", "pymatgen",
        "numpy", "pandas", "scipy", "jarvis-tools", "openpyxl"]
versions = {}
for p in pkgs:
    try:
        versions[p] = m.version(p)
    except Exception as exc:
        versions[p] = f"not found ({exc})"

versions["python"] = sys.version.split()[0]

with open("data/phase2_software_versions.json", "w") as f:
    json.dump(versions, f, indent=2)
print(json.dumps(versions, indent=2))
