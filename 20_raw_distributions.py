# Raw-data distribution figure for each modeled property across JARVIS and ESTM.
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "figures/"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})

JARVIS_COLOR = "#2b5a8c"
ESTM_COLOR = "#b23b3b"

jarvis = json.load(open("data/jarvis_target_table_ext.json"))
estm = json.load(open("data/estm_target_table_ext.json"))


def col(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return np.array(vals, dtype=float)


panels = [
    ("Seebeck coefficient ($\\mu$V/K)", "Seebeck_uVK", "Seebeck_uVK", True, True),
    ("$\\log_{10}(\\sigma)$ ($\\log_{10}$(S/m))", "log10_cond", "log10_cond", True, True),
    ("Power factor ($\\mu$W/(m K$^2$))", "powerfact", "powerfact", True, True),
    ("$\\log_{10}(\\kappa)$ (JARVIS: electronic only;\nESTM: total, W/(m K))", "log10_kappa_elec", "log10_kappa_total", True, True),
    ("$ZT$ (ESTM only, measured)", None, "ZT_reported", False, True),
]

TITLES = ["Seebeck coefficient\n($\\mu$V/K)", "$\\log_{10}(\\sigma)$\n($\\log_{10}$(S/m))", "Power factor\n($\\mu$W/(m K$^2$))", "$\\log_{10}(\\kappa)$ (JARVIS: elec.;\nESTM: total, W/(m K))", "$ZT$ (ESTM only,\nmeasured)"]
summary = {}
fig, axes = plt.subplots(2, 3, figsize=(6.5, 5.6), constrained_layout=True)
axes = axes.flatten()

for i, (title, jkey, ekey, has_j, has_e) in enumerate(panels):
    ax = axes[i]
    entry = {"title": title}
    if has_j and jkey is not None:
        jv = col(jarvis, jkey)
        ax.hist(jv, bins=60, color=JARVIS_COLOR, alpha=0.55, density=True, hatch="//", edgecolor=JARVIS_COLOR, linewidth=0.3, label=f"JARVIS (n={len(jv)})")
        entry["jarvis"] = {"n": int(len(jv)), "mean": float(np.mean(jv)), "std": float(np.std(jv)),
                            "min": float(np.min(jv)), "max": float(np.max(jv)), "median": float(np.median(jv))}
    if has_e and ekey is not None:
        ev = col(estm, ekey)
        ax.hist(ev, bins=60, color=ESTM_COLOR, alpha=0.55, density=True, hatch="..", edgecolor=ESTM_COLOR, linewidth=0.3, label=f"ESTM (n={len(ev)})")
        entry["estm"] = {"n": int(len(ev)), "mean": float(np.mean(ev)), "std": float(np.std(ev)),
                          "min": float(np.min(ev)), "max": float(np.max(ev)), "median": float(np.median(ev))}
    ax.set_title(f"({chr(97+i)}) " + TITLES[i], fontsize=8)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, handlelength=1.2)
    summary[ekey or jkey] = entry

# panel (f): mechanism-label breakdown for ESTM Seebeck, as a tie-in to Section 2.4
mech = json.load(open("data/estm_mechanism_labels.json"))
label_by_formula = {d["formula"]: d["label"] for d in mech}
estm_by_formula = {}
for r in estm:
    estm_by_formula.setdefault(r["formula"], []).append(r)

hop_seebeck, band_seebeck = [], []
for f, lab in label_by_formula.items():
    if f not in estm_by_formula:
        continue
    rows = estm_by_formula[f]
    vals = [abs(r["Seebeck_uVK"]) for r in rows if r.get("Seebeck_uVK") is not None]
    if not vals:
        continue
    if lab == "hopping_like":
        hop_seebeck.extend(vals)
    elif lab == "band_like":
        band_seebeck.extend(vals)

ax = axes[5]
ax.hist(np.array(band_seebeck), bins=40, color=JARVIS_COLOR, alpha=0.55, density=True, hatch="//", edgecolor=JARVIS_COLOR, linewidth=0.3, label=f"band-like (n={len(band_seebeck)})")
ax.hist(np.array(hop_seebeck), bins=40, color=ESTM_COLOR, alpha=0.55, density=True, hatch="..", edgecolor=ESTM_COLOR, linewidth=0.3, label=f"hopping-like (n={len(hop_seebeck)})")
ax.set_title("(f) ESTM $|S|$ by derived\nmechanism label", fontsize=8)
ax.set_xlabel("$|S|$ ($\\mu$V/K)")
ax.set_ylabel("Density")
ax.legend(fontsize=8, handlelength=1.2)
summary["estm_seebeck_by_label"] = {
    "band_like": {"n": len(band_seebeck), "mean_abs": float(np.mean(band_seebeck)) if band_seebeck else None},
    "hopping_like": {"n": len(hop_seebeck), "mean_abs": float(np.mean(hop_seebeck)) if hop_seebeck else None},
}

for i, (title, jkey, ekey, has_j, has_e) in enumerate(panels):
    axes[i].set_xlabel(title.split(" (")[0])

for a_ in axes:
    a_.set_ylim(0, a_.get_ylim()[1] * 1.4)
from matplotlib.text import Text
mn = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip())
assert mn >= 8, mn
print("fig6 min font", mn)
fig.savefig(FIG + "fig6_raw_distributions.pdf")
fig.savefig(FIG + "fig6_raw_distributions.png", dpi=220)
print("Saved figures/fig6_raw_distributions.{pdf,png}")

with open("data/phase5_raw_distributions_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("Saved data/phase5_raw_distributions_summary.json")
