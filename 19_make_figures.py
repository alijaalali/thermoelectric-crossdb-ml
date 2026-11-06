import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

FIG = "figures/"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})


def check_fonts(fig, name):
    from matplotlib.text import Text
    mn = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip())
    assert mn >= 8, (name, mn)
    print(name, "min font", mn)


prop = json.load(open("data/phase2_property_prediction_results.json"))
prop_ext = json.load(open("data/phase4_extended_property_prediction_results.json"))
mech = json.load(open("data/phase2_mechanism_classification.json"))
sweep = json.load(open("data/phase2_threshold_sweep.json"))
labsum = json.load(open("data/phase2_mechanism_labeling_summary.json"))
sizectrl = json.load(open("data/phase2_size_matched_control.json"))
preds = json.load(open("data/phase3_figure_predictions.json"))
eda = json.load(open("data/phase1_eda_summary.json"))

# ---------- Figure 2: threshold sweep + label counts ----------
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6), constrained_layout=True)
ax = axes[0]
settings = [f"m{s['min_points']}r{s['rho_thresh']}" for s in sweep]
hop_rate = [s["hopping_validation_rate"] for s in sweep]
band_rate = [s["band_validation_rate"] for s in sweep]
x = np.arange(len(sweep))
ax.plot(x, hop_rate, "o-", color="#b23b3b", markersize=4, linewidth=1.2, label="Hopping-side agreement")
ax.plot(x, band_rate, "s--", color="#2b5a8c", markersize=4, linewidth=1.2, label="Band-side agreement")
ax.set_xticks(x)
ax.set_xticklabels([f"{s['min_points']},{s['rho_thresh']}" for s in sweep], rotation=60, ha="right", fontsize=8)
ax.set_xlabel("(min. T points, $\\rho$ threshold)")
ax.set_ylabel("Agreement with literature\ncompounds")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.74))
ax.set_title("(a) Threshold-sweep stability\nof validation agreement")

ax = axes[1]
cats = ["band_like", "hopping_like", "ambiguous_no_significant_trend", "excluded_insufficient_data"]
labels_nice = ["Band-\nlike", "Hopping-\nlike", "Ambig-\nuous", "Excluded\n(few T pts)"]
vals = [labsum["label_counts"].get(c, 0) + labsum["label_counts"].get("ambiguous_mixed_signal", 0) if c == "ambiguous_no_significant_trend" else labsum["label_counts"].get(c, 0) for c in cats]
colors = ["#2b5a8c", "#b23b3b", "#999999", "#cccccc"]
ax.bar(labels_nice, vals, color=colors, edgecolor="black", linewidth=0.8)
for i, v in enumerate(vals):
    ax.text(i, v + 8, str(v), ha="center", fontsize=8)
ax.set_ylabel("Number of ESTM compounds")
ax.set_title("(b) Default-threshold label\ncounts (n=880)")
ax.set_ylim(0, max(vals) * 1.12)
check_fonts(fig, "fig2")
plt.savefig(FIG + "fig2_mechanism_labeling.pdf")
plt.savefig(FIG + "fig2_mechanism_labeling.png", dpi=220)
plt.close()

# ---------- Figure 3: parity plots ----------
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4), constrained_layout=True)
d = preds["jarvis_indb_seebeck_parity"]
ax = axes[0]
yt, yp = np.array(d["y_true"]), np.array(d["y_pred"])
ax.scatter(yt, yp, s=4, alpha=0.35, color="#2b5a8c", edgecolor="none")
lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
ax.plot(lims, lims, "k--", linewidth=1)
ax.set_xlabel("True Seebeck ($\\mu$V/K)")
ax.set_ylabel("Predicted Seebeck ($\\mu$V/K)")
ax.set_title(f"(a) JARVIS-DFT in-database\n(RF), $R^2$={d['r2_full_fold']:.3f}")

d = preds["jarvis_to_estm_seebeck_parity"]
ax = axes[1]
yt, yp = np.array(d["y_true"]), np.array(d["y_pred"])
ax.scatter(yt, yp, s=6, alpha=0.5, color="#b23b3b", edgecolor="none")
lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
ax.plot(lims, lims, "k--", linewidth=1)
ax.set_xlabel("True Seebeck ($\\mu$V/K)")
ax.set_ylabel("Predicted Seebeck ($\\mu$V/K)")
ax.set_title(f"(b) JARVIS$\\to$ESTM zero-shot\ntransfer (Ridge), $R^2$={d['r2']:.3f}")
check_fonts(fig, "fig3")
plt.savefig(FIG + "fig3_parity_plots.pdf")
plt.savefig(FIG + "fig3_parity_plots.png", dpi=220)
plt.close()

# ---------- Figure 4: model comparison bar chart (in-database R2) ----------
# Merge the original (Seebeck/conductivity) and extended (power factor)
# in-database result dicts into one lookup so all three properties can be
# plotted together; thermal conductivity and ZT are excluded because the JARVIS
# kappa column is electronic-only and does not match ESTM's total thermal
# conductivity.
prop_merged = {**prop["in_database"], **prop_ext["in_database"]}
fig, ax = plt.subplots(figsize=(5.5, 3.6), constrained_layout=True)
blocks = ["JARVIS__Seebeck_uVK", "ESTM__Seebeck_uVK", "JARVIS__log10_cond", "ESTM__log10_cond", "JARVIS__powerfact", "ESTM__powerfact"]
block_labels = ["JARVIS\nS", "ESTM\nS", "JARVIS\nlog$_{10}\\sigma$", "ESTM\nlog$_{10}\\sigma$", "JARVIS\nPF", "ESTM\nPF"]
models_order = ["Ridge", "RandomForest", "XGBoost", "LightGBM", "MLP", "GaussianProcess"]
colors_m = {"Ridge": "#999999", "RandomForest": "#2b5a8c", "XGBoost": "#3f7a4a", "LightGBM": "#c77d2e", "MLP": "#9c3d57", "GaussianProcess": "#6a4c93"}
hatch_m = {"Ridge": "", "RandomForest": "//", "XGBoost": "\\\\", "LightGBM": "xx", "MLP": "..", "GaussianProcess": "++"}
width = 0.135
xbase = np.arange(len(blocks))
for i, m in enumerate(models_order):
    vals = []
    for b in blocks:
        v = prop_merged[b].get(m, {}).get("r2_mean", np.nan)
        vals.append(v)
    ax.bar(xbase + (i - 2.5) * width, vals, width, label=m, color=colors_m[m], hatch=hatch_m[m], edgecolor="black", linewidth=0.4)
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(3.5, color="black", linewidth=0.6, linestyle=":")
ax.set_xticks(xbase)
ax.set_xticklabels(block_labels)
ax.set_ylabel("In-database cross-validated $R^2$")
ax.set_ylim(-0.5, 0.8)
ax.legend(fontsize=8, ncol=3, loc="lower left", columnspacing=0.8, handlelength=1.5)
ax.set_title("Multi-model in-database benchmark\n(composition-grouped CV; S = Seebeck, PF = power factor)")
check_fonts(fig, "fig4")
plt.savefig(FIG + "fig4_model_comparison.pdf")
plt.savefig(FIG + "fig4_model_comparison.png", dpi=220)
plt.close()

# ---------- Figure 5: classifier comparison (accuracy/F1/AUC) + SHAP ----------
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.8), constrained_layout=True)
ax = axes[0]
clf_names = ["LogisticRegression", "RandomForestClassifier", "XGBoostClassifier"]
clf_labels = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
metrics = ["accuracy_mean", "f1_mean", "auc_mean"]
metric_labels = ["Accuracy", "F1", "AUC"]
metric_colors = ["#999999", "#c77d2e", "#2b5a8c"]
metric_hatch = ["", "//", ".."]
x = np.arange(len(clf_names))
width = 0.25
for i, (met, lab, col) in enumerate(zip(metrics, metric_labels, metric_colors)):
    ht = metric_hatch[i]
    vals = [mech["cv_metrics"][c][met] for c in clf_names]
    errs = [mech["cv_metrics"][c][met.replace("_mean", "_std")] for c in clf_names]
    ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=2, label=lab, color=col, hatch=ht, edgecolor="black", linewidth=0.5)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(clf_labels)
ax.set_ylabel("Grouped 5-fold CV score")
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(fontsize=8, loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.3)
ax.set_title("(a) Mechanism classifier\nbenchmark")

ax = axes[1]
shap_feats = mech["top15_shap_features_full_model"][:12][::-1]
names = [f["feature"].replace("MagpieData ", "") for f in shap_feats]
vals = [f["mean_abs_shap"] for f in shap_feats]
ax.barh(names, vals, color="#2b5a8c", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("(b) Top compositional features\n(XGBoost classifier)")
check_fonts(fig, "fig5")
plt.savefig(FIG + "fig5_classifier_roc_shap.pdf")
plt.savefig(FIG + "fig5_classifier_roc_shap.png", dpi=220)
plt.close()

# ---------- SI Figure: size-matched control ----------
fig, ax = plt.subplots(figsize=(3.9, 3.4), constrained_layout=True)
cats = ["Full\n(imbalanced,\nn=608)", "Size-matched\n(n=300, median\nof 20 draws)"]
auc_vals = [sizectrl["full_imbalanced_result"]["auc"], sizectrl["size_matched_auc_median"]]
auc_lo = [sizectrl["full_imbalanced_result"]["auc"], sizectrl["size_matched_auc_5_95pct"][0]]
auc_hi = [sizectrl["full_imbalanced_result"]["auc"], sizectrl["size_matched_auc_5_95pct"][1]]
err_lo = [0, auc_vals[1] - auc_lo[1]]
err_hi = [0, auc_hi[1] - auc_vals[1]]
ax.bar(cats, auc_vals, color=["#2b5a8c", "#c77d2e"], edgecolor="black", yerr=[err_lo, err_hi], capsize=5)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (AUC=0.5)")
ax.set_ylabel("Classifier AUC")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Size-matched control: imbalance does\nnot explain classifier AUC")
check_fonts(fig, "figs1")
plt.savefig(FIG + "figs1_size_matched_control.pdf")
plt.savefig(FIG + "figs1_size_matched_control.png", dpi=220)
plt.close()

# ---------- SI Figure: composition overlap ----------
fig, ax = plt.subplots(figsize=(3.9, 3.4), constrained_layout=True)
cats = ["ESTM\nformulas\n(n=880)", "Exact\noverlap\n(n=33, 3.75%)", "JARVIS\nunique\n(n=18,220)"]
vals = [880, eda["exact_formula_overlap"], 18220]
ax.bar(cats, vals, color=["#b23b3b", "#999999", "#2b5a8c"], edgecolor="black")
ax.set_yscale("log")
ax.set_ylabel("Unique formulas (log scale)")
for i, v in enumerate(vals):
    ax.text(i, v * 1.15, str(v), ha="center", fontsize=8)
ax.set_ylim(top=1e5)
ax.set_title("Composition overlap between\nJARVIS-DFT and ESTM")
check_fonts(fig, "figs2")
plt.savefig(FIG + "figs2_composition_overlap.pdf")
plt.savefig(FIG + "figs2_composition_overlap.png", dpi=220)
plt.close()

print("All figures saved to", FIG)
