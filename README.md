# Cross-database ML for thermoelectric transport properties and transport-mechanism classification (JARVIS-DFT vs. ESTM)

This repository contains the code and derived results for a study of machine-learning prediction of thermoelectric properties (Seebeck coefficient, electrical conductivity, power factor, thermal conductivity, ZT) from composition, using the computational JARVIS-DFT database and the experimental ESTM database. The study benchmarks in-database and cross-database transfer, diagnoses distribution shift and formula leakage between the databases, assigns a band-like or small-polaron-hopping label to ESTM compounds from the temperature dependence of their conductivity, and trains composition-only classifiers for that label. A screening of JARVIS-DFT compounds without ESTM entries is included.

## Run order

Run all scripts from the repository root (they use paths relative to it, e.g. `data/` and `figures/`; create `figures/` first). Before step 1, clone the ESTM data into `data/SIMD`:

```
git clone https://github.com/KRICT-DATA/SIMD data/SIMD
```

Install the dependencies with `pip install -r requirements.txt` (Python 3.12).

1. `01_pull_jarvis.py`: downloads JARVIS-DFT `dft_3d` data
2. `02_filter_jarvis_te.py`: keeps records with thermoelectric fields
3. `03_eda.py`: summary statistics and composition overlap of both databases
4. `04_lit_sanity_check.py`: counts compounds from known hopping and band-transport families in ESTM
5. `05_mechanism_labeling.py`: derives band-like / hopping-like labels from each ESTM compound's conductivity and Seebeck temperature dependence
6. `06_mechanism_validation.py`: compares derived labels with known literature compound families
7. `07_duplicate_and_arithmetic_audit.py`: duplicate-record and power-factor consistency checks
8. `08_featurize.py`: Magpie composition features (plus structure descriptors for JARVIS)
9. `09_build_target_tables.py`: per-compound target tables
10. `10_property_prediction.py`: in-database and cross-database benchmark for Seebeck and conductivity
11. `11_shift_diagnosis.py`: target and feature shift between the databases
12. `12_mechanism_classifier.py`: composition-only mechanism classifiers with SHAP
13. `13_threshold_sweep.py`: sensitivity of the labels to the labeling thresholds
14. `14_size_matched_control.py`: class-size-matched control for the classifier
15. `15_bootstrap_ci.py`: grouped-bootstrap confidence intervals for transfer R2 and classifier AUC
16. `16_reproducibility.py`: records software versions
17. `17_leakage_robustness.py`: cross-database transfer excluding exact-formula overlaps
18. `18_figure_predictions.py`: single-fit predictions used for parity and ROC plots
19. `19_make_figures.py`: main figures
20. `20_family_holdout_check.py`: leave-one-element-family-out classifier check
21. `21_extend_target_tables.py`: adds power factor, thermal conductivity and ZT to the target tables (creates the `*_ext` files read by later steps)
22. `20_raw_distributions.py`: raw-data distribution figure (needs the files from step 21, so it is run after it despite its file number)
23. `22_extended_property_prediction.py`: benchmark for the added properties
24. `23_extended_ci_leakage_shift.py`: confidence intervals, leakage and shift checks for the added properties
25. `24_regression_shap.py`: SHAP attribution for the Seebeck, log10 conductivity and power factor models
26. `25_screening_demo.py`: applies the trained classifier to JARVIS compounds with no ESTM entry
27. `26_make_graphical_abstract.py`: graphical abstract (output directory set with the `FIG_DIR` environment variable, default `figures`)

## Data files (`data/`)

- `estm_mechanism_labels.json`: derived label (band-like, hopping-like, ambiguous, excluded) for each ESTM formula, with the Spearman and Arrhenius diagnostics used
- `estm_target_table.json`, `estm_target_table_ext.json`: per-compound ESTM target tables (Seebeck, conductivity, power factor, thermal conductivity, ZT)
- `phase1_*.json`: exploratory summary and literature-family counts
- `phase2_*.json`: labeling, validation, audit, featurization, benchmark, shift, bootstrap, leakage, threshold-sweep, size-matched-control and software-version results
- `phase3_*.json`: family-holdout check and prediction arrays used for the figures
- `phase4_*.json`: results for the added properties (power factor, thermal conductivity, ZT)
- `phase5_*.json`: raw-distribution summary, regression SHAP and screening results

Raw data are not included. JARVIS-DFT data are downloaded with `jarvis-tools` (`01_pull_jarvis.py`). ESTM data are cloned from `github.com/KRICT-DATA/SIMD` into `data/SIMD`. Running `01_pull_jarvis.py` through `09_build_target_tables.py` regenerates the intermediate files.

## Requirements

Python 3 with pandas, numpy, scikit-learn, xgboost, lightgbm, shap, matminer, pymatgen, jarvis-tools, matplotlib, openpyxl and scipy. Pinned versions are in `requirements.txt` and `data/phase2_software_versions.json`.

## License

MIT (see `LICENSE`).
