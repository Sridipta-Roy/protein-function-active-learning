# EDA, Feature Engineering & Baseline Modeling

This stage takes the labeled UniProt dataset and answers one question before any
protein language model is involved: **how much protein-function signal is recoverable
from simple sequence biochemistry?** It covers sequence-level EDA, handcrafted
feature engineering, and baseline supervised models.

## Dataset

- UniProt Swiss-Prot reviewed human proteins, length 50–1000 aa.
- 6 broad function classes: `enzyme`, `dna_rna_binding`, `receptor`, `transporter`,
  `structural`, `other`.
- Mild class imbalance (~3.6:1), so **macro-F1** is the headline metric throughout and
  splits are stratified.

## Sequence-level EDA

Notebook: `notebooks/02_sequence_eda.ipynb`

Central question: *do protein classes differ in sequence length or amino acid
composition?* If they do, handcrafted features have signal to work with.

Checks performed: class balance, duplicate sequences (leakage risk), missing GO/EC
annotation per class, length distribution overall and by class, and per-class amino
acid composition.

Key findings:

- **Length carries weak class signal.** Class medians cluster tightly and all boxes
  overlap heavily; `other` skews shortest and `enzyme`/`structural` slightly longer.
  Length alone cannot separate classes — an expected, useful result.
- **Composition carries more.** Per-class amino acid composition shows biochemically
  sensible deviations (e.g. hydrophobic-residue patterns), which is what the Day 4
  features are built to capture.
- **Duplicates exist** and are a leakage risk — flagged here, removed in Day 5.

Figures: `class_distribution.png`, `length_histogram.png`, `length_by_class.png`,
`aa_composition_heatmap.png`, `aa_composition_deviation.png`.

## Handcrafted feature engineering

Module: `src/features.py` · Notebook: `notebooks/03_feature_engineering.ipynb` ·
Output: `data/processed/sequence_features.csv`

Each sequence is turned into a fixed **32-feature** numeric vector using Biopython's
`ProteinAnalysis`:

| Feature group | Features | Meaning |
|---|---|---|
| Length | `seq_length` | protein size |
| Global properties | `molecular_weight`, `aromaticity`, `instability_index`, `isoelectric_point`, `gravy` | mass, stability, charge, hydrophobicity |
| Grouped composition | `frac_hydrophobic`, `frac_polar_uncharged`, `frac_positive`, `frac_negative`, `frac_aromatic`, `frac_tiny` | interpretable biochemical fractions |
| AA composition | `aa_A` … `aa_Y` (20) | residue-level pattern |

Notes:

- The full 400-feature dipeptide composition was deliberately skipped — the EDA showed
  mono-residue composition already carries the class signal, and 400 sparse columns
  would add noise and hurt interpretability. Six grouped biochemical fractions were
  added instead.
- All feature logic lives in `src/features.py` (entry point `build_feature_table`) so it
  is reused by the model and, later, the Streamlit app.

## Baseline supervised models

Module: `src/train.py` · Notebook: `notebooks/04_supervised_baselines.ipynb` ·
Output: `results/baseline_metrics.csv`, `results/confusion_matrix_baseline.png`

Setup:

- **Deduplication first.** Identical sequences are dropped (keep first) before the split
  to prevent train/test leakage. Cross-class duplicates are reported, since keep-first
  assigns them an arbitrary label.
- Stratified 80/20 train/test split; held-out test never used for tuning.
- 5-fold stratified CV on the training set for a stable model-selection signal.
- Each model is an sklearn `Pipeline`. Logistic Regression is `StandardScaler`-scaled
  (features span very different ranges); tree models pass features through unscaled.
  Labels are integer-encoded inside `run_baselines` (XGBoost requires it) and decoded
  back to class names for reporting.
- Models: Logistic Regression, Random Forest, XGBoost, HistGradientBoosting.

### Results

| Model | CV macro-F1 | Test macro-F1 | Test accuracy | Test weighted-F1 |
|---|---|---|---|---|
| XGBoost | 0.480 ± 0.015 | **0.527** | 0.565 | 0.554 |
| HistGradientBoosting | 0.476 ± 0.012 | 0.525 | 0.557 | 0.549 |
| Random Forest | 0.460 ± 0.006 | 0.509 | 0.556 | 0.535 |
| Logistic Regression | 0.399 ± 0.016 | 0.420 | 0.443 | 0.449 |

### Interpretation

- **XGBoost reaches macro-F1 0.527** on a 6-class problem (chance ≈ 0.17).
  Simple sequence biochemistry carries real but partial signal — enough to motivate ESM
  embeddings next.
- **Linear vs non-linear.** A ~0.10 macro-F1 gap separates Logistic Regression (0.42)
  from the three tree models (0.51–0.53): the class structure is largely non-linear, so
  models that capture feature interactions win clearly.
- **Feature ceiling.** XGBoost and HistGradientBoosting agree to within 0.002, and
  HistGB's balanced class weights did not lift the minority classes. Two boosting models
  landing in the same place indicates the ceiling is in the feature representation, not
  the classifier — the central argument for moving to protein language model embeddings.
- **Overfitting.** CV→test gaps are small and stable (e.g. XGBoost 0.480→0.527). No
  concern on 32 features.

Per-class behavior (XGBoost): `dna_rna_binding` is easiest (recall 0.78) thanks to a
distinctive charged-residue signature; `enzyme` is solid; `transporter` shows good
precision but lower recall (hydrophobicity helps but isn't enough); `structural` is
confident but low-recall; `receptor` is weakest (F1 0.38), bleeding into `enzyme` and
`transporter` due to shared membrane-association; `other` is diffuse, as expected for a
catch-all. The dominant confusion is everything leaking into `enzyme`, partly the
largest-class pull and partly the broad sequence space of catalytic proteins.

## How to run

```bash
# Day 3
jupyter notebook notebooks/02_sequence_eda.ipynb
# Day 4 — writes data/processed/sequence_features.csv
jupyter notebook notebooks/03_feature_engineering.ipynb
# Day 5 — writes results/baseline_metrics.csv + confusion matrix
jupyter notebook notebooks/04_supervised_baselines.ipynb
```

Dependencies: `biopython`, `scikit-learn` (≥1.4 for HistGB class weighting), `xgboost`,
`pandas`, `numpy`, `matplotlib`, `seaborn`.

## Takeaway

Handcrafted bioinformatics features reach macro-F1 ~0.53, well above random, and tree
models confirm the signal is non-linear. The agreement between two boosting models shows
the performance ceiling lies in the feature representation rather than the classifier —
which is exactly the motivation for introducing ESM-2 embeddings and a hybrid model in
the next stage.
