# Protein Function Classification with Active Learning [![Streamlit](https://img.shields.io/badge/demo-live-brightgreen)](https://protein-function-classifier-ph01.streamlit.app)

Classify human proteins into six broad function classes from sequence alone, and test whether model-guided label selection (active learning) can reach good accuracy with fewer labels than random selection.

The project runs through a deliberate progression: handcrafted biochemical features → protein language model (ESM-2) embeddings → active learning simulation. The point of that ordering is to measure how much signal each stage actually adds, rather than jumping straight to a transformer.

## Question

Can sequence-derived features and ESM-2 embeddings predict broad protein function categories, and does uncertainty-based active learning reduce the number of labels needed to train such a classifier?

## Data

- UniProt Swiss-Prot reviewed human proteins, length 50–1000 aa.
- 8,520 proteins after filtering; ~7,600 after dropping duplicate sequences.
- Six classes: `enzyme`, `dna_rna_binding`, `receptor`, `transporter`, `structural`, `other`. Mild imbalance (~3.6:1).
- Labels are assigned by a priority cascade over UniProt keywords, GO terms, and EC numbers. This means the classifier partly learns to reproduce UniProt's annotation rules from sequence rather than discovering function from scratch —  stated explicitly because it bounds what the results mean.

Full data and labeling details: [`README_data_fetching_processing.md`](README_data_fetching_processing.md).

## Results summary

Headline metric is macro-F1 (mild class imbalance; stratified splits throughout).

| Feature set | Model | Test macro-F1 |
|---|---|---|
| Handcrafted (32) | Logistic Regression | 0.420 |
| Handcrafted (32) | XGBoost | 0.527 |
| ESM-2 (1280) | Logistic Regression | 0.675 |
| ESM-2 (1280) | XGBoost | 0.717 |
| Handcrafted + ESM-2 (1312) | XGBoost | 0.722 |

Main findings:

- **ESM-2 embeddings break the handcrafted ceiling.** XGBoost goes from 0.527 (handcrafted) to 0.717 (ESM). McNemar's test on the shared test set confirms the gap is real (p ≈ 4e-45): ESM corrects 359 proteins the handcrafted model got wrong, against 67 the other way.
- **The bottleneck was the representation, not the classifier.** Two boosting models on handcrafted features agree to within 0.002, and even logistic regression on ESM (0.675) beats boosted trees on handcrafted features (0.527).
- **The hybrid model adds nothing significant.** Hybrid beats ESM-only by 0.005, but McNemar's test says that gap is not significant (p = 0.51). The handcrafted features carry no independent signal once ESM is present, so ESM-only is reported as the effective best model.

Modeling details and the earlier baseline stage:
[`README_eda_features_modeling.md`](README_eda_features_modeling.md).

## Interpretability

Two views, because the best model and the interpretable model are not the same one.

Handcrafted features (interpretable, weaker model): permutation importance is flat — no single feature dominates, consistent with the low ceiling. The per-class signatures match biochemistry: transporters and receptors are hydrophobic (high GRAVY), DNA/RNA-binding proteins are positively charged, structural proteins are cysteine-rich. SHAP adds direction and shows a subtler point — enzymes have no positive compositional signature and are instead identified largely by the *absence* of features that mark other classes (e.g. low serine pushes toward enzyme). That exclusionary signal is why the heatmap shows a flat enzyme row while SHAP still finds usable signal: the heatmap measures average position, SHAP measures per-protein model behavior, and the two diverge most for acompositionally average but heterogeneous class like enzyme.

ESM embeddings (best model, abstract): individual embedding dimensions carry no biochemical meaning, so feature-level interpretation does not apply. A 2D PCA of the embeddings is an undifferentiated blob (the top two components explain only
~53% of variance) — the class signal is distributed across many dimensions, which is the defining property of a learned representation and the reason ESM wins. A non-linear t-SNE projection recovers visible class structure that PCA cannot show.

## Active learning

Simulates a low-label discovery setting: most labels are hidden, and each round a strategy picks 100 proteins to "label". Three strategies compared on a fixed test set, using logistic regression on ESM embeddings (fast to retrain):

- **random** (averaged over 5 seeds) — the baseline.
- **uncertainty** — pick the least-confident proteins (1 − max class probability).
- **diversity** — uncertain and spread out (cluster candidates, pick representatives).

Result: uncertainty sampling reached macro-F1 ≥ 0.625 with 600 labels versus 1100 for random — about 45% fewer labels for the same accuracy. Diversity sampling did not improve on plain uncertainty (700 labels to the same target), so the simpler
method is preferred here; the deduplicated pool was apparently not redundant enough for diversity-aware selection to help. Below ~300 labels all strategies tie, since the model's uncertainty estimates are unreliable until it has enough data.

Caveat: uncertainty came from logistic-regression probabilities, which are not perfectly calibrated; better calibration could sharpen the effect.

## Demo

A Streamlit app classifies a pasted sequence: ESM-2 embedding → logistic-regression classifier → predicted class with the full probability distribution and a handcrafted-feature readout.

The notebooks use ESM-2 650M (macro-F1 ≈ 0.72). The deployed demo uses ESM-2 35M so it fits free-tier hosting (~1 GB RAM) — same pipeline, smaller model, some accuracy traded for deployability. Setup and deployment notes:
[`app/README_streamlit_demo.md`](app/README_streamlit_demo.md).

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Repository layout

```text
.
├── data/
│   ├── raw/                 UniProt TSV/CSV
│   ├── processed/           cleaned, filtered, labeled, features
│   └── embeddings/          ESM-2 embeddings (.npy + metadata)
├── notebooks/
│   ├── 001_data_collection.ipynb
│   ├── 02_sequence_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_supervised_baselines.ipynb
│   ├── 05_esm_embeddings.ipynb            (Colab GPU)
│   ├── 06_embedding_models.ipynb
│   ├── 07_interpretability.ipynb
│   ├── 08_active_learning.ipynb
│   └── 09_prepare_demo_artifacts.ipynb    (Colab GPU)
├── src/
│   ├── features.py          handcrafted features (Biopython)
│   ├── train.py             baseline pipelines
│   ├── embeddings.py        ESM-2 embedding generation
│   └── active_learning.py   sampling strategies + experiment loop
├── galaxy_inputs/           protein IDs + FASTA for Galaxy
├── galaxy_outputs/          GO / Reactome enrichment results
├── results/                 metrics + figures
└── app/
    ├── streamlit_app.py
    └── artifacts/           deployable 35M classifier
```
## Tech stack

- **Language:** Python 3.11
- **Data:** UniProt REST API, pandas, NumPy
- **Features:** Biopython (ProteinAnalysis)
- **Protein language model:** ESM-2 (650M for research, 35M for the demo) via HuggingFace Transformers, PyTorch
- **Modeling:** scikit-learn, XGBoost
- **Interpretability:** SHAP, permutation importance, t-SNE / PCA
- **Statistics:** statsmodels (McNemar's test)
- **Active learning:** custom (scikit-learn KMeans for diversity sampling)
- **Demo:** Streamlit, deployed on Streamlit Community Cloud
- **Environment:** VS Code (local), Google Colab (GPU embedding steps)
  
## Reproducing

GPU (Colab) is needed only for the two embedding steps (notebooks 05 and 09); everything else runs on CPU.

```bash
pip install -r requirements.txt
# notebooks 001 → 04 locally
# notebook 05 on Colab GPU (generates ESM-2 embeddings)
# notebooks 06 → 08 locally
# notebook 09 on Colab GPU (generates the demo's 35M classifier)
```

## Limitations

- Labels are derived from existing annotations, so the task is closer to "predict UniProt's annotation from sequence" than to de novo function discovery.
- Sequences are capped at 1000 aa; long multi-domain proteins are truncated, so their predictions reflect only the retained region (e.g. EGFR, both enzyme and receptor, is predicted receptor from its N-terminal region).
- `other` is a heterogeneous catch-all and is the hardest class throughout.
- The active-learning result is one dataset with one model; the label-efficiency gain should not be assumed to transfer to other settings without rechecking.
