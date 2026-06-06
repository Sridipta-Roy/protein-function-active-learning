# Active Learning for Protein Function Classification

Classifying broad protein function categories from sequence-derived features, Galaxy-based
functional annotations, and protein language model embeddings — and using active learning
to study how few labels are needed to reach good performance.

## Purpose

Protein function annotation is expensive and incomplete: most known protein sequences lack
experimentally verified functional labels. This project builds an interpretable supervised
classifier for broad protein function categories, then simulates a low-label discovery
setting in which the model chooses which proteins to label next.

The guiding question:

> Can protein sequence-derived features and protein language model embeddings predict broad
> functional categories, and can Galaxy-generated annotations provide biological context for
> model interpretation?

The classification target is six broad classes — `enzyme`, `transporter`, `receptor`,
`dna_rna_binding`, `structural`, and `other` — rather than thousands of GO terms, keeping
the problem interpretable and meaningful.

## Approach

The project is built in two phases.

**Supervised pipeline.** Starting from a clean UniProt Swiss-Prot dataset, it progresses
through sequence-level exploratory analysis, handcrafted biochemical feature engineering,
baseline classifiers, ESM-2 protein language model embeddings, a hybrid feature model, and
model interpretability. Galaxy workflows (GO enrichment, Reactome pathway mapping, and
optional EggNOG/InterProScan annotation) provide an external biological validation layer
for the model's predictions.

**Active learning simulation.** Treating most labels as hidden, the model starts from a
small labeled set and iteratively selects which proteins to label next. Random sampling,
uncertainty sampling, and uncertainty + diversity sampling are compared on a fixed test set
to study whether model-guided selection reaches strong performance with fewer labels.

## Dataset

- UniProt Swiss-Prot reviewed human proteins (sequence length 50–1000 aa).
- Fetched directly from the UniProt REST API with pagination and quality filtering.
- Labels: six broad functional categories derived from EC numbers, GO terms, keywords, and
  protein names.
- Mild class imbalance, so evaluation uses stratified splits and macro-F1.

## Current status

The data collection, exploratory analysis, handcrafted feature engineering, and baseline
modeling stages are complete. Handcrafted biochemical features reach a macro-F1 of ~0.53 on
the six-class problem (well above the ~0.17 random baseline), with tree-based models
clearly outperforming logistic regression — establishing the signal recoverable from simple
sequence biochemistry before introducing protein language model embeddings. ESM embeddings,
the hybrid model, interpretability, and the active learning experiments follow.

See `README_eda_features_modeling.md` for detailed results from the EDA, feature
engineering, and baseline modeling stages.

## Tech stack

**Language:** Python 3.12

**Data & numerics:** pandas, NumPy

**Machine learning:** scikit-learn (pipelines, `ColumnTransformer`, logistic regression,
random forest, histogram gradient boosting), XGBoost

**Bioinformatics:** Biopython (`ProteinAnalysis` for physicochemical features), ESM-2
(protein language model embeddings), Galaxy (GO enrichment, Reactome, EggNOG, InterProScan)

**Data source:** UniProt REST API

**Visualization:** matplotlib, seaborn

**App:** Streamlit (interactive prediction demo)

**Environment:** Jupyter notebooks for analysis, reusable `src/` modules for shared logic

## Repository layout

```text
protein-function-active-learning/
├── data/
│   ├── raw/              # raw UniProt download
│   ├── processed/        # cleaned dataset, labels, features
│   └── embeddings/        # ESM-2 embeddings
├── galaxy_inputs/        # protein IDs and FASTA for Galaxy
├── galaxy_outputs/       # GO enrichment, Reactome, annotation outputs
├── notebooks/            # data collection, EDA, features, modeling, active learning
├── src/                  # features.py, train.py, embeddings.py, active_learning.py, ...
├── results/              # metrics and figures
└── app/                  # Streamlit demo
```

## How to run

Run the notebooks in order from data collection through modeling. Each writes its outputs
into `data/processed/` or `results/` for the next stage to consume. Shared logic
(featurization, training pipelines) lives in `src/` and is imported by the notebooks.

## Relationship to previous work

This project extends an earlier protein function prediction effort that generated
natural-language function descriptions using ESM-2 and a small language model. Here the
focus shifts to interpretable classification, Galaxy-based biological annotation, and active
learning for low-label biological discovery.
