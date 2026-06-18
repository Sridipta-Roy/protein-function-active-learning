# Streamlit demo

Interactive demo: paste a protein sequence, get its predicted broad function class with confidence and an interpretability readout.

## What it does

Sequence → **ESM-2 (35M)** embedding → logistic-regression classifier → predicted class (enzyme / DNA-RNA-binding / receptor / transporter / structural / other).

## Accuracy note

The research notebooks use **ESM-2 650M** (test macro-F1 ≈ 0.72). This live demo uses **ESM-2 35M** so it fits free-tier hosting (~1 GB RAM) — same pipeline, smaller model, some accuracy traded for deployability. Predictions are illustrative.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

First run downloads the ESM-2 35M weights (~150 MB) and caches them.

## Required artifacts

The app loads two files produced by `notebooks/09_prepare_demo_artifacts.ipynb`:

```
app/artifacts/classifier_esm35m.joblib   # trained LogReg (small)
app/artifacts/model_meta.json            # model name, dim, class list
```

Generate them once (Colab GPU recommended for the embedding step), then commit
`app/artifacts/` to the repo so the deployed app can load them.

## Deploy to Streamlit Community Cloud (free)

1. Push the repo to GitHub (including `app/artifacts/`).
2. At share.streamlit.io, create an app pointing at `app/streamlit_app.py`.
3. Set the requirements path to this `requirements.txt`.

### Caveats

- **Memory:** ESM-2 35M (~150 MB) fits the ~1 GB free tier; do **not** swap in the
  650M model on free hosting — it will exceed RAM and crash.
- **Cold start:** free apps sleep after inactivity and reload the model on wake;
  the first request after sleeping is slow (model re-download + load).
- **CPU inference:** embedding one sequence on CPU takes a few seconds — fine for a
  demo, not for batch use.
