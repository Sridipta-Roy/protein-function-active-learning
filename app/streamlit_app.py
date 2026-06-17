"""
streamlit_app.py
================
Protein function classifier demo.

A user pastes an amino acid sequence; the app embeds it with ESM-2 (35M, the
deploy-friendly checkpoint), runs the trained classifier, and shows the predicted
broad function class with calibrated confidence, plus a handcrafted-feature readout
for interpretation.

Notes for the reader/reviewer:
- The notebooks use ESM-2 650M (macro-F1 ~0.72). This live demo uses ESM-2 35M so
  it runs within free-tier hosting limits (~1 GB RAM). Same pipeline, smaller model.
- Artifacts (classifier + metadata) are produced by notebook 09 and committed to
  app/artifacts/.

Run locally:   streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths & page config
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).parent
ARTIFACT_DIR = APP_DIR / "artifacts"
# src/ is one level up, for the shared feature + embedding code.
import sys
sys.path.insert(0, str(APP_DIR.parent / "src"))

st.set_page_config(page_title="Protein Function Classifier", page_icon="🧬", layout="centered")

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Human-readable notes per class, for the result explanation.
CLASS_NOTES = {
    "enzyme": "catalytic proteins (defined largely by functional annotation rather than bulk composition)",
    "dna_rna_binding": "nucleic-acid binding proteins (often basic / positively charged)",
    "receptor": "signaling receptors (typically membrane-associated)",
    "transporter": "membrane transport proteins (typically hydrophobic)",
    "structural": "structural proteins (e.g. cytoskeletal, extracellular matrix)",
    "other": "proteins not fitting the five categories above",
}


# --------------------------------------------------------------------------- #
# Cached loaders (run once, reused across interactions)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    import joblib
    clf = joblib.load(ARTIFACT_DIR / "classifier_esm35m.joblib")
    meta = json.loads((ARTIFACT_DIR / "model_meta.json").read_text())
    return clf, meta


@st.cache_resource(show_spinner="Loading ESM-2 (35M) — first load takes a moment...")
def load_embedder(model_name):
    from embeddings import ESMEmbedder
    return ESMEmbedder(model_name=model_name, max_length=1024)


def compute_handcrafted(sequence):
    """Return a small dict of interpretable features, or None if unavailable."""
    try:
        import features as F
        return F.featurize_sequence(sequence)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def clean_and_validate(raw):
    """Strip FASTA header/whitespace, uppercase, validate. Returns (seq, error)."""
    if not raw or not raw.strip():
        return None, "Paste a sequence to classify."
    lines = [ln.strip() for ln in raw.strip().splitlines()]
    lines = [ln for ln in lines if not ln.startswith(">")]   # drop FASTA headers
    seq = "".join(lines).upper().replace(" ", "")
    if len(seq) < 50:
        return None, f"Sequence is {len(seq)} residues; this model expects at least 50."
    if len(seq) > 1000:
        return None, f"Sequence is {len(seq)} residues; trimmed models expect up to 1000. Truncate and retry."
    bad = set(seq) - VALID_AA
    if bad:
        return None, f"Unexpected characters: {', '.join(sorted(bad))}. Use the 20 standard amino acids."
    return seq, None


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("Protein function classifier")
st.caption(
    "Paste an amino acid sequence to predict its broad function class. "
    "Built on ESM-2 protein language model embeddings."
)

clf, meta = load_classifier()

with st.expander("How this works & accuracy note"):
    st.markdown(
        f"""
The model classifies a protein into one of six broad categories
(**enzyme, DNA/RNA-binding, receptor, transporter, structural, other**) from
sequence alone.

**Pipeline:** sequence → ESM-2 embedding → logistic-regression classifier.

**Accuracy note (read this):** the research notebooks use ESM-2 650M and reach
macro-F1 ≈ 0.72 on a held-out test set. *This live demo* uses ESM-2 35M so it runs
on free hosting, which trades some accuracy for size. Treat predictions as
illustrative, not definitive. Model details: `{meta.get('esm_model')}`,
{meta.get('embedding_dim')}-dim embeddings.
        """
    )

EXAMPLE = (
    "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNE"
    "VTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVD"
)

raw = st.text_area(
    "Amino acid sequence",
    height=160,
    placeholder="Paste a sequence (FASTA header optional)...",
    help="Standard 20 amino acids, 50–1000 residues.",
)

col1, col2 = st.columns([1, 1])
with col1:
    go = st.button("Classify", type="primary", use_container_width=True)
with col2:
    if st.button("Use example sequence", use_container_width=True):
        st.session_state["example"] = EXAMPLE
        st.rerun()

if "example" in st.session_state and not raw:
    raw = st.session_state["example"]

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
if go or ("example" in st.session_state and raw):
    seq, err = clean_and_validate(raw)
    if err:
        st.warning(err)
    else:
        embedder = load_embedder(meta["esm_model"])
        with st.spinner("Embedding sequence and predicting..."):
            vec = embedder.embed_batch([seq])          # (1, dim)
            proba = clf.predict_proba(vec)[0]
            classes = list(clf.classes_)
            top = int(np.argmax(proba))
            pred = classes[top]
            confidence = float(proba[top])

        st.divider()
        st.subheader(f"Predicted class: {pred.replace('_', '/')}")
        st.caption(CLASS_NOTES.get(pred, ""))

        # Confidence, shown honestly (a bar + the number).
        st.metric("Confidence", f"{confidence:.0%}")
        st.progress(confidence)
        if confidence < 0.5:
            st.info("Low confidence — the sequence sits between classes. See the full distribution below.")

        # Full probability distribution (more honest than top-1 alone).
        st.markdown("**Probability across all classes**")
        order = np.argsort(proba)[::-1]
        for i in order:
            label = classes[i].replace("_", "/")
            st.write(f"{label}: {proba[i]:.0%}")
            st.progress(float(proba[i]))

        # Optional interpretability readout.
        feats = compute_handcrafted(seq)
        if feats:
            with st.expander("Sequence features (interpretation)"):
                show = {k: feats[k] for k in
                        ("seq_length", "gravy", "aromaticity", "isoelectric_point",
                         "molecular_weight", "frac_hydrophobic", "frac_positive")
                        if k in feats}
                for k, v in show.items():
                    st.write(f"{k}: {v:.3g}" if isinstance(v, float) else f"{k}: {v}")
                st.caption(
                    "These handcrafted features drive the interpretable baseline model "
                    "(see notebooks). High hydrophobicity is consistent with membrane "
                    "proteins; high positive charge with nucleic-acid binders."
                )

st.divider()
st.caption(
    "Research project — supervised classification, interpretability, and active "
    "learning for protein function prediction. Predictions are illustrative."
)
