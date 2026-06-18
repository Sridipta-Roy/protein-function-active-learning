"""
streamlit_app.py
================
Protein function classifier demo.

A user pastes an amino acid sequence; the app embeds it with ESM-2 (35M, the deploy-friendly checkpoint), runs the trained classifier, and shows the predicted
broad function class with calibrated confidence, plus a handcrafted-feature readout for interpretation.

Notes for the reader/reviewer:
- The notebooks use ESM-2 650M (macro-F1 ~0.72). This live demo uses ESM-2 35M so it runs within free-tier hosting limits (~1 GB RAM). Same pipeline, smaller model.
- Artifacts (classifier + metadata) are produced by notebook 09 and committed to app/artifacts/.

To Run locally:   streamlit run app/streamlit_app.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import streamlit as st
import joblib

# --------------------------------------------------------------------------- #
# Paths & page config
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).parent
ARTIFACT_DIR = APP_DIR / "artifacts"
# src/ is one level up, for the shared feature + embedding code.
sys.path.insert(0, str(APP_DIR.parent / "src"))

# Importing here after adjusting sys.path, so the shared code is available to the app
from embeddings import ESMEmbedder
import features as F

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
    clf = joblib.load(ARTIFACT_DIR / "classifier_esm35m.joblib")
    meta = json.loads((ARTIFACT_DIR / "model_meta.json").read_text())
    return clf, meta


@st.cache_resource(show_spinner="Loading ESM-2 (35M) — first load takes a moment...")
def load_embedder(model_name):   
    return ESMEmbedder(model_name=model_name, max_length=1024)


def compute_handcrafted(sequence):
    """Return a small dict of interpretable features, or None if unavailable."""
    try:        
        return F.featurize_sequence(sequence)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
MAX_LEN = 1022   # ESM-2's residue limit

def clean_and_validate(raw):
    """Strip FASTA header/whitespace, uppercase, validate.

    Returns (seq, error, notice). On success error is None; notice is a non-fatal
    message (e.g. truncation) or None.
    """
    if not raw or not raw.strip():
        return None, "Paste a sequence to classify.", None
    lines = [ln.strip() for ln in raw.strip().splitlines()]
    lines = [ln for ln in lines if not ln.startswith(">")]   # drop FASTA headers
    seq = "".join(lines).upper().replace(" ", "")
    if len(seq) < 50:
        return None, f"Sequence is {len(seq)} residues; this model expects at least 50.", None
    bad = set(seq) - VALID_AA
    if bad:
        return None, f"Unexpected characters: {', '.join(sorted(bad))}. Use the 20 standard amino acids.", None

    notice = None
    if len(seq) > MAX_LEN:
        original = len(seq)
        seq = seq[:MAX_LEN]
        notice = (
            f"Sequence was {original} residues; using the first {MAX_LEN} "
            "(ESM-2's limit). For long multi-domain proteins the prediction "
            "reflects only the retained N-terminal region."
        )
    return seq, None, notice


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
The model classifies a protein into one of six broad categories (**enzyme, DNA/RNA-binding, receptor, transporter, structural, other**) from
sequence alone.

**Pipeline:** sequence → ESM-2 embedding → logistic-regression classifier.

**Accuracy note (read this):** the research notebooks use ESM-2 650M and reach macro-F1 ≈ 0.72 on a held-out test set. *This live demo* uses ESM-2 35M so it runs
on free hosting, which trades some accuracy for size. Treat predictions as illustrative, not definitive. Model details: `{meta.get('esm_model')}`,
{meta.get('embedding_dim')}-dim embeddings.
        """
    )

EXAMPLE = (
    "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNE"
    "VTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVD"
)

# "Use example" writes into the text area's session_state key BEFORE the widget
# renders, so the box populates. Must run before st.text_area is called.
if st.session_state.get("load_example"):
    st.session_state["seq_input"] = EXAMPLE
    st.session_state["load_example"] = False

raw = st.text_area(
    "Amino acid sequence",
    height=160,
    key="seq_input",
    placeholder="Paste a sequence (FASTA header optional)...",
    help="Standard 20 amino acids. Sequences over 1022 residues are truncated.",
)

col1, col2 = st.columns([1, 1])
with col1:
    go = st.button("Classify", type="primary", use_container_width=True)
with col2:
    if st.button("Use example sequence", use_container_width=True):
        st.session_state["load_example"] = True
        st.rerun()

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
if go:
    seq, err, notice = clean_and_validate(raw)
    if err:
        st.warning(err)
    else:
        if notice:
            st.info(notice)
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
