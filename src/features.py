"""
Handcrafted bioinformatics features for protein function classification.

Each protein sequence -> a fixed-length numeric feature vector built from:
  - length features
  - global physicochemical properties (Biopython ProteinAnalysis)
  - amino acid composition (20 features)
  - grouped residue composition (hydrophobic / charged / etc.)

The main entry point is `build_feature_table(df)`, which takes the labeled
dataset and returns a features DataFrame aligned by `accession`.
"""

from __future__ import annotations

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# The 20 standard amino acids, in fixed order so feature columns are stable.
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Biochemical groupings .
# (Kyte-Doolittle style hydrophobic set; standard charge groups.)
RESIDUE_GROUPS = {
    "hydrophobic": set("AILMFVWC"),
    "polar_uncharged": set("STNQGP"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "aromatic": set("FWY"),
    "tiny": set("AGCS"),
}


def clean_sequence(seq: str) -> str:
    """Uppercase and strip any non-standard residue characters.

    The dataset is already filtered to 20 standard AAs, but this guards against
    stray whitespace / unexpected symbols so ProteinAnalysis never throws.
    """
    seq = str(seq).upper().strip()
    return "".join(c for c in seq if c in AMINO_ACIDS)


def amino_acid_composition(seq: str) -> dict[str, float]:
    """Fraction of each of the 20 amino acids (sums to ~1)."""
    n = len(seq)
    if n == 0:
        return {f"aa_{aa}": 0.0 for aa in AMINO_ACIDS}
    return {f"aa_{aa}": seq.count(aa) / n for aa in AMINO_ACIDS}


def grouped_composition(seq: str) -> dict[str, float]:
    """Fraction of residues falling into each biochemical group."""
    n = len(seq)
    if n == 0:
        return {f"frac_{g}": 0.0 for g in RESIDUE_GROUPS}
    return {
        f"frac_{g}": sum(seq.count(aa) for aa in residues) / n
        for g, residues in RESIDUE_GROUPS.items()
    }


def global_properties(seq: str) -> dict[str, float]:
    """Global physicochemical properties via Biopython ProteinAnalysis."""
    pa = ProteinAnalysis(seq)
    return {
        "molecular_weight": pa.molecular_weight(),
        "aromaticity": pa.aromaticity(),
        "instability_index": pa.instability_index(),
        "isoelectric_point": pa.isoelectric_point(),
        "gravy": pa.gravy(),  # hydrophobicity
    }


def featurize_sequence(seq: str) -> dict[str, float]:
    """Build the full feature dict for a single sequence."""
    seq = clean_sequence(seq)
    feats: dict[str, float] = {"seq_length": len(seq)}
    feats.update(global_properties(seq))
    feats.update(grouped_composition(seq))
    feats.update(amino_acid_composition(seq))
    return feats


def build_feature_table(
    df: pd.DataFrame,
    seq_col: str = "sequence",
    id_col: str = "accession",
    label_col: str = "function_class",
) -> pd.DataFrame:
    """Featurize every row of `df`.

    Returns a DataFrame indexed by `id_col` with all feature columns, plus the
    label column carried through for convenience. Input row order is preserved.
    """
    records = []
    for _, row in df.iterrows():
        feats = featurize_sequence(row[seq_col])
        feats[id_col] = row[id_col]
        if label_col in df.columns:
            feats[label_col] = row[label_col]
        records.append(feats)

    out = pd.DataFrame(records).set_index(id_col)

    # Order columns: label first (if present), then features.
    feature_cols = [c for c in out.columns if c != label_col]
    cols = ([label_col] if label_col in out.columns else []) + feature_cols
    return out[cols]


def feature_columns() -> list[str]:
    """List of feature column names, in the order build_feature_table produces."""
    cols = ["seq_length",
            "molecular_weight", "aromaticity", "instability_index",
            "isoelectric_point", "gravy"]
    cols += [f"frac_{g}" for g in RESIDUE_GROUPS]
    cols += [f"aa_{aa}" for aa in AMINO_ACIDS]
    return cols