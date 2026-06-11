"""
embeddings.py
=============
Frozen ESM-2 mean-pooled embeddings for protein sequences.

- Uses the HuggingFace `transformers` ESM-2 checkpoint 
- Embeddings are the mean over residue token states (excludes special tokens),
  giving one fixed-length vector per protein regardless of length.
- Default model is esm2_t33_650M (1280-dim). 

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Known ESM-2 checkpoints and their embedding dimensions.
# Smaller = faster/less memory, larger = richer representation.
MODEL_DIMS = {
    "facebook/esm2_t33_650M_UR50D": 1280,   # default
    "facebook/esm2_t30_150M_UR50D": 640,    # lighter fallback for limited GPU
    "facebook/esm2_t12_35M_UR50D": 480,     # very light, for quick smoke tests
}

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"


class ESMEmbedder:
    """Loads an ESM-2 model once and produces mean-pooled embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_length: int = 1024,
    ):
        # Imported here so the module imports cleanly even before installing torch.
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()  # frozen: no training, no dropout

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        """Mean-pooled embedding for a list of sequences. Shape: (len(seqs), dim)."""
        torch = self.torch

        enc = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**enc)

        # Last hidden state: (batch, tokens, dim)
        hidden = out.last_hidden_state
        # attention_mask marks real tokens (1) vs padding (0). We also want to
        # exclude the special CLS/EOS tokens from the mean, so we build a mask
        # that drops the first and last real position of each sequence.
        mask = enc["attention_mask"].clone()           # (batch, tokens)
        mask[:, 0] = 0                                  # drop CLS
        lengths = enc["attention_mask"].sum(dim=1)      # real length incl specials
        for i, L in enumerate(lengths):
            mask[i, L - 1] = 0                          # drop EOS (last real token)

        mask = mask.unsqueeze(-1).type_as(hidden)       # (batch, tokens, 1)
        summed = (hidden * mask).sum(dim=1)             # (batch, dim)
        counts = mask.sum(dim=1).clamp(min=1)           # avoid divide-by-zero
        mean_pooled = summed / counts

        return mean_pooled.cpu().numpy().astype(np.float32)

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        id_col: str = "accession",
        seq_col: str = "sequence",
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Embed every row of df. Returns (embeddings, accessions) with rows aligned:
        embeddings[i] corresponds to accessions[i].
        """
        ids = df[id_col].astype(str).tolist()
        seqs = df[seq_col].astype(str).str.upper().tolist()

        iterator = range(0, len(seqs), batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                iterator = tqdm(iterator, desc="Embedding", unit="batch")
            except ImportError:
                pass

        chunks = []
        for start in iterator:
            batch = seqs[start : start + batch_size]
            chunks.append(self.embed_batch(batch))

        embeddings = np.vstack(chunks)
        return embeddings, ids


def save_embeddings(
    embeddings: np.ndarray,
    accessions: Iterable[str],
    out_dir: str | Path,
    emb_name: str = "esm_embeddings.npy",
    meta_name: str = "metadata.csv",
) -> tuple[Path, Path]:
    """Save embeddings (.npy) and an aligned metadata.csv (row order = accession)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / emb_name
    meta_path = out_dir / meta_name

    np.save(emb_path, embeddings)
    pd.DataFrame({"row": range(len(embeddings)), "accession": list(accessions)}).to_csv(
        meta_path, index=False
    )
    return emb_path, meta_path


def load_embeddings(
    out_dir: str | Path,
    emb_name: str = "esm_embeddings.npy",
    meta_name: str = "metadata.csv",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata. metadata['accession'] aligns with embedding rows."""
    out_dir = Path(out_dir)
    embeddings = np.load(out_dir / emb_name)
    meta = pd.read_csv(out_dir / meta_name)
    assert len(embeddings) == len(meta), "Embedding/metadata length mismatch"
    return embeddings, meta
