"""Abstract embeddings for similarity-based scoring.

Two encoders are supported, both used with unit-normalised outputs so that a
dot product is a cosine similarity:

* ``malteos/scincl`` (default): SciBERT further trained on citation-graph
  neighbourhoods (Ostendorff et al. 2022); CLS pooling, 512 tokens.
* ``sentence-transformers/all-MiniLM-L6-v2``: general-purpose sentence
  encoder, mean pooling, 256 tokens. Used as the robustness check because
  SciNCL's training objective (citation neighbours are close) is related to
  the task being measured.

``embed_cached`` keeps one ``.npz`` per (encoder, cache name) keyed by text
hash so repeated scoring runs embed only what is new.
"""

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_ENCODER = "malteos/scincl"
CLS_ENCODERS = ("scincl", "specter")


def short_name(encoder: str) -> str:
    return encoder.split("/")[-1].lower()


def load_encoder(encoder: str = DEFAULT_ENCODER, device: str | None = None):
    from sentence_transformers import SentenceTransformer, models

    if any(k in encoder.lower() for k in CLS_ENCODERS):
        word = models.Transformer(encoder, max_seq_length=512)
        pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="cls")
        return SentenceTransformer(modules=[word, pool], device=device)
    return SentenceTransformer(encoder, device=device)


def embed_texts(texts: list[str], encoder: str = DEFAULT_ENCODER, batch_size: int = 64,
                device: str | None = None, model=None, show_progress: bool = True) -> np.ndarray:
    """(n, d) float32 unit-norm embeddings in input order."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    model = model or load_encoder(encoder, device)
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=show_progress).astype(np.float32)


def text_key(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()


def embed_cached(texts: list[str], encoder: str, cache_path: Path,
                 batch_size: int = 64, device: str | None = None) -> np.ndarray:
    """Embed ``texts`` reusing vectors stored in ``cache_path`` (an .npz).

    The cache stores ``keys`` (blake2b of the text) and ``vecs``; texts not in
    the cache are embedded and appended. Returns vectors in input order.
    """
    cache_path = Path(cache_path)
    keys, vecs = np.array([], dtype=object), None
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        keys, vecs = z["keys"], z["vecs"]
    have = {k: i for i, k in enumerate(keys)}
    want = [text_key(t) for t in texts]
    missing_idx = sorted({k: i for i, k in enumerate(want) if k not in have}.values())
    if missing_idx:
        new = embed_texts([texts[i] for i in missing_idx], encoder, batch_size, device)
        new_keys = np.array([want[i] for i in missing_idx], dtype=object)
        vecs = new if vecs is None else np.vstack([vecs, new])
        keys = np.concatenate([keys, new_keys])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        with open(tmp, "wb") as f:  # file handle: np.savez must not append ".npz"
            np.savez(f, keys=keys, vecs=vecs)
        tmp.replace(cache_path)
        have = {k: i for i, k in enumerate(keys)}
    return vecs[[have[k] for k in want]]
