"""Abstract embeddings for similarity-based scoring.

The encoder is ``Qwen/Qwen3-Embedding-0.6B``: a general-purpose text encoder,
last-token pooling, run in bfloat16 with 1,024 tokens. Both sides are encoded
as documents (no instruction prompt), so the similarity is symmetric, and
outputs are unit-normalised so that a dot product is a cosine similarity. Any
other sentence-transformers model name is loaded with its own defaults.

``embed_cached`` keeps one ``.npz`` per (encoder, cache name) keyed by text
hash so repeated scoring runs embed only what is new. Large batches are spread
over every visible GPU.
"""

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_ENCODER = "Qwen/Qwen3-Embedding-0.6B"
MULTI_GPU_MIN_TEXTS = 5000


def short_name(encoder: str) -> str:
    return encoder.split("/")[-1].lower()


def load_encoder(encoder: str = DEFAULT_ENCODER, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    if "qwen3-embedding" in encoder.lower():
        import torch
        model = SentenceTransformer(encoder, device=device, model_kwargs={"dtype": torch.bfloat16})
        model.max_seq_length = 1024
        return model
    return SentenceTransformer(encoder, device=device)


def embed_texts(texts: list[str], encoder: str = DEFAULT_ENCODER, batch_size: int = 64,
                device: str | None = None, model=None, show_progress: bool = True,
                multi_gpu_min: int = MULTI_GPU_MIN_TEXTS) -> np.ndarray:
    """(n, d) float32 unit-norm embeddings in input order. With no ``device`` given,
    ``multi_gpu_min`` or more texts are encoded on every visible GPU (one process each)."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    model = model or load_encoder(encoder, device)
    target = device
    if device is None and len(texts) >= multi_gpu_min:
        import torch
        if torch.cuda.device_count() > 1:
            target = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return np.asarray(model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                   normalize_embeddings=True, device=target,
                                   show_progress_bar=show_progress), dtype=np.float32)


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
