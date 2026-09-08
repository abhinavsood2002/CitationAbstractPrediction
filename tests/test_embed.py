import numpy as np

from a2a import embed


def test_embed_cached_reuses_vectors_and_appends(tmp_path, monkeypatch):
    calls = []

    def fake_embed(texts, encoder, batch_size=64, device=None, model=None, show_progress=True):
        calls.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    monkeypatch.setattr(embed, "embed_texts", fake_embed)
    cache = tmp_path / "enc" / "texts.npz"
    v1 = embed.embed_cached(["ab", "abc"], "enc", cache)
    v2 = embed.embed_cached(["abc", "abcd", "ab"], "enc", cache)
    assert cache.exists() and not cache.with_name("texts.npz.tmp").exists()
    assert calls == [["ab", "abc"], ["abcd"]]          # only the new text is embedded
    assert v1[:, 0].tolist() == [2, 3]
    assert v2[:, 0].tolist() == [3, 4, 2]              # input order preserved
