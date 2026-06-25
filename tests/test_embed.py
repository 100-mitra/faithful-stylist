"""P1.4: the default hashing embedder is deterministic, normalized, and discriminative."""

import numpy as np

from core.embed import HashingTfidfEmbedder, get_embedder


def test_shape_and_normalization():
    emb = HashingTfidfEmbedder(dim=256)
    vecs = emb.embed(["white gold diamond ring", "silver bangle"])
    assert vecs.shape == (2, 256)
    assert vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_deterministic_across_instances():
    a = HashingTfidfEmbedder().embed(["vintage rose gold pendant"])
    b = HashingTfidfEmbedder().embed(["vintage rose gold pendant"])
    assert np.array_equal(a, b)


def test_similar_texts_are_closer_than_dissimilar():
    emb = HashingTfidfEmbedder()
    v = emb.embed(
        [
            "white gold diamond engagement ring vintage",  # query
            "white gold diamond ring vintage romantic",  # similar
            "silver minimalist bangle daily wear",  # dissimilar
        ]
    )
    sim_close = float(v[0] @ v[1])
    sim_far = float(v[0] @ v[2])
    assert sim_close > sim_far


def test_factory_default_is_hashing():
    emb = get_embedder()
    assert emb.name.startswith("hashing-tf")
