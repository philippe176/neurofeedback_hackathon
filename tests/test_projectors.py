import numpy as np

from model.projectors import build_projector


def _make_data(n_samples: int = 24, n_features: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = np.repeat(np.arange(4, dtype=np.int64), n_samples // 4)
    return x, y


def test_neural_pca_and_lda_projectors_return_expected_shapes() -> None:
    x, y = _make_data()

    for method in ("neural", "pca", "lda"):
        projector = build_projector(method, projection_dim=2)
        z = projector.fit_transform(x, y=y)
        assert z.shape == (x.shape[0], 2)
        z_next = projector.transform(x[:5])
        assert z_next.shape == (5, 2)


def test_tsne_projector_caches_snapshot_and_supports_transform() -> None:
    x, y = _make_data()
    projector = build_projector("tsne", projection_dim=2, tsne_perplexity=5.0)

    z = projector.fit_transform(x, y=y)
    assert z.shape == (x.shape[0], 2)

    z_next = projector.transform(x[:4])
    assert z_next.shape == (4, 2)
    assert np.all(np.isfinite(z_next))
