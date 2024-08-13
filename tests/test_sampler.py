import numpy as np
import pytest
import skabc


@pytest.mark.parametrize(
    "estimator, X, y",
    [
        (
            skabc.NearestNeighborSampler(frac_neighbors=0.1, n_neighbors=3),
            np.random.normal(0, 1, (100, 3)),
            np.random.normal(0, 1, (100, 2)),
        ),
        (
            skabc.NearestNeighborSampler(),
            np.random.normal(0, 1, (100, 3)),
            np.random.normal(0, 1, (100, 2)),
        ),
        (
            skabc.NearestNeighborSampler(n_neighbors=3),
            np.random.normal(0, 1, (100, 3)),
            np.random.normal(0, 1, (100,)),
        ),
        (
            skabc.NearestNeighborSampler(n_neighbors=3),
            np.random.normal(0, 1, (100, 3)),
            np.random.normal(0, 1, (200, 2)),
        ),
    ],
)
def test_estimator_fail(estimator, X, y) -> None:
    with pytest.raises(ValueError):
        estimator.fit(X, y)


def test_vanilla_normal() -> None:
    def _model(rng, n, sigma, size) -> tuple[np.ndarray, np.ndarray]:
        size = size or ()
        mu = rng.normal(0, 1, size + (1,))
        x = mu + sigma * rng.normal(size=size + (n,))
        return x.mean(axis=-1, keepdims=True), mu

    rng = np.random.RandomState(17)

    # We seek to infer the mean of a normal distribution with known variance. First
    # sample the target distribution, then the simulations.
    sigma = 2
    n = 20
    x, _ = _model(rng, n, sigma, (5,))
    y, nu = _model(rng, n, sigma, (1_000_000,))

    sampler = skabc.NearestNeighborSampler(frac_neighbors=1e-3).fit(y, nu)
    samples = sampler.predict(x)
    assert samples.shape == (5, 1000, 1)

    # Posterior mean must be correlated with sample mean for this simple model.
    corrcoef = np.corrcoef(x.mean(axis=-1), samples.mean(axis=1).squeeze())[0, 1]
    assert corrcoef > 0.99

    # Posterior precision is equal to 1 + n / sigma ** 2.
    expected_var = sigma**2 / (sigma**2 + n)
    actual_var = samples.var(axis=1).squeeze()
    np.testing.assert_allclose(actual_var, expected_var, rtol=0.1)
