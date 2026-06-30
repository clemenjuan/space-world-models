from __future__ import annotations

import math

import numpy as np
import pytest


def test_high_variance_target_gets_r2_near_one():
    """A linearly-recoverable, high-variance target should report r2≈1 and a
    small rmse/std — the scale-free check that prevents raw-RMSE misreads."""
    from swm_eventsat.models.probes import fit_ridge_probe

    rng = np.random.default_rng(0)
    E, T, D = 4, 500, 16
    latents = rng.standard_normal((E, T, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32)
    # Single high-variance target that is an exact affine function of the latents.
    target = (latents @ w + 3.0)[..., None].astype(np.float32)

    fit = fit_ridge_probe(latents, target, attribute_names=["signal"], ridge=1e-6)

    assert "signal" not in fit.degenerate
    assert fit.r2["signal"] > 0.99
    assert fit.rmse_over_std["signal"] < 0.1


def test_zero_variance_target_flagged_degenerate_not_perfect():
    """A constant target must be reported as degenerate (r2=nan), never as a
    silent rmse≈0 'perfect' fit."""
    from swm_eventsat.models.probes import fit_ridge_probe

    rng = np.random.default_rng(1)
    E, T, D = 4, 500, 16
    latents = rng.standard_normal((E, T, D)).astype(np.float32)
    good = (latents @ rng.standard_normal(D))[..., None]
    dead = np.full((E, T, 1), 0.0, dtype=np.float32)
    targets = np.concatenate([good, dead], axis=-1).astype(np.float32)

    with pytest.warns(RuntimeWarning, match="degenerate"):
        fit = fit_ridge_probe(latents, targets, attribute_names=["good", "dead"], ridge=1e-6)

    assert fit.degenerate == ["dead"]
    assert math.isnan(fit.r2["dead"])
    assert math.isnan(fit.rmse_over_std["dead"])
    # The healthy column is unaffected.
    assert "good" not in fit.degenerate
    assert fit.r2["good"] > 0.99
