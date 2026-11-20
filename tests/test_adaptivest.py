"""adaptiveST: grids, eigenvalues, fixed vs adaptive, chunk invariance, calibration, both lanes."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import adaptivest as ast
from adaptivest import adaptive as ad

SCALES = ast.AdaptiveScales(
    precision_deg=15.0,
    truncate=2.0,
    k_sigma=(0.5, 1.0),
    k=((0.5, 0.3, 0.15, 0.08), (0.3, 0.15, 0.08, 0.04)),
    rho_max=4.0,
    noise_model="gaussian",
)


def _tubes(n=40, period=6.0, width=1.5, noise=0.2, seed=0):
    """Gaussian tubes along x on a square lattice, plus smoothed noise."""
    z, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dz = np.mod(z + period / 2, period) - period / 2
    dy = np.mod(y + period / 2, period) - period / 2
    img = np.exp(-(dz**2 + dy**2) / (2 * width**2))[:, :, None] * np.ones((1, 1, n))
    rng = np.random.default_rng(seed)
    return (img + noise * gaussian_filter(rng.standard_normal((n, n, n)), 0.7)).astype(np.float32)


def test_closed_form_eigenvalues_match_eigh():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(300, 3, 3))
    M = A @ np.swapaxes(A, 1, 2)
    S = np.stack([M[:, 0, 0], M[:, 1, 1], M[:, 2, 2], M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]])
    np.testing.assert_allclose(np.stack(ad.eigvals_sym3(S, np), 1), np.linalg.eigvalsh(M)[:, ::-1], atol=1e-9)


def test_grids_are_log_spaced_at_root_two():
    r, s = ast.rho_grid(16.0), ast.sigma_grid(16.0)
    assert r[0] == pytest.approx(np.sqrt(2)) and r[-1] == pytest.approx(16.0)
    assert s[0] == pytest.approx(0.5) and s[-1] == pytest.approx(16.0 / np.sqrt(2))
    for g in (r, s):
        assert np.allclose(np.diff(np.log(g)), np.log(np.sqrt(2)))


def test_fixed_tensor_finds_the_tube_direction():
    vol = _tubes(noise=0.05)
    e = ast.eigen(ast.structure_tensor(vol, sigma=1.0, rho=2.0))
    core = (slice(None), *(slice(10, 30),) * 3)
    assert np.median(np.abs(e["direction"][core][2])) > 0.95  # tubes run along x


def test_adaptive_tensor_does_not_depend_on_the_chunking():
    vol = _tubes(30)
    a, ma = ast.adaptive_structure_tensor(vol, SCALES, return_maps=True)
    b, mb = ast.adaptive_structure_tensor(vol, SCALES, chunk=11, return_maps=True)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ma["rho"], mb["rho"])
    assert ma["rho"].min() >= SCALES.rho_grid[0] - 1e-6 and ma["rho"].max() <= SCALES.rho_max + 1e-6


def test_adaptive_finds_the_tube_direction():
    vol = _tubes(36, noise=0.1)
    e = ast.eigen(ast.adaptive_structure_tensor(vol, SCALES))
    core = (slice(None), *(slice(8, 28),) * 3)
    assert np.median(np.abs(e["direction"][core][2])) > 0.95


def test_calibration_is_deterministic_and_complete():
    vol = _tubes(64, noise=0.3, seed=4)
    kw = dict(n_blocks=1, block=12, seed=0, rho_max=4.0, truncate=2.0, noise_model="gaussian")
    a, b = ast.calibrate(vol, **kw), ast.calibrate(vol, **kw)
    assert a == b and a.k_sigma[0] == ast.SIGMA0
    assert all(len(row) == len(a.rho_grid) and all(k > 0 for k in row) for row in a.k)
    assert ast.AdaptiveScales.from_dict(a.as_dict()) == a


@pytest.mark.gpu
def test_gpu_matches_cpu():
    pytest.importorskip("cupy")
    vol = _tubes(30)
    a = ast.adaptive_structure_tensor(vol, SCALES)
    b = ast.adaptive_structure_tensor(vol, SCALES, gpu=True)
    assert np.abs(a - b).max() <= 1e-4 * np.abs(a).max()
