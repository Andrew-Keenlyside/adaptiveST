"""adaptiveST: grids, eigenvalues, the local noise, fixed vs adaptive, chunking, masking, both lanes."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

import adaptivest as ast
from adaptivest import adaptive as ad
from adaptivest import local_noise as ln

#: Small settings so the CPU tests stay quick: rho up to 4, a 32-voxel noise window.
SCALES = ast.AdaptiveScales(precision_deg=15.0, truncate=2.0, rho_max=4.0, noise_window=32.0)


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
    e = ast.eigen(ast.structure_tensor(vol, sigma=1.0, rho=2.0, gpu=False))
    core = (slice(None), *(slice(10, 30),) * 3)
    assert np.median(np.abs(e["direction"][core][2])) > 0.95  # tubes run along x


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(0)), 0, 1)))


def test_halo_is_the_chain_plus_the_noise_pooling():
    assert ad.chain_halo(16.0, 3.0) == 94
    assert ast.AdaptiveScales().halo() == 94 + 64 + 8
    assert ast.AdaptiveScales.from_dict(SCALES.as_dict()) == SCALES
    old = {"precision_deg": 15.0, "truncate": 3.0, "k_sigma": [0.5], "k": [[1.0]], "noise_window": 0.0}
    assert ast.AdaptiveScales.from_dict(old) == ast.AdaptiveScales()


def test_band_edges_cover_the_window_resolution_to_the_corner():
    e = ln.band_edges(16.0, 128.0)
    assert len(e) == 51 and np.all(np.diff(e) > 0)
    assert e[0] == pytest.approx(3.0 / (2 * np.pi * 128.0 / 6.0)) and e[-1] > np.sqrt(3) / 2


def test_local_noise_of_white_noise_is_its_variance():
    """Pure white noise of variance 4: every band's noise power is 4 per unit spectral density, all of the
    gradient power is noise, and the drawn layer has variance 4 (away from the volume's tapered faces)."""
    rng = np.random.default_rng(2)
    vol = 2.0 * rng.standard_normal((112, 112, 112)).astype(np.float32)
    noise = ln.LocalNoise(vol, 4.0, 32.0, gaussian_filter, np)
    inner = (slice(24, -24),) * 3  # cells of 2 voxels: 48 voxels in
    psd = np.median(noise.Pn[(slice(None), *inner)] / noise.frac[:, None, None, None], axis=(1, 2, 3))
    assert np.all((psd > 3.4) & (psd < 4.6))
    assert np.median(noise.share()[48:-48, 48:-48, 48:-48]) > 0.95
    assert 3.6 < noise.layer()[48:-48, 48:-48, 48:-48].var() < 4.4


def test_local_noise_is_not_biased_low_by_a_patchy_noise_level():
    """White noise whose level varies by 20% over a sub-window's width: the plain minimum would pick the quiet
    patches; with each sub-window divided by its own noise level, the estimate stays at the window's power."""
    rng = np.random.default_rng(4)
    m = gaussian_filter(rng.standard_normal((112, 112, 112)), 8.0, mode="wrap")
    vol = ((1.0 + 0.2 * m / m.std()) * rng.standard_normal((112, 112, 112))).astype(np.float32)
    noise = ln.LocalNoise(vol, 4.0, 32.0, gaussian_filter, np)
    inner = (slice(24, -24),) * 3
    ratio = noise.Pn.sum(0)[inner] / noise.Pt.sum(0)[inner]
    assert 0.9 < np.median(ratio) <= 1.0


def test_local_noise_follows_a_step_in_the_noise():
    n = 96
    rng = np.random.default_rng(3)
    amp = np.where(np.arange(n) < n // 2, 0.1, 0.2)[:, None, None]
    vol = (_tubes(n, noise=0.0) + amp * rng.standard_normal((n, n, n))).astype(np.float32)
    noise = ln.LocalNoise(vol, 4.0, 32.0, gaussian_filter, np)
    hi_band = noise.Pn[-5:].sum(0)  # cells of 2 voxels; the step at cell 24, 20 voxels from it and the faces
    lo, hi = np.median(hi_band[9:15, 8:-8, 8:-8]), np.median(hi_band[33:39, 8:-8, 8:-8])
    assert 2.5 < hi / lo < 6.0  # variance ratio 4


def test_adaptive_tensor_agrees_between_chunkings():
    """The noise and K are window statistics with their own draw per tile: chunkings agree statistically."""
    vol = _tubes(30, noise=0.2)
    a, ma = ast.adaptive_structure_tensor(vol, SCALES, return_maps=True, gpu=False)
    b, mb = ast.adaptive_structure_tensor(vol, SCALES, chunk=15, return_maps=True, gpu=False)
    da, db = ast.eigen(a)["direction"], ast.eigen(b)["direction"]
    assert np.median(_angle(da, db)) < 2.0
    assert abs(np.median(np.log(mb["rho"] / ma["rho"]))) < 0.05
    assert ma["rho"].min() >= SCALES.rho_grid[0] - 1e-6 and ma["rho"].max() <= SCALES.rho_max + 1e-6
    assert np.isfinite(ma["noise"]).all() and 0 <= ma["noise"].min() and ma["noise"].max() <= 1 + 1e-6


def test_tensor_from_stored_scales_reproduces_the_one_shot_tensor():
    """Maps first, then the blend with only blend_halo of context: the tensor the whole chain gives."""
    vol = _tubes(30, noise=0.2)
    S, maps = ast.adaptive_structure_tensor(vol, SCALES, chunk=None, return_maps=True, gpu=False)
    h = ast.blend_halo(SCALES)
    assert h < SCALES.halo()
    pad = ad._read_padded(vol, (0, 0, 0), vol.shape, h)
    for dtype in (np.float32, np.float16):
        T = ast.tensor_from_scales(
            pad,
            maps["sigma"].astype(dtype),
            maps["rho"].astype(dtype),
            SCALES,
            gaussian_filter=gaussian_filter,
            tensor_dtype=np.float32,
            xp=np,
            margin=h,
        )[(slice(None), *(slice(h, -h),) * 3)]
        if dtype is np.float32:
            np.testing.assert_allclose(T, S, rtol=1e-5, atol=1e-7)
        assert np.percentile(_angle(ast.eigen(T)["direction"], ast.eigen(S)["direction"]), 99) < 0.5


def test_nan_scales_give_a_zero_tensor():
    vol = _tubes(20)
    sig = np.full((10,) * 3, np.nan, np.float32)
    T = ast.tensor_from_scales(
        vol[5:15, 5:15, 5:15], sig, sig, SCALES, gaussian_filter=gaussian_filter, tensor_dtype=np.float32, xp=np
    )
    assert not T.any()


def test_scale_maps_chunk_matches_the_volume_driver():
    vol = _tubes(30, noise=0.2)
    whole = ast.scale_maps(vol, SCALES, chunk=15, gpu=False)
    part = ast.scale_maps_chunk(vol, (15, 0, 15), (15, 15, 15), SCALES, gpu=False)
    for k in ("sigma", "rho", "noise"):
        np.testing.assert_array_equal(part[k], whole[k][15:, :15, 15:])


def test_one_pass_is_a_shorter_chain_that_agrees_with_two():
    one = ast.AdaptiveScales(**{**SCALES.as_dict(), "passes": 1})
    assert ast.AdaptiveScales.from_dict(one.as_dict()) == one
    assert one.halo() < SCALES.halo()
    assert ast.AdaptiveScales().halo() == 166 and ast.AdaptiveScales(passes=1).halo() == 154
    with pytest.raises(ValueError):
        ast.AdaptiveScales(passes=3)
    vol = _tubes(30, noise=0.2)
    a = ast.eigen(ast.adaptive_structure_tensor(vol, SCALES, gpu=False))["direction"]
    b, mb = ast.adaptive_structure_tensor(vol, one, return_maps=True, gpu=False)
    assert np.median(_angle(a, ast.eigen(b)["direction"])) < 3.0
    assert mb["rho"].min() >= one.rho_grid[0] - 1e-6 and mb["rho"].max() <= one.rho_max + 1e-6
    # The chunked one-pass maps: the same chain per chunk.
    part = ast.scale_maps_chunk(vol, (15, 0, 15), (15, 15, 15), one, gpu=False)
    whole = ast.scale_maps(vol, one, chunk=15, gpu=False)
    np.testing.assert_array_equal(part["rho"], whole["rho"][15:, :15, 15:])


def test_adaptive_finds_the_tube_direction():
    vol = _tubes(36, noise=0.1)
    e = ast.eigen(ast.adaptive_structure_tensor(vol, SCALES, gpu=False))
    core = (slice(None), *(slice(8, 28),) * 3)
    assert np.median(np.abs(e["direction"][core][2])) > 0.95


def test_more_noise_gives_larger_windows():
    quiet = ast.scale_maps(_tubes(30, noise=0.05), SCALES, gpu=False)["rho"]
    noisy = ast.scale_maps(_tubes(30, noise=2.0), SCALES, gpu=False)["rho"]
    assert np.median(noisy) > 2.0 * np.median(quiet)


@pytest.mark.gpu
@pytest.mark.xfail(
    reason="K is unreliable where a tile is edge-replicated along two axes (most of this small volume's tile): "
    "the padded voxels' perturbation factors dominate its ratio of sums, differently on the two backends",
    strict=False,
)
def test_gpu_matches_cpu():
    pytest.importorskip("cupy")
    vol = _tubes(30, noise=0.2)
    a, ma = ast.adaptive_structure_tensor(vol, SCALES, gpu=False, return_maps=True)
    b, mb = ast.adaptive_structure_tensor(vol, SCALES, gpu=True, return_maps=True)
    assert np.median(_angle(ast.eigen(a)["direction"], ast.eigen(b)["direction"])) < 0.5
    assert abs(np.median(np.log(mb["rho"] / ma["rho"]))) < 0.01


def test_full_mask_changes_nothing():
    vol = _tubes(30, noise=0.2)
    a, ma = ast.adaptive_structure_tensor(vol, SCALES, gpu=False, return_maps=True)
    b, mb = ast.adaptive_structure_tensor(
        vol, SCALES, mask=np.ones(vol.shape, bool), gpu=False, return_maps=True
    )
    assert np.median(_angle(ast.eigen(a)["direction"], ast.eigen(b)["direction"])) < 0.5
    assert abs(np.median(np.log(mb["rho"] / ma["rho"]))) < 0.01


def test_mask_isolates_the_kept_structure():
    """Tubes along x end at a bright slab: next to the edge the step's gradient (along x) pulls the free
    tensor off the tubes; masked to the tubes, the edge is never differentiated and the direction holds."""
    n = 36
    keep = np.broadcast_to(np.arange(n) < n // 2, (n, n, n)).copy()  # x < n/2
    vol = np.where(keep, _tubes(n, noise=0.05), 3.0).astype(np.float32)
    band = (slice(8, 28), slice(8, 28), slice(n // 2 - 3, n // 2))  # kept voxels next to the edge
    free = ast.eigen(ast.adaptive_structure_tensor(vol, SCALES, gpu=False))["direction"]
    held, maps = ast.adaptive_structure_tensor(vol, SCALES, mask=keep, return_maps=True, gpu=False)
    held = ast.eigen(held)["direction"]
    assert np.median(np.abs(held[2][band])) > 0.95 > np.median(np.abs(free[2][band]))
    assert np.isnan(held[:, ~keep]).all() and np.isnan(maps["rho"][~keep]).all()


def test_auto_chunk_fits_the_memory(monkeypatch):
    H = SCALES.halo()
    monkeypatch.setattr(ad, "_free_bytes", lambda xp: 10**12)
    assert ad.auto_chunk((40, 40, 40), SCALES, np) is None  # one tile
    tile = 32 + 2 * H + 5
    monkeypatch.setattr(
        ad, "_free_bytes", lambda xp: int(tile**3 * ad.BYTES_PER_TILE_VOXEL / ad.MEMORY_FRACTION)
    )
    assert ad.auto_chunk((4000, 4000, 4000), SCALES, np) == 32
