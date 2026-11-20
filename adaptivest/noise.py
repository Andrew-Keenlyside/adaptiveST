"""The image's noise, from its own high-frequency spectrum.

Without half-maps the noise cannot be separated from the signal, so it is
bounded from above twice:

1. a power law ``P(k) = A k^slope`` is fitted to the median radially averaged
   power spectrum of sampled blocks between 0.30 and 0.48 cycles/voxel, where
   noise dominates; it is held flat below 0.1 cycles/voxel and its slope is
   clipped to ``[-4, 0]`` (a steeper tail is probably signal);
2. along a fibre the signal does not vary, so the structure tensor's smallest
   eigenvalue there is the noise's gradient energy along the fibre, a third of
   the noise tensor's trace for isotropic noise. The mean smallest eigenvalue
   over the most fibre-like tenth of the voxels bounds that third from above;
   if a draw from the fit carries more, the fit is scaled down.

The model is isotropic; :func:`axis_anisotropy` measures how far the image's
high-frequency power departs from that.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .tensor import eigvals_sym3, gradients, tensor_from_gradients

BAND = (0.30, 0.48)
K_FLAT = 0.1
CONE_DEG = 25.0
ANISOTROPY_LIMIT = 3.0


def _windowed_power(block):
    b = np.asarray(block, dtype=np.float64)
    b = b - b.mean()
    win = np.ones_like(b)
    for ax, n in enumerate(b.shape):
        shape = [1, 1, 1]
        shape[ax] = n
        win = win * np.hanning(n).reshape(shape)
    power = np.abs(np.fft.fftn(b * win)) ** 2 / (win**2).sum()
    freqs = np.meshgrid(*[np.fft.fftfreq(n) for n in b.shape], indexing="ij")
    return power, freqs


def axis_anisotropy(blocks, *, band=BAND, cone_deg=CONE_DEG):
    """``(ratio, per-axis power)``: largest over smallest high-frequency power along the image axes."""
    per_block = []
    for blk in blocks:
        power, f = _windowed_power(blk)
        k = np.sqrt(sum(x**2 for x in f))
        inband = (k >= band[0]) & (k <= band[1])
        cos_min = np.cos(np.deg2rad(cone_deg))
        row = []
        for ax in range(3):
            sel = inband & (np.abs(f[ax]) >= cos_min * k)
            row.append(float(np.median(power[sel])) if sel.any() else np.nan)
        per_block.append(row)
    axes = np.nanmedian(np.asarray(per_block, float), axis=0)
    if not np.isfinite(axes).all() or axes.min() <= 0:
        return float("nan"), tuple(float(v) for v in axes)
    return float(axes.max() / axes.min()), tuple(float(v) for v in axes)


def radial_power(block, nbins: int = 32):
    """``(k, P)``: radially averaged power spectrum, cycles/voxel and power per voxel."""
    power, freqs = _windowed_power(block)
    kmag = np.sqrt(sum(f**2 for f in freqs))
    edges = np.linspace(0.0, 0.5, nbins + 1)
    idx = np.digitize(kmag.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    sums = np.bincount(idx[ok], weights=power.ravel()[ok], minlength=nbins)
    counts = np.bincount(idx[ok], minlength=nbins)
    k = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        return k, sums / counts


@dataclass(frozen=True)
class NoiseModel:
    """``P(k) = amplitude * max(k, k_flat) ** slope``: the noise spectrum, power per voxel."""

    slope: float
    amplitude: float
    k_flat: float = K_FLAT
    slope_clipped: bool = False
    scale: float = 1.0
    anisotropy: float = float("nan")

    @property
    def anisotropic(self) -> bool:
        return bool(np.isfinite(self.anisotropy) and self.anisotropy > ANISOTROPY_LIMIT)

    def power(self, k):
        kk = np.maximum(np.asarray(k, dtype=np.float64), self.k_flat)
        return self.amplitude * kk**self.slope

    def draw(self, shape, rng, factor: float = 1.0):
        """One realisation with this spectrum, float32."""
        w = rng.standard_normal(shape)
        freqs = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
        kmag = np.sqrt(sum(f**2 for f in freqs))
        return np.fft.ifftn(np.fft.fftn(w) * np.sqrt(self.power(kmag) * factor)).real.astype(np.float32)

    def as_dict(self) -> dict:
        return asdict(self)


def _interior_eigvals(block, sigma, rho, truncate):
    from scipy.ndimage import gaussian_filter

    m = int(np.ceil((2.0 / 3.0) * truncate * max(sigma, rho)))
    if min(block.shape) <= 2 * m + 2:
        return None
    S = tensor_from_gradients(
        gradients(block, sigma, truncate, gaussian_filter, np), rho, truncate, gaussian_filter, np, np.float64
    )
    S = S[(slice(None), *(slice(m, -m),) * 3)]
    return np.stack(eigvals_sym3(S, np)).reshape(3, -1)


def _fibre_lambda3(block, sigma, rho, truncate, top=0.1):
    lam = _interior_eigvals(np.asarray(block, np.float32), sigma, rho, truncate)
    if lam is None:
        return float("nan")
    ok = np.isfinite(lam).all(axis=0) & (lam[0] > 0)
    if not ok.any():
        return float("nan")
    lam = lam[:, ok]
    sel = lam[1] >= np.quantile(lam[1], 1.0 - top)
    return float(np.mean(lam[2, sel]))


def estimate_noise(blocks, *, sigma: float = 1.0, truncate: float = 3.0, seed: int = 0) -> NoiseModel:
    """The noise model of an image, from some of its blocks (32^3 or more recommended)."""
    spectra = [radial_power(b) for b in blocks]
    k = spectra[0][0]
    stack = np.stack([P for _, P in spectra])
    P = np.full(k.shape, np.nan)
    has = np.isfinite(stack).any(axis=0)
    P[has] = np.nanmedian(stack[:, has], axis=0)
    sel = (k >= BAND[0]) & (k <= BAND[1]) & np.isfinite(P) & (P > 0)
    if sel.sum() < 2:
        raise ValueError("the blocks' spectra have no usable high-frequency tail (blocks too small or flat)")
    slope, intercept = np.polyfit(np.log(k[sel]), np.log(P[sel]), 1)
    clipped = not -4.0 <= slope <= 0.0
    if clipped:  # keep the band-centre power, change only the extrapolation
        k0 = float(np.exp(np.log(k[sel]).mean()))
        p0 = float(np.exp(intercept + slope * np.log(k0)))
        slope = float(np.clip(slope, -4.0, 0.0))
        intercept = np.log(p0) - slope * np.log(k0)
    model = NoiseModel(float(slope), float(np.exp(intercept)), K_FLAT, bool(clipped))
    rho = 3.0 * sigma
    nu3_img = float(np.nanmedian([_fibre_lambda3(b, sigma, rho, truncate) for b in blocks]))
    draw = model.draw(tuple(int(min(64, s)) for s in blocks[0].shape), np.random.default_rng(seed))
    lam = _interior_eigvals(draw, sigma, rho, truncate)
    nu3_tail = float(np.mean(lam.sum(0)) / 3.0) if lam is not None else float("nan")
    scale = 1.0
    if np.isfinite(nu3_img) and np.isfinite(nu3_tail) and 0 < nu3_img < nu3_tail:
        scale = nu3_img / nu3_tail
    ratio, _ = axis_anisotropy(blocks)
    return NoiseModel(
        float(slope), float(np.exp(intercept)) * scale, K_FLAT, bool(clipped), float(scale), ratio
    )
