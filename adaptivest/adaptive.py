"""Locally adaptive structure-tensor scales: each voxel its own ``sigma`` and ``rho``, from the image.

Every voxel gets its own integration scale ``rho`` and derivative scale ``sigma``,
from the image alone, in voxels. Nothing is calibrated per dataset but the noise
factors ``K``, measured once at the start of the stage.

**The grids** are sampled in log scale at ratio sqrt(2) (two samples per octave,
the usual scale-space sampling): ``rho`` from :data:`RHO_MIN` = sqrt(2)
(Lindeberg's smallest integration factor over a unit derivative scale) to the
configured ``rho_max``; ``sigma`` from :data:`SIGMA0` = 0.5 to
``rho_max / sqrt(2)``. Both lie on the lattice ``2^(k/2)`` when ``rho_max`` does.

**rho(x): the smallest window that reaches a target angular precision.** At a
derivative scale ``s``, the tensor at each grid ``rho`` predicts the angular
variance of its smallest eigenvector as ``V = K(s, rho) * S``, with ``S =
sum_i (l_i + l_3) / (l_i - l_3)^2`` (first-order eigenvector perturbation:
large eigenvalue gaps, a well-determined direction) and ``K(s, rho)`` fitted
once so that ``V`` reproduces the mean squared angle by which one added draw of
the image's own noise moves the direction. ``V`` is smoothed over
:data:`V_SMOOTH` voxels; ``rho(x)`` is where it first falls to
``precision_deg^2``, interpolated in ``log rho`` (the grid's top where it never
does), and ``log rho`` is smoothed over :data:`RHO_SMOOTH` voxels.

**sigma(x): Lindeberg's scale over the voxel's own window.** The first interior
peak over ``s`` of the normalised gradient energy ``s^2 |grad L_s|^2``, averaged
with a Gaussian of width ``rho(x)`` (windows at the grid ``rho`` blended in
``log rho``), refined by a parabola in ``log s``; the largest response where
there is no interior peak.

**Three passes**, so the first ``rho`` is not skewed by an assumed ``sigma``:
``rho`` at :data:`SIGMA0` (the smallest derivative scale, nearly no
smoothing); ``sigma`` over that ``rho``; ``rho`` again at ``sigma(x)`` (``K``
blended between the calibrated sigmas). The tensor then blends the
trace-normalised tensors at the neighbouring grid ``(sigma, rho)`` bilinearly
in ``(log sigma, log rho)``, and :func:`adaptivest.eigen` reads FA and the
direction from it as from a fixed one.

**Chunking cannot change the result**: every step is a local filter or a
per-voxel operation, the noise factors are global, and :meth:`AdaptiveScales.halo`
is the full footprint of the chain.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

SQRT2 = math.sqrt(2.0)
#: The smallest integration scale (voxels): Lindeberg's smallest integration factor sqrt(2).
RHO_MIN = SQRT2
#: The first pass's derivative scale and the smallest of the sigma grid (voxels).
SIGMA0 = 0.5
#: Default largest integration scale (voxels).
RHO_MAX = 16.0
#: Smoothing of the predicted variance and of log rho (voxels).
V_SMOOTH = 2.0
RHO_SMOOTH = 2.0
#: E[theta^2] of a random axis against a fixed one: no prediction is worse.
V_RANDOM = math.pi - 2.0
#: Default target angular precision (degrees).
PRECISION_DEG = 15.0
#: Width (voxels) of the local mean intensity the Poisson noise model scales the variance by.
INTENSITY_SMOOTH = 2.0
#: Noise models: "poisson" (variance proportional to the local mean intensity) or "gaussian" (stationary).
NOISE_MODELS = ("poisson", "gaussian")

# Tensor component order: [zz, yy, xx, zy, zx, yx].
_PRODUCTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _radius(s: float, truncate: float) -> int:
    return int(float(truncate) * float(s) + 0.5)


def log_grid(lo: float, hi: float, ratio: float = SQRT2) -> tuple:
    """``lo`` to ``hi`` sampled uniformly in log scale, steps at most *ratio* (both ends included)."""
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        return (lo,)
    n = int(math.ceil(math.log(hi / lo) / math.log(ratio) - 1e-9))
    return tuple(float(x) for x in np.geomspace(lo, hi, n + 1))


def rho_grid(rho_max: float) -> tuple:
    return log_grid(RHO_MIN, rho_max)


def sigma_grid(rho_max: float) -> tuple:
    return log_grid(SIGMA0, max(SIGMA0, float(rho_max) / SQRT2))


@dataclass(frozen=True)
class AdaptiveScales:
    """The per-run constants: the precision target, the grids' top and the noise factors ``K(sigma, rho)``.

    ``k_sigma`` are the derivative scales ``K`` was calibrated at (always
    :data:`SIGMA0` first); ``k[i][j]`` is ``K`` at ``k_sigma[i]`` and the
    ``j``-th grid rho.
    """

    precision_deg: float
    truncate: float
    k_sigma: tuple
    k: tuple
    rho_max: float = RHO_MAX
    noise: dict = field(default_factory=dict)
    #: "poisson": the noise variance scales with the local mean intensity (over
    #: INTENSITY_SMOOTH voxels) relative to *intensity_ref*, the calibration
    #: tissue's mean; "gaussian": the same everywhere.
    noise_model: str = "poisson"
    intensity_ref: float = 1.0

    @property
    def rho_grid(self) -> tuple:
        return rho_grid(self.rho_max)

    @property
    def sigma_grid(self) -> tuple:
        return sigma_grid(self.rho_max)

    def halo(self) -> int:
        """The full footprint of the three passes and the blend (voxels)."""
        return chain_halo(self.rho_max, self.truncate)

    def equivalent_scales(self) -> tuple[float, float]:
        """``(sigma, rho)`` whose fixed-scale halo (``st_halo_voxels``) equals :meth:`halo`, for the stage's planners."""
        sig = float(max(self.sigma_grid))
        return sig, (self.halo() - _radius(sig, self.truncate)) / float(self.truncate)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["k_sigma"] = [float(x) for x in self.k_sigma]
        d["k"] = [[float(x) for x in row] for row in self.k]
        return d

    @classmethod
    def from_dict(cls, d: dict | AdaptiveScales | None) -> AdaptiveScales | None:
        if d is None or isinstance(d, AdaptiveScales):
            return d
        return cls(
            precision_deg=float(d["precision_deg"]),
            truncate=float(d["truncate"]),
            k_sigma=tuple(float(x) for x in d["k_sigma"]),
            k=tuple(tuple(float(x) for x in row) for row in d["k"]),
            rho_max=float(d.get("rho_max", RHO_MAX)),
            noise=dict(d.get("noise", {})),
            noise_model=str(d.get("noise_model", "poisson")),
            intensity_ref=float(d.get("intensity_ref", 1.0)),
        )


def chain_halo(rho_max: float, truncate: float) -> int:
    """Voxels of context the chain reads: rho (first pass), sigma over it, rho again at sigma(x), the blend."""
    t = float(truncate)
    r_rho = _radius(max(rho_grid(rho_max)), t)
    r_sig = _radius(max(sigma_grid(rho_max)), t)
    smooth = _radius(V_SMOOTH, t) + _radius(RHO_SMOOTH, t)
    first = _radius(SIGMA0, t) + r_rho + smooth  # rho at SIGMA0, at a voxel
    sig = max(first, r_sig + r_rho)  # sigma at a voxel: rho there, and the windowed energies
    second = smooth + max(sig, r_sig + r_rho)  # rho again: V at neighbours, each at its own sigma
    return max(second, r_sig + r_rho)  # the blend at a voxel


# ---------------------------------------------------------------------------
# Array-module helpers (numpy or cupy)
# ---------------------------------------------------------------------------


def _xp_of(a):
    try:
        import cupy

        return cupy.get_array_module(a)
    except ImportError:  # pragma: no cover - CPU-only installs
        return np


def _gradients(vol, s, truncate, gaussian_filter, xp):
    v = xp.asarray(vol, dtype=xp.float32)
    return (
        gaussian_filter(v, s, order=[1, 0, 0], mode="nearest", truncate=truncate),
        gaussian_filter(v, s, order=[0, 1, 0], mode="nearest", truncate=truncate),
        gaussian_filter(v, s, order=[0, 0, 1], mode="nearest", truncate=truncate),
    )


def _tensor(grads, r, truncate, gaussian_filter, xp, dtype):
    """(6, ...) tensor at window *r* from one set of gradients."""
    S = xp.empty((6,) + grads[0].shape, dtype=dtype)
    for i, (p, q) in enumerate(_PRODUCTS):
        gaussian_filter(
            (grads[p] * grads[q]).astype(dtype, copy=False), r, mode="nearest", output=S[i], truncate=truncate
        )
    return S


def eigvals_sym3(S, xp):
    """``(l1, l2, l3)`` descending, of the symmetric tensors ``S`` (6, ...) [zz, yy, xx, zy, zx, yx]; closed form."""
    a11, a22, a33, a12, a13, a23 = (S[i].astype(xp.float64, copy=False) for i in range(6))
    p1 = a12 * a12 + a13 * a13 + a23 * a23
    q = (a11 + a22 + a33) / 3.0
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    p2 = b11 * b11 + b22 * b22 + b33 * b33 + 2.0 * p1
    p = xp.sqrt(p2 / 6.0)
    safe = p > 1e-300
    inv = xp.where(safe, 1.0 / xp.where(safe, p, 1.0), 0.0)
    c11, c22, c33, c12, c13, c23 = b11 * inv, b22 * inv, b33 * inv, a12 * inv, a13 * inv, a23 * inv
    det = c11 * (c22 * c33 - c23 * c23) - c12 * (c12 * c33 - c23 * c13) + c13 * (c12 * c23 - c22 * c13)
    r = xp.clip(det / 2.0, -1.0, 1.0)
    phi = xp.arccos(r) / 3.0
    l1 = q + 2.0 * p * xp.cos(phi)
    l3 = q + 2.0 * p * xp.cos(phi + 2.0 * math.pi / 3.0)
    l2 = 3.0 * q - l1 - l3
    return l1, l2, l3


def perturbation_factor(l1, l2, l3, xp):
    """``S = (l1 + l3)/(l1 - l3)^2 + (l2 + l3)/(l2 - l3)^2``: small where the smallest eigenvector is well determined."""
    eps = 1e-12 * xp.maximum(xp.abs(l1), 1e-300)
    return (l1 + l3) / xp.maximum(l1 - l3, eps) ** 2 + (l2 + l3) / xp.maximum(l2 - l3, eps) ** 2


def _log_weights(x, grid, xp):
    """{index: weight array} for linear interpolation of ``log x`` on ``log grid`` (clamped)."""
    lg = np.log(np.asarray(grid, float))
    if len(lg) == 1:
        return {0: xp.ones(x.shape, dtype=xp.float64)}
    lx = xp.clip(xp.log(x), float(lg[0]), float(lg[-1]))
    out = {}
    for k in range(len(lg) - 1):
        lo, hi = float(lg[k]), float(lg[k + 1])
        inside = (lx >= lo) & ((lx < hi) | ((k == len(lg) - 2) & (lx <= hi)))
        t = (lx - lo) / (hi - lo)
        for j, w in ((k, xp.where(inside, 1.0 - t, 0.0)), (k + 1, xp.where(inside, t, 0.0))):
            if bool(w.any()):
                out[j] = out[j] + w if j in out else w
    return out


def noise_scale(block, scales: AdaptiveScales, gaussian_filter, xp):
    """The noise variance at each voxel relative to the calibration's: 1 (gaussian), or the local mean
    intensity over :data:`INTENSITY_SMOOTH` voxels relative to ``intensity_ref`` (poisson)."""
    if scales.noise_model != "poisson":
        return 1.0
    m = gaussian_filter(
        xp.asarray(block, dtype=xp.float32), INTENSITY_SMOOTH, mode="nearest", truncate=scales.truncate
    )
    return xp.maximum(m, 1e-6 * abs(scales.intensity_ref) + 1e-12) / scales.intensity_ref


# ---------------------------------------------------------------------------
# The per-tile passes
# ---------------------------------------------------------------------------


def _variance_at(grads, k_index, kk, scales, g, gaussian_filter, xp):
    """The smoothed predicted variance at the *k_index*-th grid rho, from gradients at one derivative scale."""
    t = scales.truncate
    S = _tensor(grads, scales.rho_grid[k_index], t, gaussian_filter, xp, xp.float32)
    l1, l2, l3 = eigvals_sym3(S, xp)
    del S
    V = xp.clip(kk * g * perturbation_factor(l1, l2, l3, xp), 1e-8, V_RANDOM)
    del l1, l2, l3
    V = xp.where(xp.isfinite(V), V, V_RANDOM).astype(xp.float32)
    return gaussian_filter(V, V_SMOOTH, mode="nearest", truncate=t)


def _inset(a, m):
    """*a* without *m* voxels at every face (the whole of it when *m* is 0 or too large)."""
    m = int(m)
    if m <= 0 or any(n <= 2 * m for n in a.shape):
        return a
    return a[(slice(m, -m),) * a.ndim]


def rho_map(block, scales: AdaptiveScales, gaussian_filter, xp, sigma=None, *, early_stop=True, focus=0):
    """rho(x) (voxels): at :data:`SIGMA0` when *sigma* is None, else with each voxel's derivative scale
    ``sigma(x)`` (V blended between the calibrated sigmas, clamped to their range).

    V is computed one grid rho at a time and the search stops once every voxel
    of *block* inside *focus* voxels of its faces has reached the precision
    (*early_stop*): the later rhos cannot change a voxel that has, so there the
    result is the same either way. Outside that inset the result is not used.
    """
    t = scales.truncate
    g = noise_scale(block, scales, gaussian_filter, xp)
    ks = scales.k_sigma
    if sigma is None or len(ks) == 1:
        sources = [(None, _gradients(block, ks[0], t, gaussian_filter, xp), scales.k[0])]
    else:
        ws = _log_weights(xp.clip(sigma, min(ks), max(ks)), ks, xp)
        sources = [(w, _gradients(block, ks[i], t, gaussian_filter, xp), scales.k[i]) for i, w in ws.items()]
    target = 2.0 * math.log(math.radians(scales.precision_deg))
    lr = np.log(np.asarray(scales.rho_grid, float))
    lrho, prev = None, None
    for k in range(len(lr)):
        if sources[0][0] is None:
            v = _variance_at(sources[0][1], k, sources[0][2][k], scales, g, gaussian_filter, xp)
        else:
            v = xp.zeros(block.shape, dtype=xp.float32)
            for w, grads, kvec in sources:
                v += (w * _variance_at(grads, k, kvec[k], scales, g, gaussian_filter, xp)).astype(xp.float32)
        lv = xp.log(xp.maximum(v, 1e-8))
        del v
        hit = lv <= target
        if lrho is None:
            lrho = xp.where(hit, float(lr[0]), xp.nan)
        else:
            new = hit & xp.isnan(lrho)
            f = xp.clip((prev - target) / xp.maximum(prev - lv, 1e-12), 0.0, 1.0)
            lrho = xp.where(new, float(lr[k - 1]) + f * float(lr[k] - lr[k - 1]), lrho)
        prev = lv
        if early_stop and not bool(xp.isnan(_inset(lrho, focus)).any()):
            break
    del sources, prev
    lrho = xp.where(xp.isnan(lrho), float(lr[-1]), lrho)
    return xp.exp(gaussian_filter(lrho, RHO_SMOOTH, mode="nearest", truncate=t))


def sigma_map(block, rho, scales: AdaptiveScales, gaussian_filter, xp, *, early_stop=True, focus=0):
    """sigma(x) (voxels): first interior peak of the Lindeberg response averaged over a window of rho(x).

    Scales are taken in increasing order and the search stops once every voxel
    of *block* inside *focus* voxels of its faces has an interior peak
    (*early_stop*): only the first peak and its two neighbours are read, so
    there the result is the same either way. Outside that inset it is not used.
    """
    t = scales.truncate
    grid = scales.sigma_grid
    wr = _log_weights(rho, scales.rho_grid, xp)
    R = []
    has = None
    for n, s in enumerate(grid):
        gz, gy, gx = _gradients(block, s, t, gaussian_filter, xp)
        E = (s * s) * (gz * gz + gy * gy + gx * gx)
        del gz, gy, gx
        acc = xp.zeros(E.shape, dtype=xp.float32)
        for k, w in wr.items():
            acc += (w * gaussian_filter(E, scales.rho_grid[k], mode="nearest", truncate=t)).astype(xp.float32)
        R.append(acc)
        del E
        if n >= 2:
            peak = (R[n - 1] > R[n - 2]) & (R[n - 1] >= R[n])
            has = peak if has is None else has | peak
            if early_stop and bool(_inset(has, focus).all()) and n + 1 < len(grid):
                break
    if len(grid) < 3:
        best = grid[int(np.argmax([float(r.mean()) for r in R]))]
        return xp.full(rho.shape, best, dtype=xp.float64)
    used = np.asarray(grid[: len(R)], float)
    R = xp.stack(R)
    K = R.shape[0]
    ux = xp.asarray(np.log(used))
    is_max = xp.zeros(R.shape, dtype=bool)
    is_max[1:-1] = (R[1:-1] > R[:-2]) & (R[1:-1] >= R[2:])
    has = is_max.any(axis=0)
    i = xp.where(has, xp.argmax(is_max, axis=0), xp.argmax(R, axis=0))
    ic = xp.clip(i, 1, K - 2)
    y0, y1, y2 = (xp.take_along_axis(R, (ic + d)[None], axis=0)[0].astype(xp.float64) for d in (-1, 0, 1))
    u0, u1, u2 = ux[ic - 1], ux[ic], ux[ic + 1]
    den = (u0 - u1) * (u0 - u2) * (u1 - u2)
    A = (u2 * (y1 - y0) + u1 * (y0 - y2) + u0 * (y2 - y1)) / den
    B = (u2**2 * (y0 - y1) + u1**2 * (y2 - y0) + u0**2 * (y1 - y2)) / den
    uv = xp.clip(xp.where(A < 0, -B / xp.where(A < 0, 2.0 * A, 1.0), u1), u0, u2)
    return xp.exp(xp.where(has, uv, ux[i]))


def scale_maps(block, scales: AdaptiveScales, gaussian_filter, xp, *, iterations: int = 1, trace=None):
    """``(sigma(x), rho(x))``: rho at SIGMA0, sigma over it, then *iterations* x (rho at sigma(x)[, sigma again]).

    Production runs one iteration and keeps the first pass's sigma. *trace*
    (a list) receives ``(sigma, rho)`` after every pass, recomputing sigma over
    each new rho, for studying convergence.
    """
    # What the core needs: rho and sigma of the first pass within the smoothings of it, the
    # second rho within its own smoothing. The early stops look only there.
    t = scales.truncate
    smooth = _radius(V_SMOOTH, t) + _radius(RHO_SMOOTH, t)
    halo = scales.halo()
    f1 = max(0, halo - smooth) if trace is None else 0
    f2 = max(0, halo - _radius(RHO_SMOOTH, t)) if trace is None else 0
    rho = rho_map(block, scales, gaussian_filter, xp, focus=f1)
    sig = sigma_map(block, rho, scales, gaussian_filter, xp, focus=f1)
    if trace is not None:
        trace.append((sig, rho))
    for it in range(int(iterations)):
        rho = rho_map(block, scales, gaussian_filter, xp, sigma=sig, focus=f2 if it + 1 == iterations else f1)
        if trace is not None or it + 1 < iterations:
            sig_next = sigma_map(block, rho, scales, gaussian_filter, xp, focus=f1)
            if trace is not None:
                trace.append((sig_next, rho))
            if it + 1 < iterations:
                sig = sig_next
    return sig, rho


def adaptive_tensor_block(
    block, scales: AdaptiveScales, *, gaussian_filter, tensor_dtype, xp=None, maps=None
):
    """``S`` (6, *block.shape) at *tensor_dtype*: the tensor at each voxel's own (sigma, rho), blended.

    *maps*, a dict, receives ``rho`` and ``sigma`` when given (for diagnostics).
    """
    xp = _xp_of(block) if xp is None else xp
    t = scales.truncate
    sig, rho = scale_maps(block, scales, gaussian_filter, xp)
    if maps is not None:
        maps["rho"], maps["sigma"] = rho, sig
    ws = _log_weights(sig, scales.sigma_grid, xp)
    wr = _log_weights(rho, scales.rho_grid, xp)
    del rho, sig
    out = xp.zeros((6,) + tuple(block.shape), dtype=xp.float32)
    for i, wsi in ws.items():
        pairs = [(k, wrk) for k, wrk in wr.items() if bool((wsi * wrk).any())]
        if not pairs:
            continue
        grads = _gradients(block, scales.sigma_grid[i], t, gaussian_filter, xp)
        for k, wrk in pairs:
            S = _tensor(grads, scales.rho_grid[k], t, gaussian_filter, xp, xp.float32)
            tr = S[0] + S[1] + S[2]
            w = ((wsi * wrk) / xp.where(tr > 0, tr, 1.0) * (tr > 0)).astype(xp.float32)
            out += w[None] * S
            del S, tr, w
        del grads
    return out.astype(tensor_dtype, copy=False)


# ---------------------------------------------------------------------------
# The per-run noise factors (host, once)
# ---------------------------------------------------------------------------


def _eig_host(S):
    """``(l desc (3, ...), smallest eigenvector (3, ...))`` of host tensors [zz, yy, xx, zy, zx, yx]."""
    M = np.empty(S.shape[1:] + (3, 3))
    idx = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}
    for (a, b), i in idx.items():
        M[..., a, b] = M[..., b, a] = S[i]
    w, v = np.linalg.eigh(M)
    return np.moveaxis(w[..., ::-1], -1, 0), np.moveaxis(v[..., :, 0], -1, 0)


def _freqs(shape):
    return (
        np.fft.fftfreq(shape[0])[:, None, None],
        np.fft.fftfreq(shape[1])[None, :, None],
        np.fft.rfftfreq(shape[2])[None, None, :],
    )


def _fft_gauss(vol, s, order=None):
    """Gaussian smoothing (and first derivatives, *order* per axis) by FFT: the calibration's estimates only."""
    from scipy import fft as sfft

    f = _freqs(vol.shape)
    H = np.exp(-2.0 * np.pi**2 * s * s * (f[0] ** 2 + f[1] ** 2 + f[2] ** 2))
    if order is not None:
        for a, o in enumerate(order):
            if o:
                H = H * (2j * np.pi * f[a])
    return sfft.irfftn(sfft.rfftn(vol, workers=8) * H, s=vol.shape, workers=8)


def _host_gauss(arr, s, order=None, mode="nearest", truncate=3.0, output=None):
    """A gaussian_filter for the calibration's host estimates: by FFT for all but the smallest scales."""
    from scipy.ndimage import gaussian_filter

    if float(s) > 1.5:
        res = _fft_gauss(np.asarray(arr, np.float64), float(s), order)
    else:
        res = gaussian_filter(arr, s, order=order or 0, mode=mode, truncate=truncate)
    if output is not None:
        output[...] = res
        return output
    return res


def _tensors_fft(vol, sigma, rhos, truncate):
    """Host tensors at (sigma, each rho), products FFT'd once (the calibration's estimate, not the tiles')."""
    from scipy import fft as sfft

    g = _gradients(vol, sigma, truncate, _host_gauss, np)
    P = [sfft.rfftn(np.asarray(g[a], np.float64) * g[b], workers=8) for a, b in _PRODUCTS]
    f = _freqs(vol.shape)
    k2 = f[0] ** 2 + f[1] ** 2 + f[2] ** 2
    out = {}
    for r in rhos:
        Hk = np.exp(-2.0 * np.pi**2 * r * r * k2)
        out[r] = np.stack([sfft.irfftn(p * Hk, s=vol.shape, workers=8) for p in P])
    return out


def _fit_k(th2, Sf):
    """K with mean(min(K S, V_RANDOM)) = mean(theta^2); a window already random saturates."""
    target = min(float(th2.mean()), 0.98 * V_RANDOM)
    lo, hi = 1e-12, 1e6
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if np.mean(np.minimum(mid * Sf, V_RANDOM)) < target else (lo, mid)
    return math.sqrt(lo * hi)


def _k_row(pads, tissues, draws, gs, s, rhos, truncate, H, b):
    """K at derivative scale *s* for each rho: the injected draws' angles against S (times the noise scale).

    Each block is cropped to the halo this *s* needs, not the full chain's."""
    h = min(H, _radius(s, truncate) + _radius(max(rhos), truncate) + 2)
    crop = (slice(H - h, H + b + h),) * 3
    inner = (slice(h, h + b),) * 3
    th2 = {r: [] for r in rhos}
    Sf = {r: [] for r in rhos}
    for q, m, d, g in zip(pads, tissues, draws, gs):
        T0 = _tensors_fft(q[crop], s, rhos, truncate)
        T1 = _tensors_fft((q + d)[crop], s, rhos, truncate)
        gi = g if np.isscalar(g) else g[crop][inner]
        for r in rhos:
            lam, e0 = _eig_host(T0[r][(slice(None), *inner)])
            _, e1 = _eig_host(T1[r][(slice(None), *inner)])
            S = perturbation_factor(lam[0], lam[1], lam[2], np) * gi
            ok = m & np.isfinite(S) & (S > 0)
            ang = np.arccos(np.clip(np.abs((e0 * e1).sum(0)), 0.0, 1.0))
            th2[r].append(ang[ok] ** 2)
            Sf[r].append(S[ok])
    return tuple(_fit_k(np.concatenate(th2[r]), np.concatenate(Sf[r])) for r in rhos)


def calibrate_noise(
    image,
    mask,
    *,
    precision_deg: float = PRECISION_DEG,
    truncate: float = 3.0,
    rho_max: float = RHO_MAX,
    noise_model: str = "poisson",
    n_blocks: int = 8,
    block: int = 32,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> AdaptiveScales:
    """The run's :class:`AdaptiveScales` from sampled masked blocks of *image*.

    ``K`` is fitted at :data:`SIGMA0` (the first pass), then at the grid sigmas
    the blocks' own sigma maps reach (the second pass blends between them).
    *image* and *mask* are level-0 arrays (lazy: only the sampled blocks are
    read). Deterministic for a given image, mask and *seed*. *noise_model*
    ``"poisson"`` scales each injected draw's variance by the local mean
    intensity (relative to the blocks' tissue mean) and fits ``K`` against
    ``S`` times that scale; ``"gaussian"`` injects the same noise everywhere.
    """
    if noise_model not in NOISE_MODELS:
        raise ValueError(f"noise_model is one of {NOISE_MODELS}, not {noise_model!r}")
    from .noise import estimate_noise as estimate_image_noise

    say = log or (lambda _m: None)
    rng = np.random.default_rng(seed)
    shape = np.array(image.shape[-3:])
    rhos, sigs = rho_grid(rho_max), sigma_grid(rho_max)
    H = _radius(max(sigs), truncate) + _radius(max(rhos), truncate) + 2
    b = int(min(block, *(shape - 2 * H)))
    if b < 8:
        raise ValueError(f"image {tuple(shape)} too small for the adaptive tensor's noise calibration")
    pads, tissues = [], []
    for _ in range(4000):
        o = np.array([rng.integers(H, n - b - H + 1) for n in shape])
        core = tuple(slice(x, x + b) for x in o)
        m = np.asarray(mask[core]) > 0
        if m.mean() > 0.5:
            pads.append(np.asarray(image[tuple(slice(x - H, x + b + H) for x in o)], np.float32))
            tissues.append(m)
        if len(pads) >= n_blocks:
            break
    if not pads:
        raise ValueError("no block of the mask is half tissue: cannot measure the image's noise")
    inner = (slice(H, H + b),) * 3
    model = estimate_image_noise([q[inner] for q in pads], sigma=1.0, truncate=truncate, seed=seed)
    local = [_host_gauss(q, INTENSITY_SMOOTH, truncate=truncate) for q in pads]
    ref = float(np.mean(np.concatenate([lm[inner][m] for lm, m in zip(local, tissues)])))
    probe = AdaptiveScales(
        precision_deg=float(precision_deg),
        truncate=float(truncate),
        k_sigma=(SIGMA0,),
        k=((1.0,) * len(rhos),),
        rho_max=float(rho_max),
        noise_model=noise_model,
        intensity_ref=ref,
    )
    gs = [noise_scale(q, probe, _host_gauss, np) for q in pads]
    draws = [
        model.draw(q.shape, rng) * (np.sqrt(g) if noise_model == "poisson" else 1.0) for q, g in zip(pads, gs)
    ]

    k0 = _k_row(pads, tissues, draws, gs, SIGMA0, rhos, truncate, H, b)
    # Which sigmas the second pass needs: the blocks' own first-pass rho, then sigma over it.
    first = AdaptiveScales(
        precision_deg=float(precision_deg),
        truncate=float(truncate),
        k_sigma=(SIGMA0,),
        k=(k0,),
        rho_max=float(rho_max),
        noise_model=noise_model,
        intensity_ref=ref,
    )
    # Which sigmas the second pass needs: per block one rho (where its median predicted variance at SIGMA0
    # reaches the precision), then the block's sigma map over that window. A range, so 4 blocks suffice.
    target = math.radians(precision_deg) ** 2
    seen = []
    for q, m in list(zip(pads, tissues))[:4]:
        h = _radius(SIGMA0, truncate) + _radius(max(rhos), truncate) + 2
        crop = (slice(H - h, H + b + h),) * 3
        inner0 = (slice(h, h + b),) * 3
        T = _tensors_fft(q[crop], SIGMA0, rhos, truncate)
        v = []
        for kk, r in zip(k0, rhos):
            lam = eigvals_sym3(T[r][(slice(None), *inner0)], np)
            v.append(float(np.median(np.minimum(kk * perturbation_factor(*lam, np)[m], V_RANDOM))))
        v = np.asarray(v)
        j = int(np.argmax(v <= target)) if (v <= target).any() else len(rhos) - 1
        rho_b = (
            rhos[j]
            if j == 0 or v[j] > target
            else float(
                np.exp(np.interp(np.log(target), np.log(v[[j, j - 1]]), np.log(np.asarray(rhos)[[j, j - 1]])))
            )
        )
        sig = sigma_map(q, np.full(q.shape, rho_b), first, _host_gauss, np)
        seen.append(sig[inner][m])
    seen = np.concatenate(seen)
    lo, hi = (float(v) for v in np.percentile(seen, [1, 99]))
    i_lo = max(0, int(np.searchsorted(sigs, lo, side="right")) - 1)
    i_hi = min(len(sigs) - 1, int(np.searchsorted(sigs, hi, side="left")))
    k_sigma, k = [SIGMA0], [k0]
    for s in sigs[i_lo : i_hi + 1]:
        if abs(s - SIGMA0) < 1e-9:
            continue
        k_sigma.append(float(s))
        k.append(_k_row(pads, tissues, draws, gs, s, rhos, truncate, H, b))
    order = np.argsort(k_sigma)
    noise = {
        "slope": float(model.slope),
        "slope_clipped": bool(model.slope_clipped),
        "anisotropy": float(model.anisotropy),
        "blocks": len(pads),
        "block": int(b),
        "sigma_p1_p99": [lo, hi],
    }
    out = AdaptiveScales(
        precision_deg=float(precision_deg),
        truncate=float(truncate),
        k_sigma=tuple(float(k_sigma[i]) for i in order),
        k=tuple(k[i] for i in order),
        rho_max=float(rho_max),
        noise=noise,
        noise_model=noise_model,
        intensity_ref=ref,
    )
    say(
        f"adaptive tensor: precision {precision_deg:g} deg, {noise_model} noise, rho {RHO_MIN:.2f}-{rho_max:g}; "
        f"noise factors at sigma {', '.join(f'{s:.2f}' for s in out.k_sigma)} from {len(pads)} blocks of {b}^3 "
        f"(noise slope {model.slope:.2f}{', clipped' if model.slope_clipped else ''}); halo {out.halo()} voxels"
    )
    return out


# ---------------------------------------------------------------------------
# Whole volumes
# ---------------------------------------------------------------------------


class _Threshold:
    """A lazy mask: ``volume > threshold``, or ``volume != 0`` when *threshold* is None."""

    def __init__(self, volume, threshold):
        self.volume, self.threshold, self.shape = volume, threshold, tuple(volume.shape)

    def __getitem__(self, key):
        v = np.asarray(self.volume[key])
        return v > self.threshold if self.threshold is not None else v != 0


def calibrate(
    volume,
    mask=None,
    *,
    threshold: float | None = None,
    precision_deg: float = PRECISION_DEG,
    rho_max: float = RHO_MAX,
    noise_model: str = "poisson",
    truncate: float = 3.0,
    n_blocks: int = 8,
    block: int = 32,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> AdaptiveScales:
    """Measure the noise factors of *volume* once: the :class:`AdaptiveScales` the tensor needs.

    *mask* (an array like *volume*, nonzero where there is tissue), or else
    ``volume > threshold``, or else ``volume != 0``, says where blocks are drawn.
    *volume* and *mask* may be lazy (zarr, dask, memmap): only the sampled
    blocks are read.
    """
    m = mask if mask is not None else _Threshold(volume, threshold)
    return calibrate_noise(
        volume,
        m,
        precision_deg=precision_deg,
        truncate=truncate,
        rho_max=rho_max,
        noise_model=noise_model,
        n_blocks=n_blocks,
        block=block,
        seed=seed,
        log=log,
    )


def _read_padded(volume, origin, size, halo):
    """The region ``[origin - halo, origin + size + halo)`` of *volume*, edge-replicated outside it, float32."""
    shape = np.asarray(volume.shape)
    lo, hi = np.asarray(origin) - halo, np.asarray(origin) + np.asarray(size) + halo
    clo, chi = np.maximum(lo, 0), np.minimum(hi, shape)
    a = np.asarray(volume[tuple(slice(int(x), int(y)) for x, y in zip(clo, chi))], np.float32)
    pad = [(int(c - lo_), int(hi_ - d)) for c, lo_, hi_, d in zip(clo, lo, hi, chi)]
    return np.pad(a, pad, mode="edge") if any(p != (0, 0) for p in pad) else a


def _chunks(shape, chunk):
    step = [int(chunk)] * 3 if chunk else list(shape)
    for z in range(0, shape[0], step[0]):
        for y in range(0, shape[1], step[1]):
            for x in range(0, shape[2], step[2]):
                o = (z, y, x)
                yield o, tuple(min(st, n - a) for st, n, a in zip(step, shape, o))


def adaptive_structure_tensor(
    volume,
    scales: AdaptiveScales | None = None,
    *,
    mask=None,
    threshold: float | None = None,
    chunk: int | None = None,
    gpu: bool = False,
    return_maps: bool = False,
    **calibrate_kwargs,
):
    """The structure tensor at each voxel's own ``(sigma, rho)``: ``(6, Z, Y, X)`` float32 on the host.

    *scales* comes from :func:`calibrate`; when None it is measured here first
    (*mask*, *threshold* and *calibrate_kwargs* go to :func:`calibrate`).
    *chunk* (voxels per side) bounds memory: each chunk is read with the
    method's halo, and the result does not depend on it. The volume's edges
    are handled by edge replication. *return_maps* also returns
    ``{"sigma": ..., "rho": ...}`` (voxels). *gpu* uses cupy.
    """
    from .tensor import backend

    if scales is None:
        scales = calibrate(volume, mask, threshold=threshold, **calibrate_kwargs)
    xp, gf = backend(gpu)
    shape = tuple(int(n) for n in volume.shape)
    H = scales.halo()
    S = np.empty((6,) + shape, np.float32)
    maps = {"sigma": np.empty(shape, np.float32), "rho": np.empty(shape, np.float32)} if return_maps else None
    for origin, size in _chunks(shape, chunk):
        pad = xp.asarray(_read_padded(volume, origin, size, H))
        got = {} if return_maps else None
        St = adaptive_tensor_block(pad, scales, gaussian_filter=gf, tensor_dtype=xp.float32, xp=xp, maps=got)
        core = tuple(slice(H, H + n) for n in size)
        dst = tuple(slice(a, a + n) for a, n in zip(origin, size))
        S[(slice(None), *dst)] = St[(slice(None), *core)].get() if gpu else St[(slice(None), *core)]
        if return_maps:
            for key in ("sigma", "rho"):
                m = got[key][core]
                maps[key][dst] = m.get() if gpu else m
        del pad, St
    return (S, maps) if return_maps else S


def scale_maps_volume(volume, scales: AdaptiveScales, *, chunk: int | None = None, gpu: bool = False) -> dict:
    """``{"sigma": ..., "rho": ...}`` (voxels) over *volume*, without the tensor."""
    from .tensor import backend

    xp, gf = backend(gpu)
    shape = tuple(int(n) for n in volume.shape)
    H = scales.halo()
    out = {"sigma": np.empty(shape, np.float32), "rho": np.empty(shape, np.float32)}
    for origin, size in _chunks(shape, chunk):
        pad = xp.asarray(_read_padded(volume, origin, size, H))
        sig, rho = scale_maps(pad, scales, gf, xp)
        core = tuple(slice(H, H + n) for n in size)
        dst = tuple(slice(a, a + n) for a, n in zip(origin, size))
        for key, m in (("sigma", sig[core]), ("rho", rho[core])):
            out[key][dst] = m.get() if gpu else m
    return out
