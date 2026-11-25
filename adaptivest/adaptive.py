"""Locally adaptive structure-tensor scales: each voxel its own ``sigma`` and ``rho``, from the image.

Every voxel gets its own integration scale ``rho`` and derivative scale ``sigma``,
from the image alone, in voxels. Nothing is calibrated per dataset and nothing is
sampled from the rest of the volume: the noise and the noise factors ``K`` are
measured in each voxel's own window (:mod:`adaptivest.local_noise`).

**The grids** are sampled in log scale at ratio sqrt(2) (two samples per octave,
the usual scale-space sampling): ``rho`` from :data:`RHO_MIN` = sqrt(2)
(Lindeberg's smallest integration factor over a unit derivative scale) to the
configured ``rho_max``; ``sigma`` from :data:`SIGMA0` = 0.5 to
``rho_max / sqrt(2)``. Both lie on the lattice ``2^(k/2)`` when ``rho_max`` does.

**rho(x): the smallest window that reaches a target angular precision.** At a
derivative scale ``s``, the tensor at each grid ``rho`` predicts the angular
variance of its smallest eigenvector as ``V = K(x; s, rho) * S``, with ``S =
sum_i (l_i + l_3) / (l_i - l_3)^2`` (first-order eigenvector perturbation:
large eigenvalue gaps, a well-determined direction). ``K(x; s, rho)`` is measured
in x's window: the mean squared angle by which one draw of the window's own
noise moves the direction, over the mean ``S``. ``V`` is smoothed over
:data:`V_SMOOTH` voxels; ``rho(x)`` is where it first falls to
``precision_deg^2``, interpolated in ``log rho`` (the grid's top where it never
does), and ``log rho`` is smoothed over :data:`RHO_SMOOTH` voxels.

**sigma(x): Lindeberg's scale over the voxel's own window.** Where the
normalised gradient energy ``s^2 |grad L_s|^2``, averaged with a Gaussian of
width ``rho(x)`` (windows at the grid ``rho`` blended in ``log rho``), first
stops rising over ``s``: its first peak, or its first shoulder (a positive local
minimum of the slope in ``log s``), whichever comes first; the flattest slope
where there is neither. A strong edge in a small window adds a response that
keeps rising with ``s``: it tilts the texture's peak into a shoulder, which is
still found, rather than sending sigma to the top of the grid.

**Three passes**, so the first ``rho`` is not skewed by an assumed ``sigma``:
``rho`` at :data:`SIGMA0` (the smallest derivative scale, nearly no
smoothing); ``sigma`` over that ``rho``; ``rho`` again at ``sigma(x)`` (``V``
blended between the grid sigmas). The tensor then blends the trace-normalised
tensors at the neighbouring grid ``(sigma, rho)`` bilinearly in
``(log sigma, log rho)``, and :func:`adaptivest.eigen` reads FA and the
direction from it as from a fixed one.

**Chunking** changes the result only statistically: the scale search and the
blend are local filters inside the halo, but the noise and ``K`` are window
statistics each tile measures from its own context and its own noise draw.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

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
#: Default width (voxels) of the window the noise and the noise factors are measured in.
NOISE_WINDOW = 128.0
#: z-planes per slab of the per-voxel eigen steps: bounds their float64 temporaries.
SLAB = 16

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
    """The per-run settings: the precision target, the Gaussian truncation, the grids' top and the noise window."""

    precision_deg: float = PRECISION_DEG
    truncate: float = 3.0
    rho_max: float = RHO_MAX
    #: Width (voxels) of the window the noise and ``K`` are measured in at each voxel.
    noise_window: float = NOISE_WINDOW
    #: Passes of the rho search: 2, rho at SIGMA0, sigma over it, rho again at sigma(x); 1, the first rho
    #: and the sigma over it (about 4 times faster; directions in tissue move by a median 0.5 degrees).
    passes: int = 2

    def __post_init__(self):
        if self.passes not in (1, 2):
            raise ValueError(f"passes must be 1 or 2, got {self.passes!r}")

    @property
    def rho_grid(self) -> tuple:
        return rho_grid(self.rho_max)

    @property
    def sigma_grid(self) -> tuple:
        return sigma_grid(self.rho_max)

    def halo(self) -> int:
        """Voxels of context a tile reads: the three passes and the blend, plus the noise window's pooling."""
        from .local_noise import pool_reach

        return chain_halo(self.rho_max, self.truncate, self.passes) + pool_reach(self.noise_window)

    def equivalent_scales(self) -> tuple[float, float]:
        """``(sigma, rho)`` whose fixed-scale halo (``st_halo_voxels``) equals :meth:`halo`, for the stage's planners."""
        sig = float(max(self.sigma_grid))
        return sig, (self.halo() - _radius(sig, self.truncate)) / float(self.truncate)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | AdaptiveScales | None) -> AdaptiveScales | None:
        if d is None or isinstance(d, AdaptiveScales):
            return d
        return cls(
            precision_deg=float(d.get("precision_deg", PRECISION_DEG)),
            truncate=float(d.get("truncate", 3.0)),
            rho_max=float(d.get("rho_max", RHO_MAX)),
            noise_window=float(d.get("noise_window") or NOISE_WINDOW),
            passes=int(d.get("passes", 2)),
        )


def chain_halo(rho_max: float, truncate: float, passes: int = 2) -> int:
    """Voxels of context the chain reads: rho (first pass), sigma over it, rho again at sigma(x) (when
    *passes* is 2), the blend."""
    t = float(truncate)
    r_rho = _radius(max(rho_grid(rho_max)), t)
    r_sig = _radius(max(sigma_grid(rho_max)), t)
    smooth = _radius(V_SMOOTH, t) + _radius(RHO_SMOOTH, t)
    first = _radius(SIGMA0, t) + r_rho + smooth  # rho at SIGMA0, at a voxel
    sig = max(first, r_sig + r_rho)  # sigma at a voxel: rho there, and the windowed energies
    if int(passes) == 1:
        return max(sig, r_sig + r_rho)  # the blend at a voxel reads its own scales only
    second = smooth + max(sig, r_sig + r_rho)  # rho again: V at neighbours, each at its own sigma
    return max(second, r_sig + r_rho)  # the blend at a voxel


# ---------------------------------------------------------------------------
# Array-module helpers (numpy or cupy)
# ---------------------------------------------------------------------------


def _free(xp):
    """Return cupy's cached blocks to the device between the large stages (no-op on numpy)."""
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()


def _xp_of(a):
    try:
        import cupy

        return cupy.get_array_module(a)
    except ImportError:  # pragma: no cover - CPU-only installs
        return np


_ORDERS = ([1, 0, 0], [0, 1, 0], [0, 0, 1])


def _gradients(vol, s, truncate, gaussian_filter, xp, support=None):
    """``(gz, gy, gx)`` at derivative scale *s*, float32.

    With *support* (0/1 where the image counts) the gradient is that of the
    normalised convolution ``G*(m I) / G*(m)`` (normalised differential
    convolution): only the supported intensities shape it, and the step where
    the support ends is never differentiated. Zero where the window holds no
    support.
    """
    v = xp.asarray(vol, dtype=xp.float32)
    if support is None:
        return tuple(gaussian_filter(v, s, order=o, mode="nearest", truncate=truncate) for o in _ORDERS)
    m = xp.asarray(support, dtype=xp.float32)
    mv = m * v
    c = gaussian_filter(m, s, mode="nearest", truncate=truncate)
    cv = gaussian_filter(mv, s, mode="nearest", truncate=truncate)
    ok = c > 1e-3
    inv2 = xp.where(ok, 1.0 / xp.where(ok, c, 1.0) ** 2, 0.0).astype(xp.float32)
    out = []
    for o in _ORDERS:
        dv = gaussian_filter(mv, s, order=o, mode="nearest", truncate=truncate)
        dc = gaussian_filter(m, s, order=o, mode="nearest", truncate=truncate)
        out.append(((dv * c - cv * dc) * inv2).astype(xp.float32))
    return tuple(out)


def _tensor(grads, r, truncate, gaussian_filter, xp, dtype, support=None):
    """(6, ...) tensor at window *r* from one set of gradients; with *support*, only supported voxels'
    outer products are averaged (not renormalised: fewer samples, a larger predicted variance)."""
    S = xp.empty((6,) + grads[0].shape, dtype=dtype)
    for i, (p, q) in enumerate(_PRODUCTS):
        prod = grads[p] * grads[q]
        if support is not None:
            prod = prod * support
        gaussian_filter(prod.astype(dtype, copy=False), r, mode="nearest", output=S[i], truncate=truncate)
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


def smallest_eigvec(S, l3, xp):
    """The eigenvector of *l3* (the smallest eigenvalue) of ``S`` (6, ...): the largest cross product of
    the rows of ``S - l3 I``, normalised; float32."""
    a0, a1, a2 = (S[i] - l3 for i in range(3))
    zy, zx, yx = S[3], S[4], S[5]
    r0, r1, r2 = (a0, zy, zx), (zy, a1, yx), (zx, yx, a2)

    def cross(u, w):
        return (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])

    cs = [cross(r0, r1), cross(r0, r2), cross(r1, r2)]
    nn = xp.stack([c[0] ** 2 + c[1] ** 2 + c[2] ** 2 for c in cs])
    best = xp.argmax(nn, axis=0)
    out = [xp.where(best == 0, cs[0][i], xp.where(best == 1, cs[1][i], cs[2][i])) for i in range(3)]
    norm = xp.sqrt(out[0] ** 2 + out[1] ** 2 + out[2] ** 2) + 1e-30
    return [(o / norm).astype(xp.float32) for o in out]


# ---------------------------------------------------------------------------
# The per-voxel eigen steps fused into one GPU kernel each
# ---------------------------------------------------------------------------
# The same arithmetic as eigvals_sym3, perturbation_factor and smallest_eigvec, in the same order and
# precision, without contracting to FMA: one pass per voxel instead of ~40 kernels and their float64
# temporaries. Used on cupy arrays; numpy runs the functions above.

_FUSED_PREAMBLE = r"""
__device__ __forceinline__ void eig3(float s0, float s1, float s2, float s3, float s4, float s5,
                                     double& l1, double& l2, double& l3) {
    double a11 = s0, a22 = s1, a33 = s2, a12 = s3, a13 = s4, a23 = s5;
    double p1 = a12 * a12 + a13 * a13 + a23 * a23;
    double q = (a11 + a22 + a33) / 3.0;
    double b11 = a11 - q, b22 = a22 - q, b33 = a33 - q;
    double p2 = b11 * b11 + b22 * b22 + b33 * b33 + 2.0 * p1;
    double p = sqrt(p2 / 6.0);
    bool safe = p > 1e-300;
    double inv = safe ? 1.0 / (safe ? p : 1.0) : 0.0;
    double c11 = b11 * inv, c22 = b22 * inv, c33 = b33 * inv, c12 = a12 * inv, c13 = a13 * inv, c23 = a23 * inv;
    double det = c11 * (c22 * c33 - c23 * c23) - c12 * (c12 * c33 - c23 * c13) + c13 * (c12 * c23 - c22 * c13);
    double r = det / 2.0;
    r = r < -1.0 ? -1.0 : (r > 1.0 ? 1.0 : r);
    double phi = acos(r) / 3.0;
    l1 = q + 2.0 * p * cos(phi);
    l3 = q + 2.0 * p * cos(phi + %(third)r);
    l2 = 3.0 * q - l1 - l3;
}
__device__ __forceinline__ double nmax(double a, double b) {  // numpy's maximum: NaN propagates
    return (a > b || isnan(a)) ? a : b;
}
__device__ __forceinline__ double pfactor(double l1, double l2, double l3) {
    double eps = 1e-12 * nmax(fabs(l1), 1e-300);
    double d1 = nmax(l1 - l3, eps), d2 = nmax(l2 - l3, eps);
    return (l1 + l3) / pow(d1, 2.0) + (l2 + l3) / pow(d2, 2.0);
}
__device__ __forceinline__ void evec3(float s0, float s1, float s2, float s3, float s4, float s5, float l3,
                                      float& o0, float& o1, float& o2) {
    float a0 = s0 - l3, a1 = s1 - l3, a2 = s2 - l3;
    // rows (a0, zy, zx), (zy, a1, yx), (zx, yx, a2); cross products 01, 02, 12
    float c[3][3] = {
        {s3 * s5 - s4 * a1, s4 * s3 - a0 * s5, a0 * a1 - s3 * s3},
        {s3 * a2 - s4 * s5, s4 * s4 - a0 * a2, a0 * s5 - s3 * s4},
        {a1 * a2 - s5 * s5, s5 * s4 - s3 * a2, s3 * s5 - a1 * s4}};
    int best = 0;
    float bn = powf(c[0][0], 2.0f) + powf(c[0][1], 2.0f) + powf(c[0][2], 2.0f);
    for (int j = 1; j < 3; ++j) {
        float n = powf(c[j][0], 2.0f) + powf(c[j][1], 2.0f) + powf(c[j][2], 2.0f);
        if (isnan(bn)) break;  // numpy's argmax: the first NaN wins
        if (n > bn || isnan(n)) { bn = n; best = j; }
    }
    float norm = sqrtf(powf(c[best][0], 2.0f) + powf(c[best][1], 2.0f) + powf(c[best][2], 2.0f)) + 1e-30f;
    o0 = c[best][0] / norm; o1 = c[best][1] / norm; o2 = c[best][2] / norm;
}
""" % {"third": 2.0 * math.pi / 3.0}

_S6 = "float32 s0, float32 s1, float32 s2, float32 s3, float32 s4, float32 s5"
_KERNELS = {}


def _kernel(name):
    """The fused cupy kernel *name* (compiled once)."""
    if name not in _KERNELS:
        import cupy

        body, params, outs = {
            # V = clip(K * S, 1e-8, V_RANDOM), V_RANDOM where not finite
            "variance": (
                "double l1, l2, l3; eig3(s0, s1, s2, s3, s4, s5, l1, l2, l3);"
                "double v = (double)kk * pfactor(l1, l2, l3);"
                "v = v < 1e-8 ? 1e-8 : (v > %(vr)r ? %(vr)r : v);"
                "out = isfinite(v) ? (float)v : (float)%(vr)r;" % {"vr": V_RANDOM},
                _S6 + ", T kk",
                "float32 out",
            ),
            # the smallest eigenvector and the perturbation factor
            "directions": (
                "double l1, l2, l3; eig3(s0, s1, s2, s3, s4, s5, l1, l2, l3);"
                "sf = (float)pfactor(l1, l2, l3);"
                "evec3(s0, s1, s2, s3, s4, s5, (float)l3, e0, e1, e2);",
                _S6,
                "float32 e0, float32 e1, float32 e2, float32 sf",
            ),
            # theta^2 between the smallest eigenvector and (f0, f1, f2)
            "angle2": (
                "double l1, l2, l3; eig3(s0, s1, s2, s3, s4, s5, l1, l2, l3);"
                "float e0, e1, e2; evec3(s0, s1, s2, s3, s4, s5, (float)l3, e0, e1, e2);"
                "float dot = fabsf(f0 * e0 + f1 * e1 + f2 * e2);"
                "dot = dot < 0.0f ? 0.0f : (dot > 1.0f ? 1.0f : dot);"
                "th2 = powf(acosf(dot), 2.0f);",
                _S6 + ", float32 f0, float32 f1, float32 f2",
                "float32 th2",
            ),
        }[name]
        _KERNELS[name] = cupy.ElementwiseKernel(
            params, outs, body, f"adaptivest_{name}", preamble=_FUSED_PREAMBLE, options=("--fmad=false",)
        )
    return _KERNELS[name]


def _fused(xp) -> bool:
    return xp is not np


_EW = {}


def _ew(name):
    """Small elementwise cupy kernels for the scale searches (compiled once, ``--fmad=false`` so every
    result is the bit pattern of the unfused numpy-style expression)."""
    if name not in _EW:
        import cupy

        spec = {
            # v += w * vk, in place (v is the output)
            "axpy": ("float32 w, float32 vk", "float32 v", "v = v + w * vk;"),
            # lv = log(max(v, 1e-8)); first rho: lrho = hit ? lr0 : nan
            "rho_first": (
                "float32 v, float32 tgt, float64 lr0",
                "float32 lv, float64 lrho",
                'lv = logf(fmaxf(v, 1e-8f)); lrho = (lv <= tgt) ? lr0 : nan("");',
            ),
            # later rhos: lv, and lrho where a voxel first reaches the target (interpolated between rhos)
            "rho_next": (
                "float32 v, float32 prev, float32 tgt, float32 lo, float32 dl",
                "float32 lv, float64 lrho",
                "lv = logf(fmaxf(v, 1e-8f));"
                "if ((lv <= tgt) && isnan(lrho)) {"
                "  float f = (prev - tgt) / fmaxf(prev - lv, 1e-12f);"
                "  f = f < 0.0f ? 0.0f : (f > 1.0f ? 1.0f : f);"
                "  lrho = (double)(lo + f * dl); }",
            ),
            # E = s^2 (gz^2 + gy^2 + gx^2)
            "energy": (
                "float32 gz, float32 gy, float32 gx, float32 ss",
                "float32 E",
                "E = ss * (gz * gz + gy * gy + gx * gx);",
            ),
            # acc += float32(w * f), w float64, f float32 promoted
            "acc": (
                "float64 w, float32 f",
                "float32 acc",
                "acc = acc + (float)(w * (double)f);",
            ),
            # L = log(max(acc, 1e-30)); d = (L - prev) / du
            "slope": (
                "float32 acc, float32 prev, float32 du",
                "float32 L, float32 d",
                "L = logf(fmaxf(acc, 1e-30f)); d = (L - prev) / du;",
            ),
        }[name]
        _EW[name] = cupy.ElementwiseKernel(
            spec[0], spec[1], spec[2], f"adaptivest_{name}", options=("--fmad=false",)
        )
    return _EW[name]


# ---------------------------------------------------------------------------
# The per-tile passes
# ---------------------------------------------------------------------------


def _variance_at(grads, k_index, kk, scales, gaussian_filter, xp, support=None):
    """The smoothed predicted variance at the *k_index*-th grid rho, from gradients at one derivative scale.

    *kk* is ``K`` at each voxel (or one number). The per-voxel eigen step runs in
    z-slabs of :data:`SLAB` planes (it needs no neighbours), so its float64
    temporaries stay small; the filters run on the whole block.
    """
    t = scales.truncate
    S = _tensor(grads, scales.rho_grid[k_index], t, gaussian_filter, xp, xp.float32, support)
    per_voxel = getattr(kk, "ndim", 0) > 0
    if _fused(xp):
        V = _kernel("variance")(*S, kk if per_voxel else float(kk))
        del S
        return gaussian_filter(V, V_SMOOTH, mode="nearest", truncate=t)
    V = xp.empty(S.shape[1:], dtype=xp.float32)
    for z in range(0, V.shape[0], SLAB):
        sl = slice(z, z + SLAB)
        l1, l2, l3 = eigvals_sym3(S[:, sl], xp)
        v = xp.clip((kk[sl] if per_voxel else kk) * perturbation_factor(l1, l2, l3, xp), 1e-8, V_RANDOM)
        V[sl] = xp.where(xp.isfinite(v), v, V_RANDOM)
        del l1, l2, l3, v
    del S
    return gaussian_filter(V, V_SMOOTH, mode="nearest", truncate=t)


def _inset(a, m):
    """*a* without *m* voxels at every face (the whole of it when *m* is 0 or too large)."""
    m = int(m)
    if m <= 0 or any(n <= 2 * m for n in a.shape):
        return a
    return a[(slice(m, -m),) * a.ndim]


def rho_map(
    block,
    scales: AdaptiveScales,
    gaussian_filter,
    xp,
    sigma=None,
    *,
    kmap,
    early_stop=True,
    focus=0,
    support=None,
):
    """rho(x) (voxels): at :data:`SIGMA0` when *sigma* is None, else with each voxel's derivative scale
    ``sigma(x)`` (V blended between the grid sigmas).

    *kmap* ``(s, k)`` gives ``K`` at derivative scale ``s`` and the ``k``-th grid
    rho over *block* (an array, or a number for every voxel). V is computed one
    grid rho at a time and the search stops once every voxel of *block* inside
    *focus* voxels of its faces has reached the precision (*early_stop*): the
    later rhos cannot change a voxel that has, so there the result is the same
    either way. Outside that inset the result is not used. *support* limits the
    gradients and their averages to the supported voxels.
    """
    t = scales.truncate
    if sigma is None:
        sources = [(None, SIGMA0, _gradients(block, SIGMA0, t, gaussian_filter, xp, support))]
    else:
        grid = scales.sigma_grid
        ws = _log_weights(xp.clip(sigma, min(grid), max(grid)), grid, xp)
        # One source's gradients at a time (recomputed at each rho): many sigmas would not fit at once.
        sources = [(w.astype(xp.float32), grid[i], None) for i, w in ws.items()]
        del ws
    target = 2.0 * math.log(math.radians(scales.precision_deg))
    lr = np.log(np.asarray(scales.rho_grid, float))
    lrho, prev = None, None
    fused = _fused(xp)
    for k in range(len(lr)):
        if sources[0][0] is None:
            _, s, grads = sources[0]
            v = _variance_at(grads, k, kmap(s, k), scales, gaussian_filter, xp, support)
        else:
            v = xp.zeros(block.shape, dtype=xp.float32)
            for w, s, _ in sources:
                grads = _gradients(block, s, t, gaussian_filter, xp, support)
                vk = _variance_at(grads, k, kmap(s, k), scales, gaussian_filter, xp, support)
                del grads
                if fused:
                    _ew("axpy")(w, vk, v)
                else:
                    v += w * vk
                del vk
        if fused:
            if lrho is None:
                lv, lrho = _ew("rho_first")(v, np.float32(target), np.float64(lr[0]))
            else:
                lv = xp.empty_like(v)
                _ew("rho_next")(
                    v,
                    prev,
                    np.float32(target),
                    np.float32(lr[k - 1]),
                    np.float32(float(lr[k] - lr[k - 1])),
                    lv,
                    lrho,
                )
            del v
        else:
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


def sigma_map(
    block, rho, scales: AdaptiveScales, gaussian_filter, xp, *, early_stop=True, focus=0, support=None
):
    """sigma(x) (voxels): where the Lindeberg response, averaged over a window of rho(x), first stops rising.

    ``R(s)`` is ``s^2 |grad L_s|^2`` averaged over the window; ``D`` its slope
    ``d log R / d log s`` between neighbouring grid scales (at their log midpoint).
    sigma is the first of: a peak of ``R`` (``D`` crosses zero; the crossing
    interpolated linearly), or a shoulder (``D`` has a local minimum that is still
    positive; a parabola through the three slopes). Where there is neither, the
    midpoint of the flattest slope. A peak is where a first-peak search finds it; a
    shoulder is a peak that a rising background (a strong edge in a small window)
    has tilted upwards, so sigma moves continuously from one to the other.

    Scales are taken in increasing order and the search stops once every voxel
    of *block* inside *focus* voxels of its faces has a peak or a shoulder
    (*early_stop*): each is read from the slopes up to it, so there the result
    is the same either way. Outside that inset it is not used.
    *support* limits the gradients and the windowed energies to the supported voxels.
    """
    t = scales.truncate
    grid = scales.sigma_grid
    if len(grid) < 2:
        return xp.full(rho.shape, grid[0], dtype=xp.float64)
    u = np.log(np.asarray(grid, float))
    mid = (u[:-1] + u[1:]) / 2
    wr = _log_weights(rho, scales.rho_grid, xp)
    D = []  # the last three slopes
    fused = _fused(xp)
    out = done = best = best_u = prev = None
    for n, s in enumerate(grid):
        gz, gy, gx = _gradients(block, s, t, gaussian_filter, xp, support)
        if fused:
            E = _ew("energy")(gz, gy, gx, np.float32(s * s))
        else:
            E = (s * s) * (gz * gz + gy * gy + gx * gx)
        if support is not None:
            E = E * support
        del gz, gy, gx
        acc = xp.zeros(E.shape, dtype=xp.float32)
        for k, w in wr.items():
            f = gaussian_filter(E, scales.rho_grid[k], mode="nearest", truncate=t)
            if fused:
                _ew("acc")(w, f, acc)
            else:
                acc += (w * f).astype(xp.float32)
            del f
        del E
        if fused and prev is not None:
            L, d = _ew("slope")(acc, prev, np.float32(u[n] - u[n - 1]))
        else:
            L = xp.log(xp.maximum(acc, xp.float32(1e-30)))
        del acc
        if prev is None:
            prev = L
            continue
        if not fused:
            d = ((L - prev) / (u[n] - u[n - 1])).astype(xp.float32)
        prev = L
        j = n - 1  # the midpoint index of d
        if out is None:
            out = xp.zeros(d.shape, dtype=xp.float64)
            done = xp.zeros(d.shape, dtype=bool)
            best, best_u = xp.abs(d), xp.full(d.shape, mid[0], dtype=xp.float64)
        else:
            flatter = xp.abs(d) < best
            best = xp.where(flatter, xp.abs(d), best)
            best_u = xp.where(flatter, mid[j], best_u)
        D = (D + [d])[-3:]
        if len(D) >= 2:
            # A peak of R between midpoints j-1 and j.
            a, b = D[-2], D[-1]
            peak = (a > 0) & (b <= 0) & ~done
            f = a.astype(xp.float64) / xp.where(peak, a - b, 1.0)
            out = xp.where(peak, mid[j - 1] + f * (mid[j] - mid[j - 1]), out)
            done |= peak
            del f, peak
        if len(D) == 3:
            # A shoulder: a positive local minimum of D at midpoint j-1.
            y0, y1, y2 = (x.astype(xp.float64) for x in D)
            sh = (y1 > 0) & (y1 < y0) & (y1 <= y2) & ~done
            den = y0 - 2.0 * y1 + y2
            off = xp.clip(0.5 * (y0 - y2) / xp.where(den > 0, den, 1.0), -0.5, 0.5)
            out = xp.where(sh, mid[j - 1] + off * (mid[j] - mid[j - 1]), out)
            done |= sh
            del y0, y1, y2, sh, den, off
        if early_stop and n + 1 < len(grid) and bool(_inset(done, focus).all()):
            break
    del D, prev
    out = xp.where(done, out, best_u)
    return xp.exp(xp.clip(out, u[0], u[-1]))


def scale_maps(
    block,
    scales: AdaptiveScales,
    gaussian_filter,
    xp,
    *,
    kmap,
    iterations: int = 1,
    trace=None,
    support=None,
):
    """``(sigma(x), rho(x))``: rho at SIGMA0, sigma over it, then *iterations* x (rho at sigma(x)[, sigma again]).

    Production runs one iteration and keeps the first pass's sigma. *trace*
    (a list) receives ``(sigma, rho)`` after every pass, recomputing sigma over
    each new rho, for studying convergence. *kmap* is as for :func:`rho_map`.
    """
    # What the core needs: rho and sigma of the first pass within the smoothings of it, the
    # second rho within its own smoothing. The early stops look only there.
    t = scales.truncate
    smooth = _radius(V_SMOOTH, t) + _radius(RHO_SMOOTH, t)
    halo = chain_halo(scales.rho_max, t)
    f1 = max(0, halo - smooth) if trace is None else 0
    f2 = max(0, halo - _radius(RHO_SMOOTH, t)) if trace is None else 0
    kw = dict(support=support, kmap=kmap)
    rho = rho_map(block, scales, gaussian_filter, xp, focus=f1, **kw)
    sig = sigma_map(block, rho, scales, gaussian_filter, xp, focus=f1, support=support)
    if trace is not None:
        trace.append((sig, rho))
    for it in range(int(iterations)):
        rho = rho_map(
            block, scales, gaussian_filter, xp, sigma=sig, focus=f2 if it + 1 == iterations else f1, **kw
        )
        if trace is not None or it + 1 < iterations:
            sig_next = sigma_map(block, rho, scales, gaussian_filter, xp, focus=f1, support=support)
            if trace is not None:
                trace.append((sig_next, rho))
            if it + 1 < iterations:
                sig = sig_next
    return sig, rho


def tile_scales(tile, scales: AdaptiveScales, gaussian_filter, xp, *, margin: int = 0, support=None):
    """The noise and the scale maps of a padded tile.

    Returns ``(block, support, margin, sigma, rho, noise)``: *tile* without the
    noise window's pooling reach (up to *margin*), its support, the margin left,
    and over that block sigma(x), rho(x) and the noise's share of the gradient
    power at :data:`SIGMA0`. The noise and ``K`` are measured over the whole tile.
    """
    from .local_noise import KMaps, LocalNoise, pool_reach

    tile = xp.asarray(tile, dtype=xp.float32)
    ln = LocalNoise(tile, scales.rho_max, scales.noise_window, gaussian_filter, xp)
    noisy = tile + ln.layer()
    share = ln.share()
    del ln
    P = min(int(margin), pool_reach(scales.noise_window))
    if any(n <= 2 * P for n in tile.shape):
        P = 0
    region = tuple(slice(P, n - P) for n in tile.shape)
    # What the core reads of K: the first rho within both smoothings of the second pass's sigma
    # (sigma at a voxel reads rho there), each within the smoothings of V and log rho.
    t = scales.truncate
    smooth = _radius(V_SMOOTH, t) + _radius(RHO_SMOOTH, t)
    m = int(margin) - P
    # One pass: the core reads K within one set of smoothings; two: within both passes' smoothings.
    layers = 1 if scales.passes == 1 else 2
    kreach = max(0, m - layers * smooth)
    near = (
        tuple(slice(P + kreach, n - P - kreach) for n in tile.shape) if m > layers * smooth else region
    )
    km = KMaps(tile, noisy, support, scales, gaussian_filter, xp, region=near)
    del noisy
    _free(xp)
    block, share = tile[region], share[region]
    sup = None if support is None else support[region]
    cache = {}

    def kmap(s, k):
        if (s, k) not in cache:
            cache.clear()  # one (s, k) at a time: rho_map asks in order
            cache[(s, k)] = km.at(s, k, region)
        return cache[(s, k)]

    # The passes of scale_maps, with K measured for the second pass's sigmas before it, so the tile and
    # its noisy copy are released while it runs. The second pass's rho is read only at the core, so only
    # the sigmas within its smoothings of the core are needed; elsewhere sigma is clamped to their range.
    if scales.passes == 1:
        # rho and sigma are read at the core only: rho's early stop looks within log rho's smoothing of it.
        km.prepare([SIGMA0])
        _free(xp)
        rho = rho_map(
            block, scales, gaussian_filter, xp, kmap=kmap, focus=max(0, m - _radius(RHO_SMOOTH, t)), support=sup
        )
        sig = sigma_map(block, rho, scales, gaussian_filter, xp, focus=max(0, m), support=sup)
        return block, sup, m, sig, rho, share
    halo = chain_halo(scales.rho_max, t)
    f1, f2 = max(0, halo - smooth), max(0, halo - _radius(RHO_SMOOTH, t))
    km.prepare([SIGMA0], release=False)
    km.park(True)
    _free(xp)
    rho = rho_map(block, scales, gaussian_filter, xp, kmap=kmap, focus=f1, support=sup)
    sig = sigma_map(block, rho, scales, gaussian_filter, xp, focus=f1, support=sup)
    grid = scales.sigma_grid
    used = _inset(sig, max(0, m - smooth)) if m > smooth else sig
    lo_s, hi_s = (
        float(xp.clip(used.min(), min(grid), max(grid))),
        float(xp.clip(used.max(), min(grid), max(grid))),
    )
    del used
    i_lo = max(0, int(np.searchsorted(grid, lo_s, side="right")) - 1)
    i_hi = min(len(grid) - 1, int(np.searchsorted(grid, hi_s, side="left")))
    sig2 = xp.clip(sig, grid[i_lo], grid[i_hi])
    need = [grid[i] for i in _log_weights(sig2, grid, xp)]
    _free(xp)
    km.prepare(need)
    _free(xp)
    rho = rho_map(block, scales, gaussian_filter, xp, sigma=sig2, kmap=kmap, focus=f2, support=sup)
    del sig2
    return block, sup, int(margin) - P, sig, rho, share


def _grow(inner, pad, shape):
    """*inner* (slices) grown by *pad* voxels on every side, clipped to *shape*."""
    return tuple(slice(max(0, s.start - pad), min(n, s.stop + pad)) for s, n in zip(inner, shape))


def _within(inner, outer):
    """*inner* in the coordinates of *outer* (both slices of the same array)."""
    return tuple(slice(i.start - o.start, i.stop - o.start) for i, o in zip(inner, outer))


def _at_least_f32(a, xp):
    return a.astype(xp.float32) if a.dtype.itemsize < 4 else a


def blend_halo(scales: AdaptiveScales) -> int:
    """Voxels of context :func:`tensor_from_scales` reads: the largest grid sigma's gradients over the
    largest grid rho's window (the scale search is not redone, so far less than :meth:`AdaptiveScales.halo`)."""
    t = scales.truncate
    return _radius(max(scales.sigma_grid), t) + _radius(max(scales.rho_grid), t)


def tensor_from_scales(
    block,
    sigma,
    rho,
    scales: AdaptiveScales,
    *,
    gaussian_filter,
    tensor_dtype,
    xp=None,
    margin: int = 0,
    support=None,
):
    """``S`` (6, *block.shape) at *tensor_dtype* from given scale maps: the blend step alone.

    *sigma* and *rho* (voxels) cover *block* without *margin* on every side;
    the tensor is computed there (zero in the margin), blending the
    trace-normalised tensors at the neighbouring grid ``(sigma, rho)``
    bilinearly in ``(log sigma, log rho)``. A NaN scale gives a zero tensor.
    *margin* of at least :func:`blend_halo` gives the same tensor as a whole
    volume would. *support* is as for :func:`adaptive_tensor_block`.
    """
    xp = _xp_of(block) if xp is None else xp
    t = scales.truncate
    shape = tuple(int(n) for n in block.shape)
    m = int(margin)
    inner = tuple(slice(m, n - m) for n in shape)
    if tuple(int(n) for n in sigma.shape) != tuple(s.stop - s.start for s in inner):
        raise ValueError(f"scale maps {tuple(sigma.shape)} do not cover the block {shape} without margin {m}")
    # Stored maps may be float16: weigh them at float32 at least.
    ws = _log_weights(_at_least_f32(xp.asarray(sigma), xp), scales.sigma_grid, xp)
    wr = _log_weights(_at_least_f32(xp.asarray(rho), xp), scales.rho_grid, xp)
    acc = xp.zeros((6,) + tuple(s.stop - s.start for s in inner), dtype=xp.float32)
    for i, wsi in ws.items():
        pairs = [(k, wrk) for k, wrk in wr.items() if bool((wsi * wrk).any())]
        if not pairs:
            continue
        # The gradients only where this sigma's tensors read them; each tensor only where inner reads it.
        reach = max(_radius(scales.rho_grid[k], t) for k, _ in pairs)
        crop = _grow(_grow(inner, reach, shape), _radius(scales.sigma_grid[i], t), shape)
        sup = None if support is None else support[crop]
        grads = _gradients(block[crop], scales.sigma_grid[i], t, gaussian_filter, xp, sup)
        for k, wrk in pairs:
            need = _grow(inner, _radius(scales.rho_grid[k], t), shape)
            S = _tensor(
                tuple(gr[_within(need, crop)] for gr in grads),
                scales.rho_grid[k],
                t,
                gaussian_filter,
                xp,
                xp.float32,
                None if support is None else support[need],
            )[(slice(None), *_within(inner, need))]
            tr = S[0] + S[1] + S[2]
            w = ((wsi * wrk) / xp.where(tr > 0, tr, 1.0) * (tr > 0)).astype(xp.float32)
            acc += w[None] * S
            del S, tr, w
        del grads
    out = xp.zeros((6,) + shape, dtype=tensor_dtype)
    out[(slice(None), *inner)] = acc
    return out


def adaptive_tensor_block(
    block,
    scales: AdaptiveScales,
    *,
    gaussian_filter,
    tensor_dtype,
    xp=None,
    maps=None,
    margin: int = 0,
    support=None,
):
    """``S`` (6, *block.shape) at *tensor_dtype*: the tensor at each voxel's own (sigma, rho), blended.

    *maps*, a dict, receives ``rho``, ``sigma`` and ``noise`` (the noise's share
    of the gradient power at :data:`SIGMA0`) when given. *margin* (voxels) is
    context only: the tensor is computed for the block without it (zero there),
    each (sigma, rho) pair only where that part uses it. *support* (0/1, the
    block's shape) restricts every gradient and every average to the supported
    voxels: the tensor sees only their intensities.
    """
    xp = _xp_of(block) if xp is None else xp
    full = tuple(int(n) for n in block.shape)
    block, support, m, sig, rho, share = tile_scales(
        block, scales, gaussian_filter, xp, margin=margin, support=support
    )
    off = (full[0] - block.shape[0]) // 2
    shape = tuple(int(n) for n in block.shape)
    if maps is not None:
        for key, a in (("rho", rho), ("sigma", sig), ("noise", share)):
            out = xp.full(full, xp.nan, dtype=xp.float32)
            out[tuple(slice(off, off + n) for n in shape)] = a
            maps[key] = out
    del share
    m = m if all(n > 2 * m for n in shape) else 0
    inner = tuple(slice(m, n - m) for n in shape)
    S = tensor_from_scales(
        block,
        sig[inner],
        rho[inner],
        scales,
        gaussian_filter=gaussian_filter,
        tensor_dtype=tensor_dtype,
        xp=xp,
        margin=m,
        support=support,
    )
    del rho, sig
    if shape == full:
        return S
    out = xp.zeros((6,) + full, dtype=tensor_dtype)
    out[(slice(None), *(slice(off, off + n) for n in shape))] = S
    return out


# ---------------------------------------------------------------------------
# Whole volumes
# ---------------------------------------------------------------------------


def _read_padded(volume, origin, size, halo):
    """The region ``[origin - halo, origin + size + halo)`` of *volume*, edge-replicated outside it, float32."""
    shape = np.asarray(volume.shape)
    lo, hi = np.asarray(origin) - halo, np.asarray(origin) + np.asarray(size) + halo
    clo, chi = np.maximum(lo, 0), np.minimum(hi, shape)
    a = np.asarray(volume[tuple(slice(int(x), int(y)) for x, y in zip(clo, chi))], np.float32)
    pad = [(int(c - lo_), int(hi_ - d)) for c, lo_, hi_, d in zip(clo, lo, hi, chi)]
    return np.pad(a, pad, mode="edge") if any(p != (0, 0) for p in pad) else a


#: Peak device bytes per voxel of a padded tile (measured: 76-93 at rho_max 16 on cupy, chunks 64-128), for "auto".
BYTES_PER_TILE_VOXEL = 100
#: Fraction of the free memory "auto" plans to use.
MEMORY_FRACTION = 0.9


def _free_bytes(xp) -> int:
    """Memory the next tile may use: free device memory plus cupy's cached pool (GPU), or available RAM."""
    if xp is np:
        import os

        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") * 0.5)
    free, _ = xp.cuda.runtime.memGetInfo()
    return int(free + xp.get_default_memory_pool().free_bytes())


def auto_chunk(shape, scales: AdaptiveScales, xp) -> int | None:
    """The largest chunk (a multiple of 32, at least 32) whose padded tile fits in free memory;
    None when the whole volume fits as one tile."""
    H = scales.halo()
    tile = int((MEMORY_FRACTION * _free_bytes(xp) / BYTES_PER_TILE_VOXEL) ** (1.0 / 3.0))
    if all(n + 2 * H <= tile for n in shape):
        return None
    return max(32, (tile - 2 * H) // 32 * 32)


def _chunks(shape, chunk):
    step = [int(chunk)] * 3 if chunk else list(shape)
    for z in range(0, shape[0], step[0]):
        for y in range(0, shape[1], step[1]):
            for x in range(0, shape[2], step[2]):
                o = (z, y, x)
                yield o, tuple(min(st, n - a) for st, n, a in zip(step, shape, o))


def _keep(mask, threshold, volume, dst):
    """Which voxels of the chunk *dst* are computed: *mask* nonzero, or ``volume > threshold``; None: all."""
    if mask is not None:
        return np.asarray(mask[dst]) != 0
    if threshold is not None:
        return np.asarray(volume[dst]) > threshold
    return None


def _support(mask, threshold, pad, origin, size, halo, xp):
    """The padded chunk's support (0/1 float32): *mask* nonzero or ``pad > threshold``; None: everything."""
    if mask is not None:
        return xp.asarray((_read_padded(mask, origin, size, halo) != 0).astype(np.float32))
    if threshold is not None:
        return (pad > threshold).astype(xp.float32)
    return None


def _over_chunks(volume, scales, chunk, xp, run):
    """Calls ``run(chunk)`` with *chunk* resolved; under "auto", again with smaller chunks on a device
    out-of-memory (the result does not depend on the chunk)."""
    shape = tuple(int(n) for n in volume.shape)
    auto = chunk == "auto"
    c = auto_chunk(shape, scales, xp) if auto else chunk
    while True:
        try:
            return run(c)
        except Exception as err:  # noqa: BLE001 - only a cupy out-of-memory is retried
            if not (auto and xp is not np and type(err).__name__ == "OutOfMemoryError"):
                raise
            xp.get_default_memory_pool().free_all_blocks()
            c = max(32, int((c or max(shape)) * 0.75) // 32 * 32) if (c or max(shape)) > 32 else None
            if c is None:
                raise


def adaptive_structure_tensor(
    volume,
    scales: AdaptiveScales | None = None,
    *,
    mask=None,
    threshold: float | None = None,
    chunk: int | str | None = "auto",
    gpu: bool | None = None,
    return_maps: bool = False,
):
    """The structure tensor at each voxel's own ``(sigma, rho)``: ``(6, Z, Y, X)`` float32 on the host.

    *scales* holds the settings (:class:`AdaptiveScales`; its defaults when None).
    *mask* (nonzero to keep), or else ``volume > threshold``, isolates a
    structure: its gradients come only from the kept intensities (normalised
    differential convolution, so the mask's edge is never differentiated), its
    tensor and scale searches average only kept voxels, and elsewhere the tensor
    is zero (direction undefined in :func:`adaptivest.eigen`) with NaN maps.
    Chunks with nothing to keep are skipped. The noise spectrum ignores the mask.
    *chunk* (voxels per side) bounds memory: each chunk is read with the
    method's halo. ``"auto"`` takes the largest that fits the free memory; None
    computes the volume in one piece. The volume's edges are handled by edge
    replication. *return_maps* also returns ``{"sigma", "rho", "noise"}``: the
    scales in voxels, and the noise's share of the gradient power at sigma 0.5.
    Runs on the GPU (cupy) when there is one; *gpu* False forces the CPU.
    """
    from .tensor import backend

    scales = AdaptiveScales() if scales is None else scales
    xp, gf = backend(gpu)
    gpu = xp is not np
    shape = tuple(int(n) for n in volume.shape)
    H = scales.halo()
    keys = ("sigma", "rho", "noise")

    def run(chunk):
        S = np.zeros((6,) + shape, np.float32)
        maps = {k: np.full(shape, np.nan, np.float32) for k in keys} if return_maps else None
        for origin, size in _chunks(shape, chunk):
            dst = tuple(slice(a, a + n) for a, n in zip(origin, size))
            keep = _keep(mask, threshold, volume, dst)
            if keep is not None and not keep.any():
                continue
            pad = xp.asarray(_read_padded(volume, origin, size, H))
            got = {} if return_maps else None
            St = adaptive_tensor_block(
                pad,
                scales,
                gaussian_filter=gf,
                tensor_dtype=xp.float32,
                xp=xp,
                maps=got,
                margin=H,
                support=_support(mask, threshold, pad, origin, size, H, xp),
            )
            core = tuple(slice(H, H + n) for n in size)
            Sc = St[(slice(None), *core)]
            Sc = Sc.get() if gpu else np.asarray(Sc)
            S[(slice(None), *dst)] = Sc if keep is None else Sc * keep
            if return_maps:
                for key in keys:
                    m = got[key][core]
                    m = m.get() if gpu else np.asarray(m)
                    maps[key][dst] = m if keep is None else np.where(keep, m, np.nan)
            del pad, St, Sc, got
            if gpu:
                xp.get_default_memory_pool().free_all_blocks()
        return (S, maps) if return_maps else S

    return _over_chunks(volume, scales, chunk, xp, run)


def scale_maps_chunk(
    volume,
    origin,
    size,
    scales: AdaptiveScales | None = None,
    *,
    support_mask=None,
    threshold: float | None = None,
    gpu: bool | None = None,
    backend=None,
) -> dict:
    """``{"sigma", "rho", "noise"}`` (host float32, shape *size*) for the chunk of *volume* at *origin*.

    The chunk is read with :meth:`AdaptiveScales.halo` of context (edge
    replicated outside the volume), so any chunking of a volume gives the same
    maps up to each chunk's own noise draw. *support_mask* / *threshold* are
    *mask* / *threshold* of :func:`adaptive_structure_tensor` (the gradients
    and averages see only the kept voxels); the maps are not masked here.
    *backend*, ``(xp, gaussian_filter)``, overrides *gpu*.
    """
    from .tensor import backend as _backend

    scales = AdaptiveScales() if scales is None else scales
    xp, gf = _backend(gpu) if backend is None else backend
    H = scales.halo()
    origin = tuple(int(o) for o in origin)
    size = tuple(int(n) for n in size)
    pad = xp.asarray(_read_padded(volume, origin, size, H))
    _, _, m, sig, rho, share = tile_scales(
        pad, scales, gf, xp, margin=H, support=_support(support_mask, threshold, pad, origin, size, H, xp)
    )
    core = tuple(slice(m, m + n) for n in size)
    out = {}
    for key, a in (("sigma", sig[core]), ("rho", rho[core]), ("noise", share[core])):
        a = a.astype(xp.float32)
        out[key] = a.get() if xp is not np else np.asarray(a)
    del pad, sig, rho, share
    _free(xp)
    return out


def scale_maps_volume(
    volume,
    scales: AdaptiveScales | None = None,
    *,
    mask=None,
    threshold: float | None = None,
    chunk: int | str | None = "auto",
    gpu: bool | None = None,
) -> dict:
    """``{"sigma", "rho", "noise"}`` over *volume*, without the tensor (see :func:`adaptive_structure_tensor`)."""
    from .tensor import backend

    scales = AdaptiveScales() if scales is None else scales
    xp, gf = backend(gpu)
    shape = tuple(int(n) for n in volume.shape)

    def run(chunk):
        out = {k: np.full(shape, np.nan, np.float32) for k in ("sigma", "rho", "noise")}
        for origin, size in _chunks(shape, chunk):
            dst = tuple(slice(a, a + n) for a, n in zip(origin, size))
            keep = _keep(mask, threshold, volume, dst)
            if keep is not None and not keep.any():
                continue
            got = scale_maps_chunk(
                volume, origin, size, scales, support_mask=mask, threshold=threshold, backend=(xp, gf)
            )
            for key, a in got.items():
                out[key][dst] = a if keep is None else np.where(keep, a, np.nan)
        return out

    return _over_chunks(volume, scales, chunk, xp, run)
