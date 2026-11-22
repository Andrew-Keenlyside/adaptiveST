"""The image's noise at each voxel, from that voxel's own window, and the noise factors ``K`` it implies.

Nothing here samples the rest of the volume or uses a noise model: a voxel's values come from
its window and, through the window's averaging, the windows around it (at most a window's width away).

**The noise spectrum at x.** The window around x (``noise_window`` voxels, 128 by
default) is split into frequency bands and tiled by 4³ sub-windows (a quarter of
its width: 32³). In each band, the noise power is the smallest band power among
the sub-windows, corrected for the bias of a minimum over noisy estimates (Martin
2001: the expected minimum of N Gamma variables). Signal and noise are
uncorrelated, so a band's power can never be less than its noise; where tissue
texture is patchy, the quietest sub-window holds little else. The log of that
minimum is then averaged over the window (a Gaussian, s.d. a sixth of it, cut at
3 s.d.), so the maps are smooth, and capped at the window's mean band power.

**The level correction.** Real noise is not equally strong everywhere in a window,
and a plain minimum then picks a quieter spot and reads low. So each sub-window's
band power is first divided by its noise level: its power above 0.5 cycles/voxel
(beyond the Nyquist frequency of every axis, where an image sampled at or above
its resolution holds noise only), relative to the window's. The minimum is taken
over these and scaled back by the window's mean level. This assumes the level
varies alike at all frequencies, and that the image has noise above 0.5
cycles/voxel (an upsampled or low-pass filtered image does not).

**Bands.** ``B/2`` bands of equal coefficient count, ``B = ceil(2 pi rho_max)``
(the block that resolves the largest window's frequencies), from the window's
spectral resolution (3 s.d. of the Gaussian window's transform) to the cube's
corner. Equal counts give each band the same statistical precision.

**The noise layer** is one draw of Gaussian noise with that spectrum: white noise
split into the same bands, each band scaled at each voxel to its local noise
power.

**K(x).** The tensor is computed on the image and on the image plus the layer.
``K`` at a voxel is the window's sum of squared angles between the two
directions over its sum of ``S`` (the perturbation factor), over the voxels in
the linear range (``K S`` and the angle below half a random axis's). Larger
windows use tensors on a 2× or 4× lattice.

Approximations: each tile computes these window statistics from its own context,
so near a tile's faces the windows are truncated and the noise layer is a
different draw; the results agree between chunkings only statistically, not to
the bit.
"""

from __future__ import annotations

import math

import numpy as np

from .adaptive import (
    _PRODUCTS,
    SIGMA0,
    SLAB,
    V_RANDOM,
    _fused,
    _gradients,
    _kernel,
    _radius,
    eigvals_sym3,
    perturbation_factor,
    smallest_eigvec,
)

#: Frequencies (cycles/voxel) above this are the sub-windows' noise-level reference: beyond the Nyquist
#: frequency of every axis, in the cube's corners only.
LEVEL_BAND = 0.5

#: The noise draw's seed: fixed, so a run is reproducible on the CPU and the GPU alike.
SEED = 0


def cell(noise_window: float) -> int:
    """The coarse cell (voxels) of the band-power and K maps: a sixteenth of the window, at least 1."""
    return max(1, int(round(float(noise_window) / 16.0)))


def subwindow(noise_window: float) -> int:
    """The minimum statistics' sub-window (voxels): a quarter of the window, a whole number of cells."""
    return 4 * cell(noise_window)


def pool_reach(noise_window: float) -> int:
    """Voxels a window average reads beyond its cell: the Gaussian's radius plus one cell."""
    return _radius(float(noise_window) / 6.0, 3.0) + cell(noise_window)


def band_edges(rho_max: float, noise_window: float) -> np.ndarray:
    """``B/2 + 1`` edges (cycles/voxel) of the bands of equal coefficient count, independent of any tile."""
    B = int(math.ceil(2.0 * math.pi * float(rho_max)))
    nb = max(2, B // 2)
    f = np.fft.fftfreq(B)
    kk = np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2).ravel()
    kres = 3.0 / (2.0 * math.pi * (float(noise_window) / 6.0))
    kk = np.sort(kk[kk >= kres])
    inner = kk[np.minimum((np.arange(1, nb) * len(kk)) // nb, len(kk) - 1)]
    return np.concatenate([[kres], inner, [math.sqrt(3.0) / 2.0 + 1e-6]])


def _ndi(xp):
    if xp is np:
        import scipy.ndimage as ndi
    else:
        import cupyx.scipy.ndimage as ndi
    return ndi


def _bsum(x, c):
    if c == 1:
        return x
    z, y, w = (s // c for s in x.shape)
    return x.reshape(z, c, y, c, w, c).sum(axis=(1, 3, 5))


def _expected_min(n: int, m: float) -> float:
    """E[min of *n* Gamma(*m*, 1/*m*)] (mean 1)."""
    from scipy.integrate import quad
    from scipy.special import gammaincc

    m = max(float(m), 0.5)
    return float(quad(lambda x: gammaincc(m, m * x) ** n, 0, np.inf, limit=200)[0])


def _tile_min(a, q: int, n: int, xp):
    """At each cell, the minimum of *a* over the ``n³`` positions ``q`` cells apart that tile the window
    around it (edges clamped)."""
    offs = [int(round((i - (n - 1) / 2.0) * q)) for i in range(n)]
    pad = max(abs(o) for o in offs)
    for ax in range(3):
        if pad == 0:
            break
        w = [(0, 0)] * 3
        w[ax] = (pad, pad)
        p = xp.pad(a, w, mode="edge")
        m = None
        for o in offs:
            sl = [slice(None)] * 3
            sl[ax] = slice(pad + o, pad + o + a.shape[ax])
            m = p[tuple(sl)] if m is None else xp.minimum(m, p[tuple(sl)])
        a = m
    return a


def _interp_matrix(n_out: int, n_cells: int, c: int, xp, lo: int = 0):
    """Linear interpolation from cell centres (``c i + (c - 1)/2``) to voxels ``lo..lo+n_out``, clamped."""
    pos = (np.arange(lo, lo + n_out) - (c - 1) / 2.0) / c
    i0 = np.clip(np.floor(pos).astype(int), 0, max(n_cells - 2, 0))
    fr = np.clip(pos - i0, 0.0, 1.0) if n_cells > 1 else np.zeros(n_out)
    A = np.zeros((n_out, n_cells), np.float32)
    A[np.arange(n_out), i0] += 1.0 - fr
    if n_cells > 1:
        A[np.arange(n_out), i0 + 1] += fr
    return xp.asarray(A)


class Upsampler:
    """Coarse cell maps to the voxels of a region (slices; the whole grid when a shape), by separable
    linear interpolation."""

    def __init__(self, region, cells, c, xp):
        self.xp = xp
        region = tuple(r if isinstance(r, slice) else slice(0, int(r)) for r in region)
        self.A = [_interp_matrix(r.stop - r.start, int(m), c, xp, r.start) for r, m in zip(region, cells)]

    def __call__(self, a):
        xp = self.xp
        a = xp.asarray(a, dtype=xp.float32)
        a = xp.tensordot(self.A[0], a, axes=(1, 0))
        a = xp.tensordot(self.A[1], a, axes=(1, 1)).transpose(1, 0, 2)
        return xp.tensordot(a, self.A[2], axes=(2, 1))


def _taper(n: int, w: int, xp):
    t = np.ones(n, np.float32)
    w = int(min(w, n // 2))
    if w > 0:
        r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(w) + 0.5) / w)
        t[:w] = r
        t[-w:] = r[::-1]
    return xp.asarray(t)


class LocalNoise:
    """The noise of a tile, measured voxel by voxel from its window, and the tile plus one draw of it.

    The tile is padded (reflected) to whole cells; every map covers the padded tile.
    """

    def __init__(self, tile, rho_max, noise_window, gaussian_filter, xp):
        ndi = _ndi(xp)
        W = float(noise_window)
        c, L = cell(W), subwindow(W)
        self.xp, self.c, self.shape = xp, c, tuple(int(n) for n in tile.shape)
        pshape = tuple(-(-n // c) * c for n in self.shape)
        img = xp.asarray(tile, dtype=xp.float32)
        if pshape != self.shape:
            img = xp.pad(img, [(0, p - n) for p, n in zip(pshape, self.shape)], mode="reflect")
        self.pshape, N = pshape, float(np.prod(pshape))
        edges = band_edges(rho_max, W)
        nb = len(edges) - 1
        self.kmid = 0.5 * (edges[:-1] + edges[1:])
        tp = _taper(pshape[0], L, xp)[:, None, None] * _taper(pshape[1], L, xp)[None, :, None]
        tp = tp * _taper(pshape[2], L, xp)[None, None, :]
        F = xp.fft.rfftn((img - img.mean()) * tp)
        del tp
        fz = xp.fft.fftfreq(pshape[0]).astype(xp.float32)[:, None, None]
        fy = xp.fft.fftfreq(pshape[1]).astype(xp.float32)[None, :, None]
        fx = xp.fft.rfftfreq(pshape[2]).astype(xp.float32)[None, None, :]
        kmag = xp.sqrt(fz * fz + fy * fy + fx * fx)
        shell = (xp.searchsorted(xp.asarray(edges, dtype=xp.float32), kmag, side="right") - 1).astype(
            xp.int32
        )
        shell = xp.where(kmag < float(edges[0]), -1, xp.minimum(shell, nb - 1))
        del kmag
        herm = xp.full(fx.shape, 2.0, dtype=xp.float32)
        herm[..., 0] = 1.0
        if pshape[2] % 2 == 0:
            herm[..., -1] = 1.0
        counts = xp.bincount(
            (shell + 1).ravel(), weights=xp.broadcast_to(herm, shell.shape).ravel(), minlength=nb + 1
        )
        self.frac = np.asarray(counts[1:].get() if xp is not np else counts[1:], float) / N
        cells = tuple(n // c for n in pshape)
        q, n_side = L // c, int(round(W / L))
        sw = W / 6.0 / c
        gw = lambda a: gaussian_filter(a, sw, mode="nearest", truncate=3.0)  # noqa: E731
        box = lambda a, n: ndi.uniform_filter(a, n, mode="nearest") if n > 1 else a  # noqa: E731
        E = []
        for b in range(nb):
            band = xp.fft.irfftn(F * (shell == b), s=pshape, axes=(0, 1, 2)).astype(xp.float32)
            E.append(_bsum(band * band, c) / float(c**3))
            del band
        del F
        # each sub-window's noise level: its power above the axes' Nyquist frequency, relative to the window's
        ref = sum(E[b] for b in ([b for b in range(nb) if edges[b] >= LEVEL_BAND] or [nb - 1]))
        refsub = box(ref, q)
        refsub = xp.maximum(refsub, 1e-12 * float(xp.maximum(refsub.max(), 1e-30)))
        refwin = box(ref, q * n_side)
        del ref
        Pn, Pt = [], []
        for b in range(nb):
            e = E[b]
            mn = _tile_min(box(e, q) / refsub, q, n_side, xp) * refwin
            mn = mn / _expected_min(n_side**3, L**3 * self.frac[b] / 2.0)
            tot = gw(e)
            floor = 1e-12 * float(xp.maximum(tot.max(), 1e-30))
            Pn.append(xp.minimum(xp.exp(gw(xp.log(xp.maximum(mn, floor)))), tot))
            Pt.append(tot)
            del e, mn
        del E
        self.Pn, self.Pt, self.cells, self.shell, self.nb = xp.stack(Pn), xp.stack(Pt), cells, shell, nb
        self.up = Upsampler(pshape, cells, c, xp)

    def layer(self):
        """One draw of the tile's local noise (the tile's shape)."""
        xp = self.xp
        white = xp.asarray(np.random.default_rng(SEED).standard_normal(self.pshape, dtype=np.float32))
        Fw = xp.fft.rfftn(white)
        del white
        layer = xp.zeros(self.pshape, dtype=xp.float32)
        for b in range(self.nb):
            fld = xp.fft.irfftn(Fw * (self.shell == b), s=self.pshape, axes=(0, 1, 2)).astype(xp.float32)
            layer += fld * self.up(xp.sqrt(self.Pn[b] / max(self.frac[b], 1e-30)))
            del fld
        del Fw
        return layer[tuple(slice(0, n) for n in self.shape)]

    def share(self):
        """The noise's share of the gradient power at :data:`SIGMA0`, at each voxel of the tile."""
        xp = self.xp
        k = self.kmid
        w = xp.asarray(((2 * np.pi * k) ** 2 * np.exp(-((2 * np.pi * k * SIGMA0) ** 2))).astype(np.float32))
        w = w[:, None, None, None]
        s = (w * self.Pn).sum(0) / xp.maximum((w * self.Pt).sum(0), 1e-30)
        return self.up(s)[tuple(slice(0, n) for n in self.shape)]


def _lattice_tensor(grads, r, d, truncate, gaussian_filter, xp, crop=None):
    """(6, ...) tensor at window *r* on every *d*-th voxel: a pre-smoothing at d/2, subsample, the rest;
    only the lattice points in *crop* (slices of the lattice) are kept."""
    crop = (slice(None),) * 3 if crop is None else crop
    out = None
    for i, (p, q) in enumerate(_PRODUCTS):
        prod = grads[p] * grads[q]
        if d == 1:
            sm = gaussian_filter(prod, r, mode="nearest", truncate=truncate)
        else:
            b = d / 2.0
            pre = gaussian_filter(prod, b, mode="nearest", truncate=truncate)[::d, ::d, ::d]
            sm = gaussian_filter(pre, math.sqrt(r * r - b * b) / d, mode="nearest", truncate=truncate)
            del pre
        del prod
        sm = sm[crop]
        if out is None:
            out = xp.empty((6,) + sm.shape, dtype=xp.float32)
        out[i] = sm
        del sm
    return out


def _directions(S, xp, want_s=False):
    """Smallest eigenvectors (3 arrays) of *S* (6, ...), and the perturbation factor when *want_s*."""
    if _fused(xp):
        *e, Sf = _kernel("directions")(*S)
        return e, (Sf if want_s else None)
    e = [xp.empty(S.shape[1:], dtype=xp.float32) for _ in range(3)]
    Sf = xp.empty(S.shape[1:], dtype=xp.float32) if want_s else None
    for z in range(0, S.shape[1], SLAB):
        sl = slice(z, z + SLAB)
        l1, l2, l3 = eigvals_sym3(S[:, sl], xp)
        if want_s:
            Sf[sl] = perturbation_factor(l1, l2, l3, xp)
        for a, v in zip(e, smallest_eigvec(S[:, sl], l3.astype(xp.float32), xp)):
            a[sl] = v
        del l1, l2, l3
    return e, Sf


def _lattice_step(r, c):
    """The lattice step K's tensor at window *r* is computed on: 1, 2 or 4 voxels, dividing the cell *c*."""
    d = 1 if r < 2.8 else (2 if r < 8 else 4)
    d = min(d, c)
    while c % d:
        d //= 2
    return d


def _presmoothed(grads, d, truncate, gaussian_filter, xp):
    """The 6 gradient products pre-smoothed at d/2 and subsampled every *d*-th voxel (the first step of
    :func:`_lattice_tensor`, shared by every rho on that lattice)."""
    out = []
    for p, q in _PRODUCTS:
        prod = grads[p] * grads[q]
        out.append(xp.ascontiguousarray(gaussian_filter(prod, d / 2.0, mode="nearest", truncate=truncate)[::d, ::d, ::d]))
        del prod
    return out


def _lattice_finish(pre, r, d, truncate, gaussian_filter, xp, crop):
    """(6, ...) tensor at window *r* from :func:`_presmoothed` products: the rest of :func:`_lattice_tensor`."""
    b = d / 2.0
    out = None
    for i, a in enumerate(pre):
        sm = gaussian_filter(a, math.sqrt(r * r - b * b) / d, mode="nearest", truncate=truncate)[crop]
        if out is None:
            out = xp.empty((6,) + sm.shape, dtype=xp.float32)
        out[i] = sm
        del sm
    return out


def _angles2(S1, e0, xp):
    """theta^2 between the smallest eigenvectors of *S1* (6, ...) and *e0* (3 arrays), float32."""
    if _fused(xp):
        return _kernel("angle2")(*S1, *e0)
    th2 = e0[0]  # the angle, slab by slab, over the clean direction's storage
    for z in range(0, S1.shape[1], SLAB):
        sl = slice(z, z + SLAB)
        e1 = smallest_eigvec(S1[:, sl], eigvals_sym3(S1[:, sl], xp)[2].astype(xp.float32), xp)
        dot = xp.abs(e0[0][sl] * e1[0] + e0[1][sl] * e1[1] + e0[2][sl] * e1[2])
        th2[sl] = xp.arccos(xp.clip(dot, 0.0, 1.0)) ** 2
        del e1, dot
    return th2


class KMaps:
    """``K(x)`` at each derivative scale and grid rho, from the tile and the tile plus its noise draw.

    Measured only where it is read: over *region* (slices of the tile; all of it
    when None), pooled from the angles within the window's reach of it. Computed
    when first asked for and kept as coarse cells; :meth:`at` returns the voxels
    of a region of the tile (beyond the measured cells, their edge values).
    """

    def __init__(self, tile, noisy, support, scales, gaussian_filter, xp, region=None):
        W = float(scales.noise_window)
        self.c = c = cell(W)
        self.xp, self.gf, self.scales = xp, gaussian_filter, scales
        self.shape = tuple(int(n) for n in tile.shape)
        self.pshape = tuple(-(-n // c) * c for n in self.shape)
        pad = [(0, p - n) for p, n in zip(self.pshape, self.shape)]
        grow = (lambda a: xp.pad(a, pad, mode="reflect")) if any(p for _, p in pad) else (lambda a: a)
        self.img, self.noisy = (
            grow(xp.asarray(tile, dtype=xp.float32)),
            grow(xp.asarray(noisy, dtype=xp.float32)),
        )
        valid = (xp.asarray(tile) != 0) if support is None else (xp.asarray(support) > 0)
        self.valid = grow(valid.astype(xp.float32)) > 0
        self.support = None if support is None else grow(xp.asarray(support, dtype=xp.float32))
        region = tuple(slice(0, n) for n in self.shape) if region is None else region
        R = pool_reach(W)
        self.lo = tuple(max(0, (r.start - R) // c * c) for r in region)
        self.hi = tuple(min(p, -(-(r.stop + R) // c) * c) for r, p in zip(region, self.pshape))
        self.cells = tuple((h - a) // c for a, h in zip(self.lo, self.hi))
        self.sw = W / 6.0 / c
        self._k, self._up = {}, {}

    def _row(self, s):
        """K at derivative scale *s* for every grid rho (coarse cells).

        The gradients of the tile and of its noisy copy are computed once for all rhos; the rhos that
        share a lattice step share its pre-smoothed, subsampled products (:func:`_presmoothed`).
        """
        xp, gf, t, c = self.xp, self.gf, self.scales.truncate, self.c
        pool = lambda a: gf(a, self.sw, mode="nearest", truncate=3.0)  # noqa: E731
        grads = [_gradients(a, s, t, gf, xp, self.support) for a in (self.img, self.noisy)]
        steps = [_lattice_step(r, c) for r in self.scales.rho_grid]
        pre = {}
        row = []
        for i, (r, d) in enumerate(zip(self.scales.rho_grid, steps)):
            crop = tuple(slice(a // d, h // d) for a, h in zip(self.lo, self.hi))
            if d > 1 and d not in pre:
                pre = {d: [_presmoothed(g, d, t, gf, xp) for g in grads]}
            if d > 1 and all(e == d for e in steps[i + 1 :]):
                grads = None  # the later rhos need only these pre-smoothed products
            if d == 1:
                S0 = _lattice_tensor(grads[0], r, d, t, gf, xp, crop)
            else:
                S0 = _lattice_finish(pre[d][0], r, d, t, gf, xp, crop)
            e0, Sf = _directions(S0, xp, want_s=True)
            del S0
            if d == 1:
                S1 = _lattice_tensor(grads[1], r, d, t, gf, xp, crop)
            else:
                S1 = _lattice_finish(pre[d][1], r, d, t, gf, xp, crop)
            th2 = _angles2(S1, e0, xp)
            del S1, e0
            m = self.valid[::d, ::d, ::d][crop] & xp.isfinite(Sf) & (Sf > 0)
            Sf = xp.where(m, Sf, 0.0).astype(xp.float32)
            th2 = xp.where(m, th2, 0.0).astype(xp.float32)
            cc = c // d
            m1 = m & (th2 < V_RANDOM / 2)
            K0 = pool(_bsum(th2 * m1, cc)) / xp.maximum(pool(_bsum(Sf * m1, cc)), 1e-30)
            del m1
            K0v = K0
            for ax in range(3):
                K0v = xp.repeat(K0v, cc, axis=ax)
            m2 = m & (Sf * K0v < V_RANDOM / 2)
            del K0v
            row.append(
                (pool(_bsum(th2 * m2, cc)) / xp.maximum(pool(_bsum(Sf * m2, cc)), 1e-30)).astype(xp.float32)
            )
            del th2, Sf, m, m2, K0
        del grads, pre
        return row

    def park(self, on_host: bool):
        """Move the tile, its noisy copy and the masks to host memory (*on_host*) or back to the device,
        so the other passes have the device while K is not being measured. No-op on numpy."""
        if self.xp is np or self.img is None:
            return
        names = ("img", "noisy", "valid", "support")
        for n in names:
            a = getattr(self, n)
            if a is not None:
                setattr(self, n, a.get() if on_host else self.xp.asarray(a))

    def prepare(self, sigmas, release: bool = True):
        """Measure ``K`` at each of *sigmas* now, then (*release*) drop the tile and its noisy copy."""
        self.park(False)
        for s in sigmas:
            if s not in self._k:
                self._k[s] = self._row(s)
        if release:
            self.img = self.noisy = self.valid = self.support = None

    def at(self, s, k, region):
        """``K`` at derivative scale *s* and the *k*-th grid rho over *region* (slices of the tile)."""
        if s not in self._k:
            self._k[s] = self._row(s)
        key = tuple((r.start, r.stop) for r in region)
        if key not in self._up:
            rel = tuple(slice(r.start - a, r.stop - a) for r, a in zip(region, self.lo))
            self._up = {key: Upsampler(rel, self.cells, self.c, self.xp)}
        return self._up[key](self._k[s][k])
