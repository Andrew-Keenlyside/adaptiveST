"""The fixed-scale 3D structure tensor and its eigen-analysis (cupy on a CUDA GPU when there is one, else numpy)."""

from __future__ import annotations

import math

import numpy as np

#: Tensor component order: [zz, yy, xx, zy, zx, yx] (volumes are indexed (z, y, x)).
COMPONENTS = ("zz", "yy", "xx", "zy", "zx", "yx")
_PRODUCTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def gpu_available() -> bool:
    """Whether cupy imports and sees a CUDA device."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001 - no cupy, no driver, no device: all mean CPU
        return False


def backend(gpu: bool | None = None):
    """``(xp, gaussian_filter)``: cupy and cupyx's filter (*gpu* True, or None with a GPU present), or
    numpy and scipy's."""
    if gpu is None:
        gpu = gpu_available()
    if gpu:
        import cupy
        from cupyx.scipy.ndimage import gaussian_filter

        return cupy, gaussian_filter
    from scipy.ndimage import gaussian_filter

    return np, gaussian_filter


def radius(s: float, truncate: float) -> int:
    """Voxels a Gaussian of s.d. *s* reads on each side, as scipy and cupyx size it."""
    return int(float(truncate) * float(s) + 0.5)


def gradients(vol, sigma, truncate, gaussian_filter, xp):
    """``(gz, gy, gx)``: Gaussian derivatives of *vol* at *sigma*, float32."""
    v = xp.asarray(vol, dtype=xp.float32)
    return (
        gaussian_filter(v, sigma, order=[1, 0, 0], mode="nearest", truncate=truncate),
        gaussian_filter(v, sigma, order=[0, 1, 0], mode="nearest", truncate=truncate),
        gaussian_filter(v, sigma, order=[0, 0, 1], mode="nearest", truncate=truncate),
    )


def tensor_from_gradients(grads, rho, truncate, gaussian_filter, xp, dtype=None):
    """``(6, ...)``: the gradients' outer products smoothed at *rho*."""
    dtype = xp.float32 if dtype is None else dtype
    S = xp.empty((6,) + grads[0].shape, dtype=dtype)
    for i, (p, q) in enumerate(_PRODUCTS):
        gaussian_filter(
            (grads[p] * grads[q]).astype(dtype, copy=False),
            rho,
            mode="nearest",
            output=S[i],
            truncate=truncate,
        )
    return S


def structure_tensor(
    volume, sigma: float = 1.0, rho: float = 3.0, *, truncate: float = 3.0, gpu: bool | None = None
):
    """The structure tensor at fixed scales: ``(6, Z, Y, X)`` float32, order :data:`COMPONENTS`.

    *sigma*: the derivative (gradient) scale; *rho*: the integration window;
    both Gaussian standard deviations in voxels. Volume edges are handled as
    ``mode="nearest"``. Runs on the GPU when there is one (*gpu* None), or as
    *gpu* says; returned on the host (numpy) either way.
    """
    xp, gf = backend(gpu)
    S = tensor_from_gradients(gradients(volume, sigma, truncate, gf, xp), rho, truncate, gf, xp)
    return S if xp is np else S.get()


def eigvals_sym3(S, xp=np):
    """``(l1, l2, l3)`` descending, of the symmetric tensors ``S`` (6, ...); closed form (float64)."""
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


def eigen(S, *, slab: int = 16) -> dict:
    """The tensor's eigen-analysis, on the host.

    Returns a dict of ``values`` (3, ...) descending, ``direction`` (3, ...) the
    smallest-eigenvalue eigenvector in (z, y, x) components (the fibre
    direction: the direction the image does not change along; sign arbitrary;
    NaN where the tensor is zero),
    ``fa``, ``linearity`` ``(l2 - l3)/l1`` (tube-like: two strong gradient
    directions) and ``planarity`` ``(l1 - l2)/l1`` (edge- or sheet-like).
    """
    S = np.asarray(S)
    shp = S.shape[1:]
    vals = np.empty((3,) + shp)
    vec = np.empty((3,) + shp, np.float32)
    idx = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}
    for z0 in range(0, shp[0], slab):
        sl = slice(z0, z0 + slab)
        M = np.empty(S[:, sl].shape[1:] + (3, 3))
        for (a, b), i in idx.items():
            M[..., a, b] = M[..., b, a] = S[i, sl]
        w, v = np.linalg.eigh(M)
        vals[:, sl] = np.moveaxis(w[..., ::-1], -1, 0)
        vec[:, sl] = np.moveaxis(v[..., :, 0], -1, 0)
    vec[:, ~(vals.sum(0) > 0)] = np.nan  # a zero tensor (masked out, or flat image) has no direction
    l1, l2, l3 = vals
    m = vals.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = np.sqrt(1.5 * ((vals - m) ** 2).sum(0) / np.maximum((vals**2).sum(0), 1e-300))
        den = np.maximum(l1, 1e-300)
        return {
            "values": vals,
            "direction": vec,
            "fa": fa.astype(np.float32),
            "linearity": ((l2 - l3) / den).astype(np.float32),
            "planarity": ((l1 - l2) / den).astype(np.float32),
        }


def dec(direction, weight=None):
    """Direction-encoded colour ``(..., 3)`` RGB = |x|, |y|, |z| of a (3, ...) (z, y, x) direction field."""
    d = np.abs(np.nan_to_num(np.asarray(direction)))
    rgb = np.stack([d[2], d[1], d[0]], -1)
    if weight is not None:
        rgb = rgb * np.clip(np.asarray(weight), 0, 1)[..., None]
    return np.clip(rgb, 0, 1)
