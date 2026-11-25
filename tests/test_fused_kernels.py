"""The fused elementwise kernels of the scale searches give the bit pattern of the unfused expressions."""

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu

import adaptivest.adaptive as A  # noqa: E402

if cp.cuda.runtime.getDeviceCount() == 0:  # pragma: no cover
    pytest.skip("no CUDA device", allow_module_level=True)

N = (24, 30, 36)


def _r(rng, lo, hi, dt=np.float32):
    return cp.asarray(rng.uniform(lo, hi, N).astype(dt))


def test_rho_step_kernels_bit_identical():
    rng = np.random.default_rng(0)
    sc = A.AdaptiveScales()
    target = 2.0 * math.log(math.radians(sc.precision_deg))
    lr = np.log(np.asarray(sc.rho_grid, float))
    v, vk, w = _r(rng, 1e-9, 0.05), _r(rng, 0, 3e-3), _r(rng, 0, 1)
    ref_v = v.copy()
    ref_v += w * vk
    got_v = v.copy()
    A._ew("axpy")(w, vk, got_v)
    assert cp.array_equal(ref_v, got_v)
    lv = cp.log(cp.maximum(ref_v, 1e-8))
    lrho = cp.where(lv <= target, float(lr[0]), cp.nan)
    lv2, lrho2 = A._ew("rho_first")(got_v, np.float32(target), np.float64(lr[0]))
    assert cp.array_equal(lv, lv2)
    assert cp.array_equal(lrho, lrho2, equal_nan=True)
    v2 = _r(rng, 1e-9, 0.05) * _r(rng, 0, 1)
    lv_n = cp.log(cp.maximum(v2, 1e-8))
    f = cp.clip((lv - target) / cp.maximum(lv - lv_n, 1e-12), 0.0, 1.0)
    ref = cp.where((lv_n <= target) & cp.isnan(lrho), float(lr[0]) + f * float(lr[1] - lr[0]), lrho)
    got, out = lrho2.copy(), cp.empty_like(lv)
    A._ew("rho_next")(
        v2, lv2, np.float32(target), np.float32(lr[0]), np.float32(float(lr[1] - lr[0])), out, got
    )
    assert cp.array_equal(lv_n, out)
    assert cp.array_equal(ref, got, equal_nan=True)


def test_sigma_step_kernels_bit_identical():
    rng = np.random.default_rng(1)
    gz, gy, gx = _r(rng, -1, 1), _r(rng, -1, 1), _r(rng, -1, 1)
    s = 1.41421356
    assert cp.array_equal(
        (s * s) * (gz * gz + gy * gy + gx * gx), A._ew("energy")(gz, gy, gx, np.float32(s * s))
    )
    acc, w, f = _r(rng, 0, 5), _r(rng, 0, 1, np.float64), _r(rng, 0, 9)
    ref = acc.copy()
    ref += (w * f).astype(cp.float32)
    got = acc.copy()
    A._ew("acc")(w, f, got)
    assert cp.array_equal(ref, got)
    prev, du = _r(rng, -3, 3), np.float64(0.3465735902799727)
    L = cp.log(cp.maximum(acc, cp.float32(1e-30)))
    d = ((L - prev) / du).astype(cp.float32)
    L2, d2 = A._ew("slope")(acc, prev, np.float32(du))
    assert cp.array_equal(L, L2) and cp.array_equal(d, d2)
