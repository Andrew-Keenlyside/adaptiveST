# adaptiveST

3D structure tensors for volumetric images, at fixed scales or with **each voxel's
own derivative scale σ and integration scale ρ, chosen from the image itself**.
NumPy/SciPy on the CPU; CuPy on a CUDA GPU (optional).

```bash
pip install -e .            # CPU
pip install -e ".[gpu]"     # + CuPy (CUDA 12)
```

## Quick start

```python
import numpy as np
import adaptivest as ast

volume = np.load("volume.npy")                     # (Z, Y, X); zarr / memmap arrays work too

# Fixed scales: one sigma and one rho for every voxel
S = ast.structure_tensor(volume, sigma=1.0, rho=3.0)
e = ast.eigen(S)               # e["direction"] (3, Z, Y, X), e["fa"], e["linearity"], e["planarity"]

# Adaptive scales: measure the image's noise once, then a sigma and rho per voxel
scales = ast.calibrate(volume, threshold=100)       # or calibrate(volume, mask)
S, maps = ast.adaptive_structure_tensor(volume, scales, chunk=128, return_maps=True)
e = ast.eigen(S)
maps["sigma"], maps["rho"]                          # the scales each voxel used, in voxels

rgb = ast.dec(e["direction"][:, 64])                # direction-encoded colour of slice 64
```

Every function takes `gpu=True` to run on CuPy; results come back as NumPy arrays.
The tensor's components are ordered `[zz, yy, xx, zy, zx, yx]` (`ast.COMPONENTS`); the
direction is the eigenvector of the smallest eigenvalue, in (z, y, x) components, with
an arbitrary sign.

## Fixed against adaptive scales

The structure tensor has two Gaussian scales:

| | What it does |
|---|---|
| **σ**, the derivative scale | The image is differentiated with Gaussian derivatives of s.d. σ. It selects the texture whose orientation is measured (the filter responds most to a period of about 2πσ) and suppresses noise above it. |
| **ρ**, the integration scale | The gradients' outer products are averaged with a Gaussian of s.d. ρ. It sets how many gradients agree on one direction: larger ρ averages away more noise and blurs more structure. |

**Fixed scales** apply one (σ, ρ) to the whole volume. That is a compromise wherever
noise and contrast vary: a window large enough for the noisiest, lowest-contrast tissue
blurs well-resolved structure elsewhere, and a window small enough for that structure
leaves the noisy tissue as speckle.

**Adaptive scales** give each voxel the smallest window that meets a stated precision,
and a derivative scale matched to its local texture.

- **ρ(x): the smallest window that reaches a target angular precision.** For each ρ on
  the grid, the tensor predicts the angular variance of its direction as
  `V = K(σ, ρ) · S`, with `S = (λ1 + λ3)/(λ1 − λ3)² + (λ2 + λ3)/(λ2 − λ3)²`. That is the
  first-order perturbation of the smallest eigenvector: large eigenvalue gaps, a
  well-determined direction. `K(σ, ρ)` is fitted once per volume (`calibrate`): noise
  drawn from the image's own fitted noise spectrum is added to a few sampled blocks,
  and `K` is set so `V` reproduces the mean squared angle the noise moves the
  direction. ρ(x) is where V (smoothed over 2 voxels) first falls to
  `precision_deg²`, interpolated in log ρ.
- **σ(x): Lindeberg's scale over the voxel's own window.** The first peak over scale
  of the normalised gradient energy `s² |∇L_s|²`, averaged with a Gaussian of width
  ρ(x). For a lattice of fibres the peak sits at the lattice period over 2π; where
  the finest texture is noise, at about 0.7 voxels.
- **Three passes.** ρ at σ = 0.5 (nearly no smoothing, so the first window is not
  skewed by an assumed σ); σ over that window; ρ again at each voxel's σ. Further
  passes change ρ by under 0.2% (measured on HiP-CT), so one is enough.
- **The tensor** blends trace-normalised tensors at the neighbouring grid (σ, ρ),
  bilinearly in (log σ, log ρ), so both scales vary continuously from voxel to voxel.
- **The grids** are log-spaced at ratio √2, two samples per octave: ρ from √2
  (Lindeberg's smallest integration factor) to `rho_max`, σ from 0.5 to `rho_max/√2`.

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `precision_deg` | 15 | the target angular s.d. of each direction under the image's noise; smaller gives larger windows |
| `rho_max` | 16 | the largest window (voxels); also sets σ's range, the halo and much of the cost |
| `noise_model` | `"poisson"` | `"poisson"`: noise variance proportional to the local mean intensity; `"gaussian"`: the same everywhere |
| `truncate` | 3 | Gaussian kernel radius in s.d. |

**Choosing the precision from a tracker's tolerance.** If a tractography step rejects
a link whose direction differs by more than α from a node's direction, noise of s.d. θ
wrongly rejects a correct link with probability about `exp(−α²/2θ²)`. A false-rejection
rate p gives `θ = α / √(2 ln(1/p))`: α = 40° and p = 5% give θ ≈ 16°.

**What is measured, what is chosen.** Measured: the noise spectrum and `K`; at every
voxel S, ρ and σ. Chosen: θ; `rho_max`; the √2 grid spacing and its ends (theory:
scale-space sampling and Lindeberg's integration factor); 2-voxel smoothings of V and
log ρ; the noise model's fitting band; how many calibration blocks.

## Example: 20 µm HiP-CT (human brain, deep grey and white matter)

On a 384³ crop at 20 µm, with θ = 15° and `rho_max` = 16:

- **σ** stays near 0.75–0.85 voxels almost everywhere: the finest texture is at the
  noise's own scale. It rises to about 1.1–1.25 over coarse, streaky fascicular texture.
- **ρ** ranges from about 3.5 to 9.5 voxels (10th–90th percentile). It is smaller in
  well-oriented fascicles, larger in noisy, low-contrast tissue.
- In a tractography pipeline built on this tensor (BRIDGE, with the graph limits
  calibrated separately for each variant), the adaptive tensor gave:

| | Fixed σ 1, ρ 3 | Fixed σ 1, ρ 8 | Adaptive |
|---|---|---|---|
| tracks ending inside deep white matter, per cm | 0.148 | 0.030 | 0.004 |
| median end-to-end extent (voxels) | 249 | 311 | 326 |
| straightness | 0.846 | 0.898 | 0.899 |

Most of the improvement over σ 1, ρ 3 comes from a larger window at all. Adaptivity
adds a further drop in tracks ending inside white matter and some length, at a much
higher cost. There is no ground truth for fibre directions in this tissue.

## Cost

The method computes several tensors per voxel: about 8 window sizes in each ρ pass,
the Lindeberg search, and the blend.

- **Halo:** 94 voxels at `rho_max` 16 and `truncate` 3 (against 12 for σ 1, ρ 3). Use
  `chunk` to bound memory; the result does not depend on it.
- **Speed:** tens of seconds per 128³ chunk on a laptop-class GPU, much more on the CPU.
- **Calibration:** about 1.5 minutes once per volume (CPU).

Lowering `rho_max` cuts all three.

## Limitations

- **The noise factors set the level of ρ.** Without half-maps, noise and the finest
  texture cannot be separated. Two calibrations of one image (different blocks and
  fitting details) differed by up to several times in K, which moves ρ by about 1.6×.
  More calibration blocks help.
- **The noise model is isotropic.** Streaks along one axis are not represented
  (`NoiseModel.anisotropic` reports them).
- **On reconstructed CT the noise may not follow intensity.** On the HiP-CT example
  it did not, and `"gaussian"` fits better than `"poisson"`.
- **Bright specks and strong edges get small windows.** High contrast predicts a
  precise direction even when that direction is the speck's or the edge's.
- **Bias is not modelled.** The rule limits noise only; a large window in weak texture
  can still average across neighbouring structures.
- **Scalars from the adaptive tensor change meaning with the scales.** FA and
  linearity are computed over windows that differ from voxel to voxel.

## API

| Function | |
|---|---|
| `structure_tensor(volume, sigma, rho, *, truncate=3, gpu=False)` | fixed-scale tensor, (6, Z, Y, X) |
| `calibrate(volume, mask=None, *, threshold=None, precision_deg=15, rho_max=16, noise_model="poisson", truncate=3, n_blocks=8, block=32, seed=0)` | the `AdaptiveScales` (noise factors) for a volume |
| `adaptive_structure_tensor(volume, scales=None, *, chunk=None, gpu=False, return_maps=False, ...)` | per-voxel-scale tensor; calibrates first when `scales` is None |
| `scale_maps(volume, scales, *, chunk=None, gpu=False)` | `{"sigma", "rho"}` maps only |
| `eigen(S)` | `values`, `direction`, `fa`, `linearity`, `planarity` |
| `dec(direction, weight=None)` | direction-encoded colour |
| `AdaptiveScales.as_dict()` / `from_dict()` | save and reuse a calibration |

## Tests

```bash
pytest -q             # CPU
pytest -q -m gpu      # CuPy against NumPy
```
