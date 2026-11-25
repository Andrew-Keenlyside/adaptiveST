# Plan: optional support for anisotropic voxels

Status: plan only, nothing built. Branch `anisotropic-spacing`, off `main` at 0.6.1.

## Goal

Let adaptiveST take volumes whose voxels have different sizes along z, y and x (light-sheet,
serial sections, some CT), through one optional setting:

```python
sc = ast.AdaptiveScales(spacing=(4.0, 1.0, 1.0))      # z, y, x voxel size, any unit
S, maps = ast.adaptive_structure_tensor(volume, sc, return_maps=True)
S = ast.structure_tensor(volume, sigma=1.0, rho=3.0, spacing=(4.0, 1.0, 1.0))
```

Directions come out in physical space; σ, ρ and `rho_max` are in units of the **finest**
voxel. `spacing=None` (the default) must leave every result **bit-identical** to 0.6.1, and
the public signatures that FibreGraph calls (`scale_maps_chunk`, `tensor_from_scales`,
`AdaptiveScales.halo()`, `equivalent_scales()`) must keep working.

## Why not resample first

| | Upsample the coarse axis | Downsample to the coarse spacing |
|---|---|---|
| Works today | yes | yes |
| Cost | volume grows by the anisotropy ratio | in-plane resolution lost |
| Noise estimate | **breaks**: the level correction needs noise above 0.5 cycles/voxel, which interpolation removes | valid |

Native support keeps the full in-plane resolution and a valid noise estimate. The
resampling routes are the baselines the native version has to beat (step 5).

## What is already spacing-proof

- **`K` and the ρ rule.** `K` is measured: the angle one draw of the window's noise moves
  the direction, over `S`. Once the tensor is physical, the precision rule calibrates
  itself, including the loss of samples along the coarse axis.
- **The perturbation factor, eigen steps and fused kernels** work on the tensor only.
- **Masking** (normalised convolution) works per axis without change.

## Design decisions (to settle before step 2)

1. **Units.** Normalise `spacing` by its minimum, `a = spacing / min(spacing)`, so σ and ρ
   stay in finest-voxel units and the defaults (`rho_max` 16, `noise_window` 128) keep
   their meaning. Alternative: physical units throughout (every default then depends on
   the data). *Recommended: normalised.*
2. **The coarse-axis floor.** Along axis i a scale s is `s / a_i` voxels. Below about
   0.5 voxel a Gaussian derivative is undersampled. Options: (a) floor every axis at
   `SIGMA0` = 0.5 voxels, so at small s the kernel is wider along the coarse axis than in
   physical space; (b) no floor, accept aliasing. *Recommended: (a), and record in
   `maps` the physical kernel actually used.* The same floor for ρ is not needed: a
   window narrower than a voxel along z just averages nothing along z, and `K` sees it.
3. **Lindeberg's σ search under a floor.** The normalised energy `s² |∇L_s|²` assumes an
   isotropic scale space. With physical gradients and s in finest-voxel units it is
   unchanged while no axis is floored; once one is, the response along that axis stops
   scaling with s. Options: (a) keep the formula, read s as the in-plane scale;
   (b) normalise per axis, `Σ_i s_i² (∂_i L)²`. *Decide from step-2 phantoms*: on a
   lattice, both should peak near period/2π in-plane; whichever does at ratios 2–8 wins.
4. **Noise bands.** Today the bands are spherical shells of `|k|` in cycles/voxel
   (`band_edges`, `LocalNoise.__init__`). That is right if the noise is white per voxel,
   whatever the spacing. It is wrong if the coarse axis is blurred (light-sheet PSF, thick
   sections, slice interpolation): its high frequencies then hold less noise. Options:
   (a) keep voxel-space shells; (b) two-dimensional bins, in-plane `|k_yx|` × axial
   `|k_z|`, when exactly one axis differs; (c) per-axis product bins in general.
   *Recommended: (b) behind a `noise_bands="voxel" | "axial"` switch, (a) the default*,
   and (c) only if a dataset needs it.
5. **Level reference.** The sub-window level correction uses power above
   `LEVEL_BAND` = 0.5 cycles/voxel (the cube's corners). With a blurred axis that region
   holds little noise along it. With option 4(b), use in-plane `|k_yx| > 0.5` at any
   `k_z`.
6. **Window geometry.** The noise window is `noise_window` finest voxels in physical
   size: `noise_window / a_i` voxels along each axis, 4 sub-windows per axis as now, the
   cell size per axis. Refuse (clear error) when any axis would have fewer than 16
   voxels, or fall back to fewer sub-windows along it.
7. **Settings and hashes.** `spacing` is left out of `as_dict()` when None (as `stride`
   is at 1), so 0.6.x settings dicts and FibreGraph's hashes do not change.

## Steps

Each step ends with targeted tests (not the whole suite), and `spacing=None` checked
bit-identical against `main` on the existing test volumes.

### 1. Plumbing: per-axis scales and physical gradients (½–1 day)

- A helper `_axes(s, a)` → per-axis voxel sigmas (with the floor from decision 2), used
  by every `gaussian_filter` call that takes a scale: about 60 in `adaptive.py`, 13 in
  `local_noise.py`, 11 in `tensor.py`. cupyx and scipy both accept a sigma per axis.
- `_gradients` and `tensor.gradients`: divide each derivative by `a_i`, so the tensor
  is in physical units.
- `structure_tensor(..., spacing=None)` for the fixed path.
- `_radius` → per-axis radii where kernel sizes are computed.
- **Tests:** `spacing=(1,1,1)` and `None` bit-identical to `main`; a cylinder sampled at
  ratio 2 and 4 (partial-volume along z) gives the same fixed-scale direction as the
  isotropic cylinder, within a stated tolerance.

### 2. Scale grids and the σ search (1–2 days, includes decisions 2–3)

- `sigma_map`, `rho_map`, `_variance_at`, `tensor_from_scales`: scales in finest-voxel
  units, per-axis voxel sigmas from `_axes`.
- Lattice steps per axis: `_step(r / a_i, stride)`; `_lattice_products`,
  `_lattice_up`, `_lattice_smooth` take a step per axis.
- Decide 3 with the phantom harness (below) at ratios 1, 2, 4, 8.
- **Tests:** a line-pair lattice at ratio 4 gives σ within 10% of the isotropic run's;
  ρ and direction error within the isotropic run's 95% CI.

### 3. Noise model (1–2 days, decisions 4–6)

- `LocalNoise`: window, sub-windows and cells per axis; band bins per decision 4;
  level reference per decision 5. `KMaps` and `Upsampler`: cells per axis.
- `band_edges` gains the axial variant; `share()` weights by the physical `|k|`.
- **Tests:** white noise at ratio 4: noise share within 5% of 1 and the draw's spectrum
  matching the input's per band; z-blurred noise (Gaussian PSF along z) at ratio 4:
  `"axial"` bands within 10% per band where `"voxel"` bands are not.

### 4. Halos and chunking (~1 day)

- `chain_halo`, `pool_reach`, `blend_halo`, `AdaptiveScales.halo()`: compute per axis;
  add `halo_zyx()`; `halo()` keeps returning one int (the largest) for FibreGraph.
- `_read_padded`, `scale_maps_chunk`, `adaptive_tensor_block`, `tile_scales`: halo per
  axis. `auto_chunk` and `_chunks`: a chunk shape per axis (the coarse axis needs fewer
  voxels for the same physical extent, so tiles get cheaper).
- **Tests:** chunked against one-tile results at ratio 4 agree as closely as they do
  isotropically today (statistical, not bit-for-bit); edge handling unchanged.

### 5. Validation (1–2 days)

- **Phantoms**: extend `phantom_reps.py` (in `/hdd/andrew/adaptivest_phantoms/`): build
  the phantom in physical space, integrate it over anisotropic voxels, add noise white
  per voxel or blurred along z. Grid: ratio 1, 2, 4, 8 × diameter × noise, 6
  replicates, fibre polar angle to the coarse axis recorded. Report direction error
  against that angle: anisotropy should cost accuracy mainly for fibres along the
  coarse axis.
- **Real data with ground truth**: take isotropic HiP-CT crops (I74 20 µm, LADAF
  23.4 µm, the five README VOIs), average every 2/4/8 z-slices to make anisotropic
  copies, and compare their directions with the isotropic run (angle, median and 90th
  percentile) for: native `spacing`, upsample-then-run, downsample-then-run.
- **Accept** if native beats both resampling routes on directions at ratios 2 and 4,
  and the isotropic case is unchanged.

### 6. Documentation (½ day)

README: the `spacing` setting, the floor, the band options, what anisotropy costs
(error against polar angle, from step 5) and the limit: at 5:1 or more, orientations near
the coarse axis are barely sampled, whatever the method does.

## Effort and risks

About a week in total: plumbing and chunking 2 days, the σ search and noise model 2–4
days with decisions, validation and docs 2 days.

- **The σ search under a floor** is the least certain part; it may need a per-axis
  normalisation that changes σ maps (not directions) noticeably.
- **Thin coarse axes**: at ratio 8 the default noise window is 16 voxels along z, at the
  limit of the minimum statistics (4 sub-windows of 4 voxels).
- **Memory planning**: `BYTES_PER_TILE_VOXEL` was measured on cubic tiles; re-measure.
- **FibreGraph**: keeps working while `halo()` returns an int and `spacing` stays out of
  the settings hash; passing spacing through FibreGraph is a separate change there.
