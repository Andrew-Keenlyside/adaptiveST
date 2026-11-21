"""Fixed against adaptive scales on a volume: DEC slices and the scale maps.

    python examples/compare_fixed_adaptive.py VOLUME.npy [--threshold T] [--sigma 1 --rho 3] [--cpu]

VOLUME is a (Z, Y, X) .npy (or any array np.load returns). Writes compare.png beside it.
"""

import argparse
from pathlib import Path

import numpy as np

import adaptivest as ast

p = argparse.ArgumentParser()
p.add_argument("volume")
p.add_argument("--threshold", type=float, default=None)
p.add_argument("--sigma", type=float, default=1.0)
p.add_argument("--rho", type=float, default=3.0)
p.add_argument("--precision", type=float, default=15.0)
p.add_argument("--rho-max", type=float, default=16.0)
p.add_argument("--noise-window", type=float, default=128.0)
p.add_argument("--chunk", default="auto", type=lambda v: v if v == "auto" else int(v))
p.add_argument("--cpu", action="store_true", help="force the CPU (default: the GPU when there is one)")
a = p.parse_args()
vol = np.load(a.volume, mmap_mode="r")
fixed = ast.eigen(ast.structure_tensor(np.asarray(vol), a.sigma, a.rho, gpu=False if a.cpu else None))
scales = ast.AdaptiveScales(precision_deg=a.precision, rho_max=a.rho_max, noise_window=a.noise_window)
S, maps = ast.adaptive_structure_tensor(
    vol, scales, threshold=a.threshold, chunk=a.chunk, gpu=False if a.cpu else None, return_maps=True
)
adapt = ast.eigen(S)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

k = vol.shape[0] // 2
fig, ax = plt.subplots(1, 5, figsize=(25, 5))
ax[0].imshow(vol[k], cmap="gray")
ax[0].set_title("image")
ax[1].imshow(ast.dec(fixed["direction"][:, k]))
ax[1].set_title(f"fixed σ {a.sigma:g}, ρ {a.rho:g}")
ax[2].imshow(ast.dec(adapt["direction"][:, k]))
ax[2].set_title(f"adaptive (θ {a.precision:g}°)")
h = ax[3].imshow(maps["sigma"][k])
ax[3].set_title("σ(x), voxels")
plt.colorbar(h, ax=ax[3])
h = ax[4].imshow(maps["rho"][k])
ax[4].set_title("ρ(x), voxels")
plt.colorbar(h, ax=ax[4])
for x in ax:
    x.set_xticks([])
    x.set_yticks([])
out = Path(a.volume).with_name("compare.png")
fig.tight_layout()
fig.savefig(out, dpi=90)
print("written", out)
