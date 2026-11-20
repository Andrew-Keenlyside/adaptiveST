"""adaptiveST: 3D structure tensors at fixed scales, or with each voxel's own sigma and rho from the image.

::

    import adaptivest as ast

    S = ast.structure_tensor(volume, sigma=1.0, rho=3.0)            # fixed scales
    scales = ast.calibrate(volume, mask)                             # once per volume: the noise factors
    S = ast.adaptive_structure_tensor(volume, scales, chunk=128)     # adaptive scales
    e = ast.eigen(S)                                                  # fa, direction, linearity, planarity
"""

from .adaptive import (
    PRECISION_DEG,
    RHO_MAX,
    RHO_MIN,
    SIGMA0,
    AdaptiveScales,
    adaptive_structure_tensor,
    calibrate,
    rho_grid,
    sigma_grid,
)
from .adaptive import scale_maps_volume as scale_maps
from .noise import NoiseModel, estimate_noise
from .tensor import COMPONENTS, dec, eigen, structure_tensor

__version__ = "0.1.0"

__all__ = [
    "COMPONENTS",
    "PRECISION_DEG",
    "RHO_MAX",
    "RHO_MIN",
    "SIGMA0",
    "AdaptiveScales",
    "NoiseModel",
    "adaptive_structure_tensor",
    "calibrate",
    "dec",
    "eigen",
    "estimate_noise",
    "rho_grid",
    "scale_maps",
    "sigma_grid",
    "structure_tensor",
]
