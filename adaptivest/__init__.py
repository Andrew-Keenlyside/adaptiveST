"""adaptiveST: 3D structure tensors at fixed scales, or with each voxel's own sigma and rho from the image.

::

    import adaptivest as ast

    S = ast.structure_tensor(volume, sigma=1.0, rho=3.0)            # fixed scales
    S = ast.adaptive_structure_tensor(volume)                        # adaptive scales (noise measured locally)
    e = ast.eigen(S)                                                  # fa, direction, linearity, planarity
"""

from .adaptive import (
    PRECISION_DEG,
    RHO_MAX,
    RHO_MIN,
    SIGMA0,
    AdaptiveScales,
    adaptive_structure_tensor,
    rho_grid,
    sigma_grid,
)
from .adaptive import scale_maps_volume as scale_maps
from .tensor import COMPONENTS, dec, eigen, structure_tensor

__version__ = "0.2.1"

__all__ = [
    "COMPONENTS",
    "PRECISION_DEG",
    "RHO_MAX",
    "RHO_MIN",
    "SIGMA0",
    "AdaptiveScales",
    "adaptive_structure_tensor",
    "dec",
    "eigen",
    "rho_grid",
    "scale_maps",
    "sigma_grid",
    "structure_tensor",
]
