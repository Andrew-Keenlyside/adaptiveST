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
    auto_chunk,
    blend_halo,
    rho_grid,
    scale_maps_chunk,
    sigma_grid,
    tensor_from_scales,
)
from .adaptive import scale_maps_volume as scale_maps
from .tensor import COMPONENTS, dec, eigen, structure_tensor

__version__ = "0.6.0"

__all__ = [
    "COMPONENTS",
    "PRECISION_DEG",
    "RHO_MAX",
    "RHO_MIN",
    "SIGMA0",
    "AdaptiveScales",
    "adaptive_structure_tensor",
    "auto_chunk",
    "blend_halo",
    "dec",
    "eigen",
    "rho_grid",
    "scale_maps",
    "scale_maps_chunk",
    "sigma_grid",
    "structure_tensor",
    "tensor_from_scales",
]
