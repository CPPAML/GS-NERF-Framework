# Re-export functional Gaussian Splatting pieces under the renderers namespace
from NeRF.gs_modules import (
    ray_sampler as gs_ray_sampler,
    geom_proj,
    gaussian_proj,
    gaussian_eval,
    gaussian_renderer,
    init_gaussians_from_aabb,
)

__all__ = [
    "gs_ray_sampler",
    "geom_proj",
    "gaussian_proj",
    "gaussian_eval",
    "gaussian_renderer",
    "init_gaussians_from_aabb",
]
