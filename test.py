from OmniObject3D import OmniObject3D
from torch.utils.data import DataLoader
from nerf_modules import ray_generation, ray_sampler, depth_sampling

ds = OmniObject3D(object_id="anise_001", views_per_batch=4)

for batch in ds:
    num_rays = 16
    num_samples_per_ray = 32
    ray_samples = ray_sampler(batch["images"], num_rays)
    ray_dict = ray_generation(batch["Ks"], batch["poses"], ray_samples["pixels_uv"], batch["aabb"])
    depth_samples = depth_sampling(ray_dict["t_near"], ray_dict["t_far"], num_rays, num_samples_per_ray)
