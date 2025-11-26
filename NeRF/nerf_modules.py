import torch
def ray_generation(K,
                   Poses,
                   pixels,
                   aabb
                   ):
    with torch.no_grad():
        eps = torch.tensor(1e-3, device=K.device, dtype=K.dtype)
        k_inv = torch.linalg.inv(K)[:, :].expand(Poses.shape[0], -1, -1)

        pixels_h = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)  # (V, N_r, 3)
        pixels = pixels_h[..., None] # (V, N_r, 3, 1)

        d_cam = torch.matmul(k_inv[:, None, :, :], pixels)
        d_cam = d_cam.squeeze(-1)  # (V, N_r, 3)
        n = torch.clamp(torch.norm(d_cam, dim=-1, keepdim=True), min=1e-8)
        d_cam = d_cam / n

        R = Poses[:, :3, :3]
        t = Poses[:, :3, 3]

        d_world = torch.matmul(d_cam, R.transpose(-1, -2)) # (V, N_r, 3)
        o_world = t[:, None, :].expand(-1, pixels.shape[1], -1) # (V, N_r, 3)

        with torch.no_grad():
            sign = torch.sign(d_world)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            safe_d = torch.where(d_world.abs() > eps, d_world, sign * eps)
            inv_d = 1.0 / safe_d
            t_k1 = (aabb[0] - o_world) * inv_d
            t_k2 = (aabb[1] - o_world) * inv_d

            t_min = torch.minimum(t_k1, t_k2)
            t_max = torch.maximum(t_k1, t_k2)

            t_enter = t_min.amax(dim=-1)
            t_exit = t_max.amin(dim=-1)

            hit_mask = (t_exit > t_enter)
            t_near = torch.maximum(t_enter, eps)
            t_far = t_exit

            # Filter out rays that are behind the camera or invalid
            valid_interval = (t_far > t_near)
            hit_mask = hit_mask & valid_interval
            
            # Ensure t_far >= t_near so downstream doesn't explode with negative delta
            t_far = torch.where(valid_interval, t_far, t_near)

    return {
        "rays_o": o_world,  # (V, N_r, 3)
        "rays_d": d_world,  # (V, N_r, 3)
        "t_near": t_near,  # (V, N_r)
        "t_far": t_far,  # (V, N_r)
        "hit_mask": hit_mask # (V, N_r)
    }

def ray_sampler(images,
                num_rays,
                sampling_method: str = "uniform"
                ):
    with torch.no_grad():
        height, width = images.shape[-3:-1]
        u_idx = torch.randint(0, width, (images.shape[0],num_rays), device=images.device, dtype=torch.int64)
        v_idx = torch.randint(0, height, (images.shape[0],num_rays), device=images.device, dtype=torch.int64)

        batch = torch.arange(images.shape[0], device=images.device)[:, None]
        pixels_gt = images[batch, v_idx, u_idx , :]

        u, v = u_idx.float() + 0.5, v_idx.float() + 0.5
        pixels_uv = torch.stack([u, v], dim=-1)

        return {
            "pixels_uv": pixels_uv, # (V, Nr, 2)
            "pixels_gt": pixels_gt,  # (V, Nr, 3)
            "num_rays": num_rays,
        }


def depth_sampling(t_near,
                   t_far,
                   num_rays,
                   num_samples_per_ray,
                   method: str = "stratified",  # or deterministic,
                   ):
    with torch.no_grad():
        if method == "deterministic":
            z_k = (torch.arange(num_samples_per_ray, device=t_near.device) + 0.5) / num_samples_per_ray
            z_k = z_k.expand(t_near.shape[0], num_rays, num_samples_per_ray)
            t_vals = t_near[...,None] + (t_far - t_near)[...,None] * z_k
            delta = t_vals[..., 1:] - t_vals[..., :-1]
            delta = torch.cat([delta, delta[..., -1:]], dim=-1)
            return {
                "t_vals": t_vals,
                "delta": delta
            }
        elif method == "stratified":
            z_k = torch.linspace(0, 1, num_samples_per_ray + 1, device=t_near.device)
            z_left, z_right = z_k[:-1], z_k[1:]
            u = torch.rand((t_near.shape[0], num_rays, num_samples_per_ray), device=t_near.device)
            z_prime = z_left + (z_right - z_left) * u
            t_vals = t_near[..., None] + (t_far - t_near)[..., None] * z_prime
            delta = t_vals[..., 1:] - t_vals[..., :-1]
            delta = torch.cat([delta, delta[..., -1:]], dim=-1)
            return {
                "t_vals": t_vals,
                "delta": delta
            }
        else:
            raise ValueError(f"Unknown sampling method: {method}, please use one of 'deterministic' or 'stratified'.")


def field_query(rays_o,
                rays_d,
                t_vals,
                l,
                concat_enc,
                ):
    x = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., None]
    x_enc = positional_encoding(x, l, concat_enc)
    x_enc = x_enc.flatten(0, -2)

    return x_enc


def positional_encoding(x,
                        l: int = 10,
                        concat_enc: bool = False):
    freqs = 2.0 ** torch.arange(l, device=x.device, dtype=x.dtype)
    x_in = x
    x = x_in[..., None] * freqs[None, :] * torch.pi
    enc = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    if concat_enc:
        enc = torch.cat([enc, x_in[..., None]], dim=-1)
    return enc.reshape(*x.shape[:-2], -1)


def volume_renderer(sigma,
                    rgb,
                    delta,
                    t_vals = None,
                    hit_max = None,
                    bg_color = None):
    eps = torch.tensor(1e-10, device=sigma.device, dtype=sigma.dtype)
    alpha = 1.0 - torch.exp(-torch.clamp(sigma,0.0,50.0) * delta[..., None])

    t_i = torch.cumprod(1.0 - alpha + eps, dim=-2)
    t_i = torch.cat([torch.ones_like(t_i[..., :1, :]), t_i[..., :-1, :]], dim=-2) # (V, Nr, N, 1)

    weights = alpha * t_i
    if hit_max is not None:
        weights = weights * hit_max[..., None, None]

    accumulated_opacity = torch.sum(weights, dim=-2) # (V, Nr, 1)
    rgb_pred = torch.sum(weights * rgb, dim=-2) # (V, Nr, 3)
    depth = torch.sum(weights * t_vals[..., None], dim=-2) if t_vals is not None else None # (V, Nr, 1)

    if bg_color is not None:
        rgb_pred += (1.0 - accumulated_opacity) * bg_color.view(1, 1, 3)

    return{
        "rgb_pred": rgb_pred, # (V, Nr, 3)
        "depth": depth, # (V, Nr, 1)
        "accumulated_opacity": accumulated_opacity, # (V, Nr, 1)
        "weights": weights, # (V, Nr, N, 1)
    }

def importance_resampler(weights,
                         t_vals,
                         num_fine):
    eps = 1e-4
    with torch.no_grad():
        weights = weights[..., :-1, 0]
        p_i = (weights + eps) / torch.sum(weights + eps, dim=-1, keepdim=True)
        cdf = torch.cumsum(p_i, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1).clamp(0.0, 1.0)
        u = torch.rand(*weights.shape[:2], num_fine, device=weights.device)
        inds = torch.searchsorted(cdf, u, right=True)
        inds = torch.clamp(inds, 1, cdf.shape[-1]-1)
        inds_lo = inds - 1
        gather = lambda X, I: torch.gather(X, -1, I)
        cdf_lo = gather(cdf, inds_lo)
        cdf_hi = gather(cdf, inds)
        t_lo = gather(t_vals, inds_lo)
        t_hi = gather(t_vals, inds)
        denom = torch.clamp(cdf_hi - cdf_lo, min=eps)
        frac = (u - cdf_lo) / denom
        t_fine = t_lo + frac * (t_hi - t_lo)
    return t_fine

def delta_recomputation(t_vals,
                        t_fine,
                        ):
    with torch.no_grad():
        t_combined = torch.cat([t_vals, t_fine], dim=-1)
        t_combined = torch.sort(t_combined, dim=-1).values
        delta_main = t_combined[..., 1:] - t_combined[..., :-1]
        delta_last = t_combined[..., -1:] - t_combined[..., -2:-1]
        delta = torch.cat([delta_main, delta_last], dim=-1)
        delta_new = torch.clamp(delta, min=1e-6)
    return {
        "t_combined":t_combined,
        "delta_new":delta_new
    }

def dir_encoding(rays_d,
                 samples_per_ray,
                ):
    rays_dir = rays_d.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1)
    rays_dir = positional_encoding(rays_dir)
    rays_dir = rays_dir.flatten(0, -2)
    return rays_dir

def model_stabilizer(sigma,
                     noise
                     ):
    return sigma + (torch.randn_like(sigma) * noise)


def loss_computation(rgb_pred_coarse, rgb_pred_fine, rgb_gt):
    coarse_loss = torch.mean((rgb_pred_coarse - rgb_gt)**2)
    fine_loss = torch.mean((rgb_pred_fine - rgb_gt)**2)
    photometric_loss = coarse_loss + fine_loss
    psnr = 10 * torch.log10(1.0 / torch.clamp(fine_loss, min=1e-10))
    return photometric_loss, psnr