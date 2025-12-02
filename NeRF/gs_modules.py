import torch

def ray_sampler(images: torch.Tensor,
                num_rays: int,
                sampling_method: str = "uniform"
                )-> dict:
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


def geom_proj(k: torch.Tensor,
              poses: torch.Tensor,
              mu_w: torch.Tensor,
              ) -> dict:
    mu_w = mu_w.unsqueeze(0).to(k.device) # (1, N, 3)
    mu_w = mu_w.expand(poses.shape[0], -1, -1) # (B, N, 3)

    R = poses[:, :3, :3]
    R_w2c = R.transpose(-1, -2)
    t = poses[:, :3, 3]
    t_w2c = -R_w2c @ t

    x_cam = torch.einsum('bij, bnj->bni', R_w2c, mu_w) + t_w2c # (B, N, 3)

    x = x_cam[..., 0]
    y = x_cam[..., 1]
    z = x_cam[..., 2]

    x_norm = x / z
    y_norm = y / z

    fx = k[..., 0, 0]
    fy = k[..., 1, 1]
    cx = k[..., 0, 2]
    cy = k[..., 1, 2]

    u_proj = fx * x_norm + cx
    v_proj = fy * y_norm + cy

    pixel_coords = torch.stack([u_proj, v_proj], dim=-1)

    return {
        "pixel_coords": pixel_coords, # (B, N, 2)
        "depth": z, # (B, N)
        "x_cam": x_cam # (B, N, 3)
    }

def gaussian_proj(l: torch.Tensor,  # (N, 3, 3)
                  poses: torch.Tensor,  # (B, 4, 4)
                  k: torch.Tensor,  # (B, 3, 3)
                  x_cam: torch.Tensor,  # (B, N, 3)
                  ) -> dict:
    R = poses[:, :3, :3] # (B, 3, 3)
    R_w2c = R.transpose(-1, -2) # (B, 3, 3)

    sigma_w = l @ l.transpose(-1, -2) #(N, 3, 3)
    sigma_w = sigma_w.unsqueeze(0).expand(poses.shape[0], -1, -1, -1) # (B, N, 3, 3)
    sigma_cam = torch.einsum('bik, bnkl, bjl -> bnij', R_w2c, sigma_w, R_w2c.transpose(-1, -2)) # (B, N, 3, 3)

    fx = k[..., 0, 0] # (B,)
    fy = k[..., 1, 1] # (B,)
    fx = fx.view(-1, 1)  # (B, 1)
    fy = fy.view(-1, 1)  # (B, 1)

    x = x_cam[..., 0] # (B, N)
    y = x_cam[..., 1] # (B, N)
    z = x_cam[..., 2] # (B, N)

    eps = 1e-8
    z_safe = torch.clamp(z, min=eps)

    du_dX = fx / z_safe                 # f_x / Z
    du_dY = torch.zeros_like(du_dX)     # 0
    du_dZ = -fx * x / (z_safe * z_safe) # -f_x * X / Z^2

    dv_dX = torch.zeros_like(du_dX)     # 0
    dv_dY = fy / z_safe                 # f_y / Z
    dv_dZ = -fy * y / (z_safe * z_safe) # -f_y * Y / Z^2

    B, N = x.shape
    jacobian = torch.zeros(B, N, 2, 3, device=x_cam.device, dtype=x_cam.dtype)
    # Row 0: [du/dX, du/dY, du/dZ]
    jacobian[..., 0, 0] = du_dX
    jacobian[..., 0, 1] = du_dY
    jacobian[..., 0, 2] = du_dZ

    # Row 1: [dv/dX, dv/dY, dv/dZ]
    jacobian[..., 1, 0] = dv_dX
    jacobian[..., 1, 1] = dv_dY
    jacobian[..., 1, 2] = dv_dZ

    sigma_2d = torch.einsum('bnap, bnpq, bncq -> bnac',
                            jacobian, sigma_cam, jacobian.transpose(-1, -2)) # (B, N, 2, 2)

    return{
        "sigma_2d": sigma_2d, # (B, N, 2, 2)
        "sigma_cam": sigma_cam, # (B, N, 3, 3)
    }

def gaussian_eval(pixels_uv: torch.Tensor, # (B, R, 2)
                  sigma_2d: torch.Tensor, # (B, N, 2, 2)
                  pixel_coords: torch.Tensor, # (B, N, 2)
                  opacity: torch.Tensor, # (B, N)
                  ) -> dict:
    offset_2d = pixels_uv[:, :, None, :] - pixel_coords[:, None, :, :] # (B, R, N, 2)

    l_sigma2d = torch.linalg.cholesky(sigma_2d) # (B, N, 2, 2)
    l_sigma2d = l_sigma2d[:, None, :, :, :].expand(-1, offset_2d.shape[1], -1, -1, -1)

    y = torch.linalg.solve_triangular(l_sigma2d, offset_2d[..., None], upper=False)
    y = y.squeeze(-1)

    mani_dist = torch.sum(y**2, dim=-1)
    gaussians = torch.exp(-1 / 2 * mani_dist) # (B, R, N)

    if opacity.dim() == 1:       # (N,)
        opacity_b = opacity.view(1, 1, -1)        # (1, 1, N)
    elif opacity.dim() == 2:     # (B, N)
        opacity_b = opacity.unsqueeze(1)         # (B, 1, N)
    else:
        raise ValueError("opacity must be (N,) or (B, N)")

    alpha = 1.0 - torch.exp(-opacity_b * gaussians) # (B, R, N)

    return {
        "gaussians": gaussians,
        "alpha": alpha, # (B, R, N)
    }

def gaussian_renderer(alpha: torch.Tensor,  # (B, R, N)
                      rgb: torch.Tensor,    # (B, N, 3)
                      depth: torch.Tensor,  # (B, N)
                      bg_color: torch.Tensor = None  # (3,)
                      ) -> dict:
    eps = 1e-8
    indxs = torch.argsort(depth, dim=-1, descending=True) # (B, N)

    idx_alpha = indxs.unsqueeze(1).expand(-1, alpha.shape[1], -1) # (B, R, N)
    alpha = torch.gather(alpha, dim=-1, index=idx_alpha) # (B, R, N)

    idx_rgb = indxs.unsqueeze(-1).expand(-1, -1, 3) # (B, N, 3)
    rgb = torch.gather(rgb, dim=1, index=idx_rgb) # (B, N, 3)

    depth = torch.gather(depth, dim=-1, index=indxs) # (B, N)

    alpha = alpha.clamp_min(eps)

    one_minus_alpha = 1.0 - alpha + eps # (B, R, N)
    prefix = torch.ones_like(alpha[..., :1]) # (B, R, 1)
    T = torch.cumprod(torch.cat([prefix, one_minus_alpha[..., :-1]], dim=-1), dim=-1)  # (B, R, N)

    weights = T * alpha  # (B, R, N)

    tmp_weights = weights[..., :, None].expand(-1, -1, -1, 3) # (B, R, N, 3)
    tmp_rgb = rgb[:, None, :, :].expand(-1, weights.shape[1], -1, -1) # (B, R, N, 3)
    color_weighted = torch.sum(tmp_weights * tmp_rgb, dim=-2) # (B, R, 3)

    alpha_pred = weights.sum(dim=-1) # (B, R)
    alpha_pred = alpha_pred.clamp_min(eps)

    if bg_color is not None:
        T_bg = torch.cumprod(one_minus_alpha, dim=-1)[..., -1] # (B, R)
        bg_color = bg_color.view(1, 1, 3) # (1, 1, 3)
        color_weighted = color_weighted + T_bg[..., None] * bg_color # (B, R, 3)

    return {
        "alpha_pred": alpha_pred, # (B, R)
        "weights": weights, # (B, R, N)
        "rgb_pred": color_weighted, # (B, R, 3)
    }


import torch

def init_gaussians_from_aabb(
    aabb: torch.Tensor,      # (2, 3): [min; max]
    num_gaussians: int,
    device=None,
    dtype=torch.float32,
):
    if device is None:
        device = aabb.device

    aabb = aabb.to(device=device, dtype=dtype)
    aabb_min = aabb[0] # (3,)
    aabb_max = aabb[1] # (3,)

    u = torch.rand(num_gaussians, 3, device=device, dtype=dtype) # (N, 3)
    mu_world = aabb_min + u * (aabb_max - aabb_min) # (N, 3)

    scene_extent = (aabb_max - aabb_min).norm(p=2) # scalar
    s0 = 0.01 * scene_extent # base scale

    l = torch.zeros(num_gaussians, 3, 3, device=device, dtype=dtype)
    l[:, 0, 0] = s0
    l[:, 1, 1] = s0
    l[:, 2, 2] = s0

    sigma_world = l @ l.transpose(-1, -2)  # (N, 3, 3)

    color = 0.5 + 0.1 * torch.randn(num_gaussians, 3, device=device, dtype=dtype)
    color = color.clamp(0.0, 1.0)

    p0 = 0.05
    ell0 = torch.log(torch.tensor(p0 / (1.0 - p0), device=device, dtype=dtype))
    opacity_logits = ell0 * torch.ones(num_gaussians, device=device, dtype=dtype)

    return {
        "mu_world": mu_world, # (N, 3)
        "L_world": l, # (N, 3, 3)
        "sigma_world": sigma_world, # (N, 3, 3)
        "color": color, # (N, 3)
        "opacity_logits": opacity_logits, # (N,)
    }

