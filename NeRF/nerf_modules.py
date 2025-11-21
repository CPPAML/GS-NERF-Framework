import torch, numpy

def ray_generation(K, Poses, pixels, aabb):
    eps = torch.tensor(1e-3, device=K.device, dtype=K.dtype)
    k_inv = torch.linalg.inv(K)[None, :, :].expand(Poses.shape[0], -1, -1)

    pixels_h = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)  # (V, N_r, 3)
    pixels = pixels_h[..., None] # (V, N_r, 3, 1)

    d_cam = torch.matmul(k_inv[:, None, :, :], pixels)
    d_cam = d_cam.squeeze(-1)  # (V, N_r, 3)
    n = torch.clamp(torch.norm(d_cam, dim=-1, keepdim=True), min=1e-8)
    d_cam = d_cam / n

    R = Poses[:, :3, :3]
    t = Poses[:, :3, 3]
    d_world = torch.matmul(R[:, None, :, :], d_cam).squeeze(-1) # (V, N_r, 3)
    o_world = t[:, None, :].expand(-1, pixels.shape[1], -1) # (V, N_r, 3)

    with torch.no_grad():
        inv_d = 1.0 / torch.where(d_world.abs() > 1e-8, d_world, d_world.sign() * 1e-8)
        t_k1 = (aabb[0] - o_world) * inv_d
        t_k2 = (aabb[1] - o_world) * inv_d

        t_min = torch.minimum(t_k1, t_k2)
        t_max = torch.maximum(t_k1, t_k2)

        t_enter = t_min.amax(dim=-1)
        t_exit = t_max.amin(dim=-1)

        hit_mask = (t_exit > t_enter)
        t_near = torch.maximum(t_enter, eps)
        t_far = t_exit

    return {
        "rays_o": o_world,  # (V, N_r, 3)
        "rays_d": d_world,  # (V, N_r, 3)
        "t_near": t_near,  # (V, N_r)
        "t_far": t_far,  # (V, N_r)
        "hit_mask": hit_mask # (V, N_r)
    }
