import torch

from dataio.OmniObject3D import OmniObject3D
from NeRF.neural_architectures import NeRF
from NeRF.nerf_modules import *
import os, datetime, shutil, platform, sys
import glob
import importlib.util
import numpy as np
import cv2

def find_cl_path():
    roots = [
        r"C:\Program Files (x86)\Microsoft Visual Studio",
        r"C:\Program Files\Microsoft Visual Studio"
    ]
    found_paths = []
    for root in roots:
        if not os.path.exists(root):
            continue
        pattern = os.path.join(root, "*", "*", "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64", "cl.exe")
        found_paths.extend(glob.glob(pattern))
    
    if not found_paths:
        return None
    
    found_paths.sort(reverse=True)
    return found_paths[0]

class NerfWrapper:
    def __init__(self,
                 object_id: str,
                 views_per_batch: int = 24,
                 num_rays: int = 16,
                 num_samples_per_ray: int = 32,
                 ray_sampler_method: str = "uniform",
                 depth_sampler_method: str = "stratified",
                 num_fine_samples: int = 16,
                 use_bgcolor: bool = True,
                 num_hidden_layers: int = 8,
                 hidden_dim_width: int = 256,
                 optimizer_factory=torch.optim.AdamW,
                 learning_rate: float = 1e-3,
                 l_dimensionality: int = 10,
                 concat_enc: bool = True,
                 use_amp: bool = True,
                 seed: int = 42,
                 compile: bool = True,
                 dataset_split_strategy: str = "random",
                 val_ratio: float = 0.15,
                 view_type: str = "uniform",
                 ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")
        print("Python executable:", sys.executable)
        print("Torch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        self.views_per_batch = views_per_batch
        self.num_fine_samples = num_fine_samples
        self.num_rays = num_rays
        self.seed = seed
        self.view_type = view_type
        self.compile = compile
        self.object_id = object_id
        self.val_ratio = val_ratio
        self.use_amp = use_amp and torch.cuda.is_available()
        self.concat_enc = concat_enc
        self.learning_rate = learning_rate
        self.l_dimensionality = l_dimensionality
        self.num_samples_per_ray = num_samples_per_ray
        self.ray_sampler_method = ray_sampler_method
        self.depth_sampler_method = depth_sampler_method
        self.use_bgcolor = use_bgcolor
        self.run_dir = None

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.ds = OmniObject3D(object_id=self.object_id,
                               views_per_batch=self.views_per_batch,
                               view_type=self.view_type,
                               seed=self.seed,
                               val_ratio=self.val_ratio,
                               split_strategy=dataset_split_strategy,
                              )

        self.NeRF = NeRF(hidden_dim=hidden_dim_width,
                         hidden_layers=num_hidden_layers,
                         input_dim=(3 * 2 * self.l_dimensionality)
                         if not self.concat_enc else (3 * 2 * self.l_dimensionality + 3)
                         ).to(self.device)

        if self.compile:
            if self.compile and self.device.type == "cuda" and importlib.util.find_spec("triton") is None:
                print("Warning: Triton not found. Disabling torch.compile to prevent runtime errors on CUDA.")
                self.compile = False

            if self.compile and platform.system() == "Windows" and shutil.which("cl") is None:
                # Try to find it manually
                cl_path = find_cl_path()
                if cl_path:
                    cl_dir = os.path.dirname(cl_path)
                    print(f"Found MSVC compiler at: {cl_path}")
                    print(f"Adding {cl_dir} to PATH.")
                    os.environ["PATH"] += os.pathsep + cl_dir
                else:
                    print("Warning: MSVC compiler (cl.exe) not found. Disabling torch.compile to prevent runtime errors.")
                    self.compile = False

            if self.compile:
                # On Windows, 'reduce-overhead' (Inductor+CUDAGraphs) can cause OverflowError.
                # We use default mode (Inductor) instead, which works.
                compilation_mode = "reduce-overhead"
                if platform.system() == "Windows":
                    compilation_mode = "default"

                try:
                    self.NeRF = torch.compile(self.NeRF, mode=compilation_mode)
                except Exception:
                    print("torch.compile failed; continuing without compile.")

        self.optimizer = optimizer_factory(self.NeRF.parameters(), lr=self.learning_rate)

        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        if self.use_bgcolor:
            self.bgcolor = torch.zeros(3, dtype=torch.float32, device=self.device)

    # ==============================================================
    # =======================  TRAIN LOOP  =========================
    # ==============================================================

    def train(self, iters=10000, log_every=50, eval_every=500, ckpt_every=1000, clip_grad=1.0,
              preview_every=500,           # SAVE: how often (in steps) to snapshot a render
              preview_downscale=2,         # SAVE: 1 = full res; 2,3,... = faster
              preview_chunk_rays=8192,     # SAVE: memory/speed tradeoff
              preview_view: int = 0,       # SAVE: which dataset view to render
              save_depth_preview: bool = True):
        self.NeRF.train()
        best_val = float("inf")
        # NEW: create a run dir
        self.run_dir = getattr(self, "run_dir", None) or self._make_run_dir()

        print(f"Starting training loop for {iters} iterations...")
        for step, batch in enumerate(self.ds):
            for k, v in batch.items():
                if k != "images":
                    batch[k] = v.to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                # 0) sample rays
                ray_s = ray_sampler(batch["images"], self.num_rays, sampling_method=self.ray_sampler_method)
                ray_s["pixels_uv"] = ray_s["pixels_uv"].to(self.device)
                ray_s["pixels_gt"] = ray_s["pixels_gt"].to(self.device)

                # 1) generate rays
                rays = ray_generation(batch["Ks"], batch["poses"], ray_s["pixels_uv"], batch["aabb"])
                hit_ratio = rays["hit_mask"].float().mean().item()
                if (step + 1) % log_every == 0 or step == 0:
                    print(f"  hit_ratio={hit_ratio:.3f}")
                # 2) sample depths
                depth = depth_sampling(rays["t_near"], rays["t_far"], self.num_rays,
                                       self.num_samples_per_ray, method=self.depth_sampler_method)
                # 3) encode positions
                x_enc = field_query(rays["rays_o"], rays["rays_d"], depth["t_vals"],
                                    self.l_dimensionality, self.concat_enc)
                V, R, N = rays["rays_o"].shape[0], rays["rays_o"].shape[1], depth["t_vals"].shape[-1]
                # 4) coarse network
                if step == 0 and self.compile:
                    print("  [NerfWrapper] Performing first forward pass (compilation may take a while)...")

                sigma, rgb = self.NeRF(x_enc, V, R, N)
                sigma = model_stabilizer(sigma, noise=0.0)
                if (step + 1) % 50 == 0 or step == 0:
                    with torch.no_grad():
                        print("sigma mean:", sigma.mean().item(), "rgb mean:", rgb.mean().item())

                # 5) coarse volume render
                vol = volume_renderer(sigma, rgb, depth["delta"], t_vals=depth["t_vals"],
                                      hit_max=rays["hit_mask"],
                                      bg_color=self.bgcolor if self.use_bgcolor else None)
                # 6) importance resample
                t_fine = importance_resampler(vol["weights"], depth["t_vals"], self.num_fine_samples)
                # 7) recompute deltas
                delta = delta_recomputation(depth["t_vals"], t_fine)
                # 8) fine query + render
                x_fine = field_query(rays["rays_o"], rays["rays_d"], delta["t_combined"],
                                     self.l_dimensionality, self.concat_enc)
                sigma_f, rgb_f = self.NeRF(x_fine, V, R, delta["t_combined"].shape[-1])
                sigma_f = model_stabilizer(sigma_f, noise=0.0)
                vol_f = volume_renderer(sigma_f, rgb_f, delta["delta_new"], t_vals=delta["t_combined"],
                                        hit_max=rays["hit_mask"],
                                        bg_color=self.bgcolor if self.use_bgcolor else None)
                # 9) loss
                loss, psnr = loss_computation(vol["rgb_pred"], vol_f["rgb_pred"], ray_s["pixels_gt"])

            if not torch.isfinite(loss):
                print("Non-finite loss encountered. Lower LR / disable AMP / check NaNs.")
                break

            # backward / optimize
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.NeRF.parameters(), clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.NeRF.parameters(), clip_grad)
                self.optimizer.step()

            # logging
            if (step + 1) % log_every == 0 or step == 0:
                print(f"[{step + 1}] train_loss={loss.item():.4f}  psnr={psnr.item():.2f}dB")

            # --- periodic render snapshot ---
            if preview_every is not None and preview_every > 0 and ((step + 1) % preview_every == 0):
                self._render_and_save_preview(
                    step=step + 1,
                    view_id=preview_view,
                    downscale=preview_downscale,
                    save_depth=save_depth_preview,
                    chunk_rays=preview_chunk_rays
                )

            if (step + 1) % eval_every == 0:
                val_loss, val_psnr = self.evaluate(max_views=2)
                print(f"   ↳ val_loss={val_loss:.4f}  val_psnr={val_psnr:.2f}dB")
                if val_loss < best_val:
                    best_val = val_loss
                    self.save_checkpoint(step + 1, is_best=True)

            if (step + 1) % ckpt_every == 0:
                self.save_checkpoint(step + 1)

            if step + 1 >= iters:
                break

    # ==============================================================
    # =======================  EVALUATION  =========================
    # ==============================================================

    def evaluate(self, max_views=4):
        self.NeRF.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            total_loss, total_psnr, count = 0.0, 0.0, 0
            for i, batch in enumerate(self.ds.val_iter(max_views)):
                for k, v in batch.items():
                    if k != "images":
                        batch[k] = v.to(self.device)

                ray_s = ray_sampler(batch["images"], self.num_rays, sampling_method=self.ray_sampler_method)
                ray_s["pixels_uv"] = ray_s["pixels_uv"].to(self.device)
                ray_s["pixels_gt"] = ray_s["pixels_gt"].to(self.device)

                rays = ray_generation(batch["Ks"], batch["poses"], ray_s["pixels_uv"], batch["aabb"])
                depth = depth_sampling(rays["t_near"], rays["t_far"], self.num_rays,
                                       self.num_samples_per_ray, method=self.depth_sampler_method)

                x_enc = field_query(rays["rays_o"], rays["rays_d"], depth["t_vals"],
                                    self.l_dimensionality, self.concat_enc)
                V, R, N = rays["rays_o"].shape[0], rays["rays_o"].shape[1], depth["t_vals"].shape[-1]
                sigma, rgb = self.NeRF(x_enc, V, R, N)
                sigma = model_stabilizer(sigma, noise=0.0)
                vol = volume_renderer(sigma, rgb, depth["delta"], t_vals=depth["t_vals"],
                                      hit_max=rays["hit_mask"],
                                      bg_color=self.bgcolor if self.use_bgcolor else None)

                # hierarchical fine pass
                t_fine = importance_resampler(vol["weights"], depth["t_vals"], self.num_fine_samples)
                delta = delta_recomputation(depth["t_vals"], t_fine)
                x_fine = field_query(rays["rays_o"], rays["rays_d"], delta["t_combined"],
                                     self.l_dimensionality, self.concat_enc)
                sigma_f, rgb_f = self.NeRF(x_fine, V, R, delta["t_combined"].shape[-1])
                sigma_f = model_stabilizer(sigma_f, noise=0.0)
                vol_f = volume_renderer(sigma_f, rgb_f, delta["delta_new"], t_vals=delta["t_combined"],
                                        hit_max=rays["hit_mask"],
                                        bg_color=self.bgcolor if self.use_bgcolor else None)

                loss, psnr = loss_computation(vol["rgb_pred"], vol_f["rgb_pred"], ray_s["pixels_gt"])
                total_loss += loss.item()
                total_psnr += psnr.item()
                count += 1
                if i + 1 >= 1:
                    break  # lightweight validation
        self.NeRF.train()
        return total_loss / max(count, 1), total_psnr / max(count, 1)

    # ==============================================================
    # =====================  CHECKPOINT I/O  =======================
    # ==============================================================

    def save_checkpoint(self, step, is_best=False):
        state = {
            "step": step,
            "model": self.NeRF.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "cfg": {
                "views_per_batch": self.views_per_batch,
                "num_rays": self.num_rays,
                "num_samples_per_ray": self.num_samples_per_ray,
                "num_fine_samples": self.num_fine_samples,
            }
        }
        path = f"ckpt_{step}.pth"
        torch.save(state, path)
        if is_best:
            torch.save(state, "ckpt_best.pth")

    def load_checkpoint(self, path="ckpt_best.pth"):
        state = torch.load(path, map_location=self.device)
        self.NeRF.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None and self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])
        print(f"Resumed from step {state['step']}")
        return state["step"]

    # === Add these methods inside NerfWrapper ===================================

    @torch.no_grad()
    def render_view(self,
                    K: torch.Tensor = None,
                    pose: torch.Tensor = None,
                    H: int = None,
                    W: int = None,
                    aabb: torch.Tensor = None,
                    chunk_rays: int = 8192,
                    deterministic_coarse: bool = True,
                    return_depth: bool = True,
                    bg_color: torch.Tensor = None):
        """
        Renders a single view from (K, pose, H, W).
        Returns:
            rgb_img:  (H, W, 3) torch.float32 on CPU, in [0,1]
            depth:    (H, W, 1) torch.float32 on CPU (if return_depth=True)
        """
        self.NeRF.eval()
        dev = self.device

        # Defaults from dataset if not provided
        if K is None:        K = self.ds.K
        if pose is None:     pose = self.ds.frames[self.ds.indexes[0]]
        if H is None or W is None:
            H = getattr(self.ds, "height")
            W = getattr(self.ds, "width")

        if aabb is None:     aabb = self.ds.aabb
        if bg_color is None:
            bgc = (self.bgcolor if self.use_bgcolor else None)
        else:
            bgc = bg_color.to(dev, dtype=torch.float32)

        # Prep tensors
        K = K.to(dev, dtype=torch.float32).unsqueeze(0)  # (1,3,3)
        pose = pose.to(dev, dtype=torch.float32).unsqueeze(0)  # (1,4,4)
        aabb = aabb.to(dev, dtype=torch.float32)  # (2,3)

        # Make full-resolution pixel grid (u,v) with 0.5 center offset
        uu = torch.arange(W, device=dev, dtype=torch.float32)
        vv = torch.arange(H, device=dev, dtype=torch.float32)
        U, V = torch.meshgrid(uu, vv, indexing="xy")  # (H,W)
        pixels_uv = torch.stack([U.reshape(-1) + 0.5,
                                 V.reshape(-1) + 0.5], dim=-1)  # (H*W, 2)
        # (1, Nr, 2)
        pixels_uv = pixels_uv.unsqueeze(0)

        # Output buffers
        rgb_out = torch.zeros(H * W, 3, device=dev, dtype=torch.float32)
        depth_out = torch.zeros(H * W, 1, device=dev, dtype=torch.float32) if return_depth else None

        # Render in chunks to control memory
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            start = 0
            while start < H * W:
                end = min(start + chunk_rays, H * W)

                # Slice current chunk
                pix_chunk = pixels_uv[:, start:end, :]  # (1, n, 2)

                # Generate rays for this chunk
                rays = ray_generation(K, pose, pix_chunk, aabb)  # rays_o/d: (1,n,3)

                # Depth sampling (deterministic for smoother renders)
                depth = depth_sampling(
                    rays["t_near"], rays["t_far"],
                    num_rays=rays["rays_o"].shape[1],
                    num_samples_per_ray=self.num_samples_per_ray,
                    method=("deterministic" if deterministic_coarse else self.depth_sampler_method)
                )

                # Coarse pass
                x_enc = field_query(rays["rays_o"], rays["rays_d"], depth["t_vals"],
                                    self.l_dimensionality, self.concat_enc)
                V, R, N = 1, rays["rays_o"].shape[1], depth["t_vals"].shape[-1]
                sigma, rgb = self.NeRF(x_enc, V, R, N)
                sigma = model_stabilizer(sigma, noise=0.0)  # no training noise at render time
                vol = volume_renderer(sigma, rgb, depth["delta"], t_vals=depth["t_vals"],
                                      hit_max=rays["hit_mask"], bg_color=bgc)

                # Hierarchical fine pass
                t_fine = importance_resampler(vol["weights"], depth["t_vals"], self.num_fine_samples)
                delt = delta_recomputation(depth["t_vals"], t_fine)
                x_fine = field_query(rays["rays_o"], rays["rays_d"], delt["t_combined"],
                                     self.l_dimensionality, self.concat_enc)
                sigma_f, rgb_f = self.NeRF(x_fine, V, R, delt["t_combined"].shape[-1])
                sigma_f = model_stabilizer(sigma_f, noise=0.0)
                vol_f = volume_renderer(sigma_f, rgb_f, delt["delta_new"], t_vals=delt["t_combined"],
                                        hit_max=rays["hit_mask"], bg_color=bgc)

                # Write into buffers
                rgb_out[start:end, :] = vol_f["rgb_pred"].reshape(-1, 3)
                if return_depth:
                    # Expect depth in same units as t (ray param); OK for viz/relative depth.
                    depth_out[start:end, :] = vol_f["depth"].reshape(-1, 1)

                start = end

        # Reshape to images on CPU
        rgb_img = rgb_out.reshape(H, W, 3).clamp(0.0, 1.0).detach().cpu()
        if return_depth:
            depth_img = depth_out.reshape(H, W, 1).detach().cpu()
            return rgb_img, depth_img
        return rgb_img, None

    @torch.no_grad()
    def render_dataset_view(self, view_id: int,
                            chunk_rays: int = 8192,
                            deterministic_coarse: bool = True,
                            return_depth: bool = True):
        """
        Convenience: render using dataset intrinsics & a specific dataset pose.
        """
        K = self.ds.K
        pose = self.ds.frames[view_id]
        H = self.ds.height
        W = self.ds.width
        return self.render_view(K, pose, H, W,
                                aabb=self.ds.aabb,
                                chunk_rays=chunk_rays,
                                deterministic_coarse=deterministic_coarse,
                                return_depth=return_depth)

    @torch.no_grad()
    def render_path(self, Ks, poses, H, W,
                    aabb: torch.Tensor = None,
                    chunk_rays: int = 8192,
                    deterministic_coarse: bool = True,
                    return_depth: bool = False):
        """
        Render a sequence (useful for fly-throughs).
        Args:
            Ks:    (T, 3, 3) or (3,3) tensor
            poses: (T, 4, 4) tensor of c2w
        Returns:
            list of rgb (and optionally depth) tensors per frame
        """
        if isinstance(Ks, torch.Tensor) and Ks.ndim == 2:
            Ks = Ks.unsqueeze(0).expand(poses.shape[0], -1, -1)
        frames_rgb = []
        frames_depth = [] if return_depth else None
        for i in range(poses.shape[0]):
            rgb, d = self.render_view(
                K=Ks[i], pose=poses[i], H=H, W=W,
                aabb=(self.ds.aabb if aabb is None else aabb),
                chunk_rays=chunk_rays,
                deterministic_coarse=deterministic_coarse,
                return_depth=return_depth
            )
            frames_rgb.append(rgb)
            if return_depth: frames_depth.append(d)
        return (frames_rgb, frames_depth) if return_depth else (frames_rgb, None)

    # ------------------------------ Utils: preview saving ------------------------------

    def _make_run_dir(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("data", "rendered_imgs", f"{ts}_{self.object_id}_run")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    @staticmethod
    def _to_uint8(img: torch.Tensor):
        """img: (H,W,3) or (H,W,1) on CPU, float32 in [0,1]"""
        arr = np.clip(img.numpy(), 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
        return arr

    @staticmethod
    def _save_png_rgb(path: str, rgb_cpu: torch.Tensor):
        """rgb_cpu: (H,W,3) RGB float32 [0,1] on CPU"""
        arr = NerfWrapper._to_uint8(rgb_cpu)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, arr_bgr)

    @staticmethod
    def _save_png_depth(path: str, depth_cpu: torch.Tensor):
        """depth_cpu: (H,W,1) float32 (arbitrary scale). We min-max normalize per frame."""
        d = depth_cpu.numpy()
        dmin, dmax = float(d.min()), float(d.max())
        if dmax > dmin:
            dn = (d - dmin) / (dmax - dmin)
        else:
            dn = np.zeros_like(d)
        dn = (dn * 255.0 + 0.5).astype(np.uint8)  # grayscale
        dn = dn.squeeze(-1)  # (H,W)
        cv2.imwrite(path, dn)

    @staticmethod
    def _get_downscaled_K(K, downscale):
        if downscale <= 1:
            return K
        K_scaled = K.clone()
        K_scaled[:2, :] /= float(downscale)
        return K_scaled

    @torch.no_grad()
    def _render_and_save_preview(self,
                                 step: int,
                                 view_id: int,
                                 downscale: int = 2,
                                 save_depth: bool = True,
                                 chunk_rays: int = 8192):
        """
        Renders a dataset view and saves PNGs into self.run_dir.
        downscale: >1 to speed up previews.
        """
        # Prepare size
        H, W = self.ds.height, self.ds.width
        if downscale > 1:
            H = max(1, H // downscale)
            W = max(1, W // downscale)

        # Scale Intrinsics
        K_scaled = self._get_downscaled_K(self.ds.K, downscale)

        # Render
        rgb, depth = self.render_view(
            K=K_scaled,
            pose=self.ds.frames[view_id],
            H=H, W=W,
            aabb=self.ds.aabb,
            chunk_rays=chunk_rays,
            deterministic_coarse=True,
            return_depth=save_depth,
        )

        # Save
        rgb_path = os.path.join(self.run_dir, f"step_{step:07d}_view_{view_id:03d}_rgb.png")
        self._save_png_rgb(rgb_path, rgb.cpu())
        if save_depth and depth is not None:
            dep_path = os.path.join(self.run_dir, f"step_{step:07d}_view_{view_id:03d}_depth.png")
            self._save_png_depth(dep_path, depth.cpu())

    @torch.no_grad()
    def autotune_preview_chunk(self, start=65536, floor=8192, view_id=0, downscale=4):
        """Find the largest safe preview_chunk_rays via quick test render."""
        test_sizes = [start, start//2, start//3*2, 49152, 32768, 24576, 16384, 12288, floor]
        tried = set()
        
        # Prepare size
        H = max(1, self.ds.height // downscale)
        W = max(1, self.ds.width  // downscale)
        
        # Scale Intrinsics
        K_scaled = self._get_downscaled_K(self.ds.K, downscale)
        
        for s in test_sizes:
            s = int(max(floor, s))
            if s in tried:
                continue
            tried.add(s)
            try:
                # tiny render just to allocate the buffers
                self.render_view(
                    K=K_scaled,
                    pose=self.ds.frames[view_id],
                    H=H, W=W,
                    aabb=self.ds.aabb,
                    chunk_rays=s,
                    deterministic_coarse=True,
                    return_depth=False
                )
                print(f"[autotune] Using preview_chunk_rays={s}")
                return s
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        print(f"[autotune] Falling back to floor={floor}")
        return int(floor)
