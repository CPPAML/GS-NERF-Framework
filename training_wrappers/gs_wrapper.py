import os
import datetime
from typing import Optional

import torch
import torch.nn.functional as F
import cv2

from core.base import TrainerBase
from dataio.OmniObject3D import OmniObject3D
from fields.gaussian_field import GaussianField
from renderers.gs_render import (
    gs_ray_sampler,
    geom_proj,
    gaussian_proj,
    gaussian_eval,
    gaussian_renderer,
    init_gaussians_from_aabb,
)


class GsWrapper(TrainerBase):
    """
    Minimal Gaussian Splatting training wrapper, mirroring the NeRF wrapper structure.
    Ties together dataset, field (parameters), and rendering functions.
    """

    def __init__(
        self,
        object_id: str,
        views_per_batch: int = 24,
        num_rays: int = 1024,
        num_gaussians: int = 50_000,
        optimizer_factory=torch.optim.AdamW,
        learning_rate: float = 1e-2,
        use_bgcolor: bool = True,
        use_amp: bool = True,
        seed: int = 42,
        dataset_split_strategy: str = "random",
        val_ratio: float = 0.15,
        view_type: str = "uniform",
    ) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GsWrapper] Device: {self.device}  Torch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")

        # Config
        self.object_id = object_id
        self.views_per_batch = int(views_per_batch)
        self.num_rays = int(num_rays)
        self.num_gaussians = int(num_gaussians)
        self.learning_rate = float(learning_rate)
        self.use_bgcolor = bool(use_bgcolor)
        self.use_amp = bool(use_amp and torch.cuda.is_available())
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.view_type = view_type

        # Seeding & perf
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Dataset
        self.ds = OmniObject3D(
            object_id=self.object_id,
            views_per_batch=self.views_per_batch,
            view_type=self.view_type,
            seed=self.seed,
            val_ratio=self.val_ratio,
            split_strategy=dataset_split_strategy,
        )

        # Initialize Gaussian parameters from scene AABB
        init = init_gaussians_from_aabb(self.ds.aabb, self.num_gaussians, device=self.device)
        # Map to expected ctor names (l_world vs L_world)
        self.field = GaussianField(
            mu_world=init["mu_world"],
            l_world=init["L_world"],
            color=init["color"],
            opacity_logits=init["opacity_logits"],
        ).to(self.device)

        # Optimizer & AMP
        self.optimizer = optimizer_factory(self.field.parameters(), lr=self.learning_rate)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Background color (RGB)
        self.bgcolor = torch.ones(3, dtype=torch.float32, device=self.device) if self.use_bgcolor else None

        self.run_dir: Optional[str] = None

    # -------------------------- Public API --------------------------
    def train(
        self,
        iters: int = 10000,
        log_every: int = 50,
        eval_every: int = 500,
        ckpt_every: int = 0,
        clip_grad: Optional[float] = 1.0,
        preview_every: int = 500,
        preview_downscale: int = 2,
        preview_chunk_rays: int = 8192,
        preview_view: int = 0,
    ):
        self.field.train()
        self.run_dir = self.run_dir or self._make_run_dir()

        print(f"[GsWrapper] Starting training for {iters} iterations...")
        for step, batch in enumerate(self.ds):
            if step >= iters:
                break

            # Move batch tensors (except images handled separately)
            Ks = batch["Ks"].to(self.device)
            poses = batch["poses"].to(self.device)
            aabb = batch["aabb"].to(self.device)

            # Sample rays and targets
            images = batch["images"].to(self.device)
            ray_s = gs_ray_sampler(images, self.num_rays)
            pixels_uv = ray_s["pixels_uv"].to(self.device)
            pixels_gt = ray_s["pixels_gt"].to(self.device)

            # Handle RGBA targets (composite over bg if requested)
            if pixels_gt.shape[-1] == 4:
                rgb_gt = pixels_gt[..., :3]
                alpha_gt = pixels_gt[..., 3:]
                if self.bgcolor is not None:
                    rgb_gt = rgb_gt * alpha_gt + self.bgcolor * (1.0 - alpha_gt)
            else:
                rgb_gt = pixels_gt

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                # Project Gaussians to views
                geo = geom_proj(Ks, poses, self.field.mu_world)  # pixel_coords (B,N,2), depth (B,N)
                gpr = gaussian_proj(self.field.l_world, poses, Ks, geo["x_cam"])  # sigma_2d (B,N,2,2)

                # Evaluate alpha at sampled rays
                eval_out = gaussian_eval(pixels_uv, gpr["sigma_2d"], geo["pixel_coords"], self.field.opacity())
                # Expand per-Gaussian color to batch dimension expected by renderer (B,N,3)
                color_b = self.field.color.unsqueeze(0).expand(Ks.shape[0], -1, -1)
                rend = gaussian_renderer(eval_out["alpha"], color_b, geo["depth"], bg_color=self.bgcolor)

                rgb_pred = rend["rgb_pred"]  # (B,R,3)
                loss = F.mse_loss(rgb_pred, rgb_gt)
                psnr = self._psnr_from_mse(loss)

            if not torch.isfinite(loss):
                print("[GsWrapper] Non-finite loss encountered; stopping.")
                break

            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if clip_grad is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.field.parameters(), clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.field.parameters(), clip_grad)
                self.optimizer.step()

            # Logs
            if (step + 1) % log_every == 0 or step == 0:
                print(f"[GsWrapper][{step + 1}] train_loss={loss.item():.4f}  psnr={psnr.item():.2f}dB")

            # Periodic preview
            if preview_every and (step + 1) % preview_every == 0:
                self._render_and_save_preview(
                    step=step + 1,
                    view_id=preview_view,
                    downscale=preview_downscale,
                    chunk_rays=preview_chunk_rays,
                )

            # Periodic eval (PSNR over a few val views)
            if eval_every and (step + 1) % eval_every == 0:
                with torch.no_grad():
                    mean_psnr = self.evaluate(max_views=2, num_batches=1)
                    print(f"[GsWrapper][{step + 1}] val_psnr={mean_psnr:.2f}dB")

            # Optional checkpoints (state dict only)
            if ckpt_every and (step + 1) % ckpt_every == 0:
                self._save_checkpoint(step + 1)

    @torch.no_grad()
    def evaluate(self, max_views: int = 4, num_batches: int = 1) -> float:
        self.field.eval()
        psnrs = []
        for batch in self.ds.val_iter(max_views=max_views, num_batches=num_batches) or []:
            Ks = batch["Ks"].to(self.device)
            poses = batch["poses"].to(self.device)
            images = batch["images"].to(self.device)

            B, H, W, C = images.shape
            assert B >= 1
            for b in range(B):
                pred = self.render_view(
                    Ks[b:b+1], poses[b:b+1],
                    H=H, W=W,
                    downscale=2,
                    chunk_rays=16384,
                )  # (H', W', 3)
                # Downscale GT to match preview output
                Hs, Ws = pred.shape[:2]
                gt = images[b].detach().cpu().numpy()
                if gt.shape[2] == 4:
                    gt = gt[..., :3]
                gt_u8 = (gt * 255.0).astype('uint8')
                gt_small = cv2.resize(gt_u8, (Ws, Hs), interpolation=cv2.INTER_AREA)
                gt_small_t = torch.from_numpy(gt_small).to(torch.float32).to(self.device) / 255.0
                pred_t = torch.from_numpy(pred).to(self.device) / 255.0
                mse = F.mse_loss(pred_t, gt_small_t)
                psnrs.append(self._psnr_from_mse(mse).item())
        self.field.train()
        return float(sum(psnrs) / max(1, len(psnrs)))

    # -------------------------- Utilities --------------------------
    def _make_run_dir(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("runs", "gs", f"{self.object_id}_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"[GsWrapper] Run directory: {run_dir}")
        return run_dir

    @staticmethod
    def _psnr_from_mse(mse: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return -10.0 * torch.log10(torch.clamp(mse, min=eps))

    @torch.no_grad()
    def render_view(
        self,
        Ks: torch.Tensor,           # (1,3,3)
        poses: torch.Tensor,        # (1,4,4)
        H: int,
        W: int,
        downscale: int = 2,
        chunk_rays: int = 8192,
    ) -> "np.ndarray":
        """Render a single view with chunked ray evaluation. Returns uint8 RGB array."""
        import numpy as np

        self.field.eval()

        Hs = max(1, H // downscale)
        Ws = max(1, W // downscale)

        # Pixel centers in original resolution coordinates
        u = (torch.arange(Ws, device=self.device, dtype=torch.float32) + 0.5) * downscale
        v = (torch.arange(Hs, device=self.device, dtype=torch.float32) + 0.5) * downscale
        uu, vv = torch.meshgrid(u, v, indexing='xy')
        uv = torch.stack([uu, vv], dim=-1).view(1, -1, 2)  # (1, Hs*Ws, 2)

        # Precompute per-view projections
        geo = geom_proj(Ks, poses, self.field.mu_world)
        gpr = gaussian_proj(self.field.l_world, poses, Ks, geo["x_cam"])  # (1,N,2,2)

        rgb_out = torch.zeros(uv.shape[1], 3, device=self.device, dtype=torch.float32)
        for start in range(0, uv.shape[1], chunk_rays):
            end = min(start + chunk_rays, uv.shape[1])
            rays = uv[:, start:end, :]  # (1,R,2)
            eval_out = gaussian_eval(rays, gpr["sigma_2d"], geo["pixel_coords"], self.field.opacity())
            color_b = self.field.color.unsqueeze(0)  # (1,N,3)
            rend = gaussian_renderer(eval_out["alpha"], color_b, geo["depth"], bg_color=self.bgcolor)
            rgb_out[start:end, :] = rend["rgb_pred"].squeeze(0)

        img = rgb_out.view(Ws, Hs, 3).permute(1, 0, 2).clamp(0.0, 1.0)  # (Hs, Ws, 3)
        img_np = (img.detach().cpu().numpy() * 255.0).astype(np.uint8)
        self.field.train()
        return img_np

    @torch.no_grad()
    def _render_and_save_preview(
        self,
        step: int,
        view_id: int = 0,
        downscale: int = 2,
        chunk_rays: int = 8192,
    ) -> None:
        # Prepare single-view inputs
        ids = [int(view_id)]
        batch = self.ds._pack_ids(ids)
        Ks = batch["Ks"][0:1].to(self.device)
        poses = batch["poses"][0:1].to(self.device)
        H, W = batch["images"].shape[1:3]

        img = self.render_view(Ks, poses, H, W, downscale=downscale, chunk_rays=chunk_rays)
        # Save PNG (convert RGB->BGR for OpenCV)
        out_dir = os.path.join(self.run_dir, "previews")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"step_{step:06d}.png")
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, bgr)
        print(f"[GsWrapper] Saved preview to {out_path}")

    def _save_checkpoint(self, step: int) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, f"ckpt_{step:06d}.pt")
        torch.save({
            "step": step,
            "state_dict": self.field.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"[GsWrapper] Saved checkpoint: {path}")
