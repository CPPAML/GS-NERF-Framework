from functools import partial
import torch

# Robust import: allow running from project root or with training_wrappers on PYTHONPATH
try:
    from gs_wrapper import GsWrapper  # running with working dir set to training_wrappers
except ModuleNotFoundError:
    from training_wrappers.gs_wrapper import GsWrapper  # standard project-root import


# AdamW with weight decay (the wrapper will pass lr)
opt_factory = partial(torch.optim.AdamW, weight_decay=1e-2, betas=(0.9, 0.99))


# Instantiate GS training wrapper
wrapper = GsWrapper(
    object_id="anise_001",          # object id in OmniObject3D dataset
    views_per_batch=8,                # number of views per training step
    num_rays=1024,                    # rays per view (effective batch = 8*1024)
    num_gaussians=50_000,             # number of 3D Gaussians
    optimizer_factory=opt_factory,    # (lr set below in train call)
    learning_rate=1e-2,               # default works decently for GS params
    use_amp=torch.cuda.is_available(),# enable AMP when on CUDA
    seed=42,
    dataset_split_strategy="random",
    val_ratio=0.15,
    view_type="uniform",
)


# Pick a reasonable preview chunk size (tuned for CUDA vs CPU)
best_chunk = 16384 if torch.cuda.is_available() else 4096
print(f"Using preview chunk rays: {best_chunk} (cuda={torch.cuda.is_available()})")


# Training schedule (mirrors nerf test structure)
wrapper.train(
    iters=80_000,
    log_every=50,
    eval_every=1_000,
    ckpt_every=5_000,
    clip_grad=20.0,

    # Previews (PNG snapshots)
    preview_every=50,
    preview_downscale=1,
    preview_chunk_rays=best_chunk,
    preview_view=0,
)
