from functools import partial
import torch
from nerf_wrapper import NerfWrapper

# AdamW with weight decay (the wrapper will pass lr)
opt_factory = partial(torch.optim.AdamW, weight_decay=1e-2, betas=(0.9, 0.99))

wrapper = NerfWrapper(
    object_id="anise_001",            # object id
    views_per_batch=8,                # fewer views per step -> more rays per view budget
    num_rays=1024,                    # rays per view (effective batch = 8*1024)
    num_samples_per_ray=64,           # coarse samples
    num_fine_samples=64,              # fine samples
    ray_sampler_method="uniform",
    depth_sampler_method="stratified",
    use_bgcolor=False,
    num_hidden_layers=8,              # classic NeRF depth
    hidden_dim_width=256,             # classic NeRF width
    optimizer_factory=opt_factory,    # (lr set below in train call)
    learning_rate=3e-4,               # works well w/ AdamW + AMP
    l_dimensionality=10,              # standard NeRF positional freq
    concat_enc=True,
    use_amp=True,                     # mixed precision on CUDA
    seed=42,
    compile=False,                    # uses torch.compile if available.
    dataset_split_strategy="random",
    val_ratio=0.15,
    view_type="uniform",
)

best_chunk = wrapper.autotune_preview_chunk(start=65536)
print(f"Found value for best chunk: {best_chunk}")
# Training schedule:
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
    save_depth_preview=True,
)
