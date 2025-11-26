from torch.utils.data import Dataset
from pathlib import Path
import json, os, torch, cv2, math, random
from typing import Iterator, Dict, Any, List, Tuple


class OmniObject3D(Dataset):
    """
    Iterable multi-view dataset with train/val split.
    - __iter__() -> infinite stream of TRAIN batches
    - val_iter(max_views, num_batches=1) -> small deterministic VAL iterator
    - reshuffle() affects TRAIN only (when view_type='uniform')
    Assumes camera-to-world (c2w) transforms in transforms.json.
    """
    def __init__(self,
                 object_id: str,
                 views_per_batch: int = 24,
                 view_type: str = "uniform",     # 'uniform' | 'contiguous' | 'sliding'
                 seed: int = 42,
                 val_ratio: float = 0.1,         # fraction of frames reserved for validation
                 split_strategy: str = "random"  # 'random' | 'tail'
                 ):
        super().__init__()
        self.view_type = view_type
        self.views_per_batch = int(views_per_batch)
        self._rng = random.Random(seed)
        self._seed = seed
        self.val_ratio = float(val_ratio)
        self.split_strategy = split_strategy

        # ---------- Paths / JSON ----------
        self.root = Path("data/omniobject3d___OmniObject3D-New/raw/blender_renders_24_views/img") / object_id
        tf_path = self.root / "transforms.json"
        if not tf_path.exists():
            raise FileNotFoundError(f"transforms.json not found at: {tf_path}")

        with open(tf_path, "r") as f:
            meta = json.load(f)

        self.aabb = torch.tensor(meta.get("aabb", [[-1,-1,-1],[1,1,1]]), dtype=torch.float32)  # (2,3)
        cam_angle_x = float(meta["camera_angle_x"])  # radians
        frames = meta["frames"]
        del meta

        # ---------- Files & poses ----------
        self.file_list: List[Path] = []
        self.frames: List[torch.Tensor] = []
        for fr in frames:
            img_path = self.root / fr["file_path"]
            self.file_list.append(img_path)
            T = torch.tensor(fr["transform_matrix"], dtype=torch.float32)  # (4,4) c2w
            
            # Blender (OpenGL) to OpenCV conversion
            # Blender: X Right, Y Up, Z Back (Look -Z)
            # OpenCV: X Right, Y Down, Z Forward (Look +Z)
            T[:3, 1:3] *= -1
            
            self.frames.append(T)

        if len(self.file_list) == 0:
            raise RuntimeError(f"No frames found in: {self.root}")

        # ---------- Read first image to get H,W ----------
        probe = cv2.imread(str(self.file_list[0]), cv2.IMREAD_UNCHANGED)
        if probe is None:
            raise RuntimeError(f"Failed to read example image: {self.file_list[0]}")
        if probe.ndim == 2:
            probe = cv2.cvtColor(probe, cv2.COLOR_GRAY2RGB)
        elif probe.shape[2] == 4:
            probe = cv2.cvtColor(probe, cv2.COLOR_BGRA2BGR)
        self.height, self.width = probe.shape[:2]

        # ---------- Intrinsics from FOVx + aspect ----------
        fx = 0.5 * self.width / math.tan(0.5 * cam_angle_x)
        fovy = 2.0 * math.atan((self.height / self.width) * math.tan(0.5 * cam_angle_x))
        fy = 0.5 * self.height / math.tan(0.5 * fovy)
        cx, cy = self.width * 0.5, self.height * 0.5
        self.K = torch.tensor([[fx, 0.0, cx],
                               [0.0, fy, cy],
                               [0.0, 0.0, 1.0]], dtype=torch.float32)  # (3,3)

        # ---------- Load all images (RGB or RGBA, float32 in [0,1]) ----------
        self.imgs: List[torch.Tensor] = []
        for p in self.file_list:
            img_cv = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                raise RuntimeError(f"Failed to read image: {p}")
            
            if img_cv.ndim == 2:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
            
            if img_cv.shape[2] == 4:
                # BGRA -> RGBA
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA)
            else:
                # BGR -> RGB
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            img = torch.from_numpy(img_cv).to(torch.float32) / 255.0
            self.imgs.append(img)

        self.indexes = list(range(len(self.imgs)))

        # ---------- Split train/val at frame level ----------
        self.train_ids, self.val_ids = self._split_indices(self.indexes, self.val_ratio, self.split_strategy)

        # ---------- Build grouped batches for train/val ----------
        self.train_groups: List[List[int]] = self._make_index_groups(self.train_ids, self.view_type)
        # Validation is kept deterministic & simple: always contiguous groups
        self.val_groups: List[List[int]] = self._make_index_groups(self.val_ids, "contiguous") if len(self.val_ids) else []

        # Pointers for iteration
        self._train_cursor = 0

    # ------------------- Split helpers -------------------

    def _split_indices(self, ids: List[int], val_ratio: float, strategy: str) -> Tuple[List[int], List[int]]:
        n = len(ids)
        n_val = max(1, int(round(n * val_ratio))) if 0.0 < val_ratio < 1.0 else int(val_ratio)
        n_val = max(0, min(n_val, n))

        if n_val == 0:
            return ids, []

        if strategy == "random":
            perm = ids.copy()
            self._rng.shuffle(perm)
            val_ids = sorted(perm[:n_val])
            train_ids = sorted(perm[n_val:])
        elif strategy == "tail":
            val_ids = list(range(n - n_val, n))
            train_ids = list(range(0, n - n_val))
        else:
            raise ValueError(f"Unknown split_strategy: {strategy}. Use 'random' or 'tail'.")

        # If views_per_batch > len(train_ids) or len(val_ids), warn but proceed
        if len(train_ids) < self.views_per_batch:
            print(f"[OmniObject3D] Warning: train frames ({len(train_ids)}) < views_per_batch ({self.views_per_batch}).")
        if len(val_ids) and len(val_ids) < self.views_per_batch and self.view_type == "sliding":
            # sliding requires at least V frames to make one group
            print(f"[OmniObject3D] Warning: val frames ({len(val_ids)}) < views_per_batch ({self.views_per_batch}) for sliding; "
                  f"validation will use smaller groups.")

        return train_ids, val_ids

    # ------------------- Grouping helpers -------------------

    def _make_index_groups(self, pool: List[int], view_type: str) -> List[List[int]]:
        if len(pool) == 0:
            return []

        V = self.views_per_batch
        if view_type == "uniform":
            perm = pool.copy()
            self._rng.shuffle(perm)
            return [perm[i:i + V] for i in range(0, len(perm), V)]
        elif view_type == "contiguous":
            return [pool[i:i + V] for i in range(0, len(pool), V)]
        elif view_type == "sliding":
            if V > len(pool):
                # Produce a single (smaller) group as best-effort
                return [pool.copy()]
            return [pool[i:i + V] for i in range(0, len(pool) - V + 1)]
        else:
            raise ValueError(f"Unknown view_type: {view_type}. Use 'uniform', 'contiguous', or 'sliding'.")

    # ------------------- Standard Dataset API ---------------------

    def __len__(self) -> int:
        # Length of the TRAIN grouping (for epoch sizing)
        return len(self.train_groups)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ids = self.train_groups[idx]
        return self._pack_ids(ids)

    # ------------------- Iterators -------------------

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Infinite iterator over TRAIN batches.
        'uniform': reshuffle TRAIN groups at epoch wrap.
        Other view_types: cycle deterministically.
        """
        while True:
            if len(self.train_groups) == 0:
                raise RuntimeError("No train groups available. Check split configuration.")
            if self._train_cursor >= len(self.train_groups):
                self._train_cursor = 0
                if self.view_type == "uniform":
                    # reshuffle training only
                    self.train_groups = self._make_index_groups(self.train_ids, "uniform")
            idx = self._train_cursor
            self._train_cursor += 1
            yield self._pack_ids(self.train_groups[idx])

    def val_iter(self, max_views: int = 4, num_batches: int = 1) -> Iterator[Dict[str, Any]]:
        """
        Deterministic iterator over up to 'num_batches' VAL batches.
        Validation groups are contiguous and stable (no reshuffle).
        """
        if len(self.val_groups) == 0:
            return
        count = 0
        for g in self.val_groups:
            ids = g[:max_views] if max_views < len(g) else g
            yield self._pack_ids(ids)
            count += 1
            if count >= num_batches:
                break

    # ------------------- Utilities -------------------

    def _pack_ids(self, ids: List[int]) -> Dict[str, Any]:
        images = torch.stack([self.imgs[i] for i in ids], dim=0)      # (V,H,W,3)
        poses  = torch.stack([self.frames[i] for i in ids], dim=0)    # (V,4,4) c2w
        Ks     = self.K.expand(len(ids), 3, 3).contiguous()           # (V,3,3)
        return {
            "images": images,
            "Ks": Ks,
            "poses": poses,
            "view_ids": torch.tensor(ids, dtype=torch.int64),
            "aabb": self.aabb,
        }

    def reshuffle(self) -> None:
        """
        Explicit reshuffle for TRAIN groups when view_type='uniform'.
        Validation remains deterministic.
        """
        if self.view_type == "uniform":
            self.train_groups = self._make_index_groups(self.train_ids, "uniform")
            self._train_cursor = 0
            print("Reshuffled train groups!")
        else:
            print("Reshuffling is only supported for 'uniform' view_type.")

    # -------------- Info / diagnostics --------------

    def epoch_size(self) -> int:
        return len(self.train_groups)

    @property
    def aspect(self) -> float:
        return float(self.width) / float(self.height)

    def split_sizes(self) -> Tuple[int, int]:
        """Returns (#train_frames, #val_frames)."""
        return len(self.train_ids), len(self.val_ids)
