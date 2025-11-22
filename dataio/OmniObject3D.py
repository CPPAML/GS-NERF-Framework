from torch.utils.data import Dataset
from pathlib import Path
import json, os, torch, cv2, random, math


class OmniObject3D(Dataset):
    def __init__(self,
                 object_id: str,
                 views_per_batch: int = 24,
                 view_type: str = "uniform", #also contiguous and sliding
                 seed: int = 42,
                 ):
        self.view_type = view_type
        self.views_per_batch = views_per_batch
        self.file_path = Path(f"data/omniobject3d___OmniObject3D-New/raw/blender_renders_24_views/img/{object_id}")
        self.file_list = []
        self.transforms_json = json.load(open(os.path.join(self.file_path, "transforms.json")))
        self.aabb = torch.tensor(self.transforms_json["aabb"], dtype=torch.float32)
        self.camera_angle = self.transforms_json["camera_angle_x"]
        self.frames = self.transforms_json["frames"]
        self.transforms_json.clear()
        self._rng = random.Random(seed)

        self.temp_transforms = []
        for i in range(len(self.frames)):
            file_path = self.frames[i]["file_path"]
            transform_matrix = torch.tensor(self.frames[i]["transform_matrix"], dtype=torch.float32)
            self.temp_transforms.append((file_path, transform_matrix))
            self.file_list.append(os.path.join(self.file_path, self.frames[i]["file_path"]))
        self.frames = self.temp_transforms

        img = cv2.imread(self.file_list[0])
        self.height, self.width, self.color = img.shape

        fx = 0.5 * self.width / math.tan(0.5 * self.camera_angle)
        fy = fx * (self.height / self.width)
        cx, cy = self.width / 2, self.height / 2
        self.K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

        self.imgs = []
        for i in range(len(self.file_list)):
            bgr_img = cv2.imread(self.file_list[i])
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            normalized_img = torch.tensor(rgb_img, dtype=torch.float32) / 255.0
            self.imgs.append(normalized_img)
        self.indexes = list(range(len(self.imgs)))

        self.idxs = []
        if self.view_type == "uniform":
            perm = self.indexes.copy()
            random.shuffle(perm)
            self.idxs = [perm[i : i + self.views_per_batch] for i in range(0, len(perm), self.views_per_batch)]
        elif self.view_type == "contiguous":
            self.idxs = [self.indexes[i : i+ self.views_per_batch] for i in range(0, len(self.indexes), self.views_per_batch)]
        elif self.view_type == "sliding":
            self.idxs = [self.indexes[i : i + self.views_per_batch] for i in range(0, len(self.indexes) - self.views_per_batch + 1)]
        else:
            raise ValueError(f"Unknown view_type: {self.view_type}, please use one of 'uniform', 'contiguous' or 'sliding'.")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        num = self.idxs[idx]
        images = torch.stack([self.imgs[i] for i in num], 0)
        poses = torch.stack([self.frames[i][1] for i in num], 0)
        Ks = self.K.expand(len(num), 3, 3)
        return {
            "images": images,
            "Ks": Ks,
            "poses": poses,
            "view_ids": torch.tensor(num, dtype=torch.int64),
            "aabb": self.aabb,
        }

    def reshuffle(self):
        if self.view_type == "uniform":
            perm = self.indexes.copy()
            self._rng.shuffle(perm)
            self.idxs = [perm[i: i + self.views_per_batch] for i in range(0, len(perm), self.views_per_batch)]
            print("Reshuffled!")
        else:
            print("Reshuffling is only supported for 'uniform' view_type.")