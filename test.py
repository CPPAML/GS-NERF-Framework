from OmniObject3D import OmniObject3D
from torch.utils.data import DataLoader

ds = OmniObject3D(object_id="Category_Object", views_per_batch=4)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)