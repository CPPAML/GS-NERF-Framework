import torch
from core.base import FieldBase
from NeRF.neural_architectures import GaussianNetwork as _GaussianImpl


class GaussianField(_GaussianImpl, FieldBase):
    """Adapter around parameter set of 3D Gaussians to fit the Field abstraction."""
    def opacity(self):
        return torch.sigmoid(self.opacity_logits)
