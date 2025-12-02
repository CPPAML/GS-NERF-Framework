import torch.nn as nn
from core.base import FieldBase
from NeRF.neural_architectures import NeRF as _NeRFImpl


class NerfField(_NeRFImpl, FieldBase):
    """Thin adapter around the existing NeRF network to fit Field abstraction."""
    pass
