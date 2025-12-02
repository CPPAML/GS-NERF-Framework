from abc import ABC, abstractmethod
import torch.nn as nn


class FieldBase(nn.Module):
    """Base class for scene fields (e.g., NeRF MLP, Gaussian parameter field)."""
    def __init__(self):
        super().__init__()


class RendererBase(ABC):
    """Base class for renderers. Functionality is provided via free functions in this project,
    but this abstraction reserves a place for future OO renderers."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class TrainerBase(ABC):
    """Base class for training wrappers."""

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
