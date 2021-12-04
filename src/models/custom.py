import torch
from torch import nn


class BaseModule(nn.Module):
    
    def __init(self):
        super().__init__()
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        raise NotImplementedError
    
    def param_count(self) -> int:
        """Parameter counter"""
        return sum(param.numel() for param in self.parameters())
    
    def reset(self) -> None:
        """Reset weights"""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def __str__(self):
        raise NotImplementedError

    
class DenseModule(BaseModule):

    def __init(self):
        super().__init__()
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        raise NotImplementedError
    
    def is_dense(self):
        return True


class SparseModule(BaseModule):

    def __init(self):
        super().__init__()
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        raise NotImplementedError
    
    def is_dense(self):
        return False
    
    
class WeightInitializableModule(nn.Module):
    
    @staticmethod
    def weights_init(layer: torch.tensor) -> None:
        """
        Initialize model weights

        Args:
            layer (torch.tensor): Model layer
        """
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)
