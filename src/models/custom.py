import torch
from torch import nn
from torch_geometric.nn import MessagePassing


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
    def weights_init(layer: torch.nn.Module) -> None:
        """
        Initialize model weights

        Args:
            layer (torch.nn.Module): Model layer
        """
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)
        elif isinstance(layer, nn.LSTM):
            WeightInitializableModule.init_named_params(layer)
        elif isinstance(layer, nn.ModuleList):
            for module in layer:
                if isinstance(layer, MessagePassing):
                    WeightInitializableModule.init_named_params(layer)
        
    @staticmethod
    def init_named_params(layer: torch.nn.Module) -> None:
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            

class GaussianNoise(nn.Module):
    """
    Gaussian noise regularizer.
    source: https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/3

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = torch.zeros_like(x).normal_() * scale
            x = x + sampled_noise
        return x
