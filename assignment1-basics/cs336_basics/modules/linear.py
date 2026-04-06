import torch
import torch.nn as nn


class Linear(nn.Module):
    """
        Implement a Linear class that inherits from torch.nn.Module and performs a linear transformation. 
        Your implementation should follow the interface of PyTorch’s built-in nn.Linear module, 
        except for not having a bias argument or parameter.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        """
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        self.W = nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        
        mean = 0.0
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=mean, std=std, a=-3 * std, b = 3 * std)
    
    def forward(self, x):
        """Apply the linear transformation to the input."""
        
        return x @ self.W