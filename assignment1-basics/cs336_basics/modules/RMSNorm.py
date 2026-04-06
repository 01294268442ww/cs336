import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Implement RMSNorm as a torch.nn.Module."""
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None) -> None:
        super().__init__()
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """

        self.g = nn.Parameter(torch.empty(d_model, device=device, dtype=torch.float32))
        self.eps = eps
    
    def forward(self, x):
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
            and return a tensor of the same shape.
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x * self.g / rms

        return result.to(in_dtype)
