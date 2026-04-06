import einops
import torch
import torch.nn as nn

class RoPEEmbedding(nn.Module):
    """
    Implement a class RotaryPositionalEmbedding that applies RoPE to the input tensor.
    """
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        """
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k))

        self.register_buffer("inv_freq", inv_freq, persistent=False)


    def _rotate_half(self, x):
        x = einops.rearrange(x, "... (d j)-> ... d j", j=2)
        x1, x2 = x.unbind(dim=-1)
        return einops.rearrange(torch.stack((-x2, x1), dim=-1), "... d j-> ...(d j)")
        
    
    def forward(self, x, token_positions):
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        Note that you should tolerate x with an arbitrary number of batch dimensions. 
        You should assume that the token positions are a tensor of shape (..., seq_len) 
        specifying the token positions of x along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.
        """

        if token_positions is None:
            seq_len = x.size(-2)
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(0)
        
        theta = torch.einsum("...i, j -> ... i j", token_positions, self.inv_freq)
        cos = torch.cos(theta).repeat_interleave(2, dim=-1)
        sin = torch.sin(theta).repeat_interleave(2, dim=-1)

        x_rotated = (x * cos) + (self._rotate_half(x) * sin)

        return x_rotated