import einops
import torch
import torch.nn as nn
from .utility import softmax

class ScaledDotProductAttention(nn.Module):
    """
    Implement the scaled dot-product attention function. Your implementation should
    handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
    dimensions (if provided). The implementation should return an output with the shape (batch_size,
    ..., d_v). See section 3.3 for a discussion on batch-like dimensions.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, Q, K, V, mask=None):
        
        d_k = Q.size(-1)
        attn = torch.einsum("... q d, ... k d ->... q k", Q, K)
        attn /= (d_k ** 0.5)

        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))

        attn = softmax(attn)
        attn = torch.einsum("... q k, ... k d -> ... q d", attn, V)

        return attn
    

class CausalMultiHeadAttention(nn.Module):
    def __init__(
            self, 
            d_model, 
            num_heads, 
            use_rope = False, 
            theta=1000.0, 
            max_seq_len = 2048,
            window_size = float("inf"),
            device=None, 
            dtype=None
    ) -> None:
        super().__init__()
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        """
        from cs336_basics.modules.linear import Linear
        from cs336_basics.modules.RoPE import RoPEEmbedding

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.window_size = window_size

        self.Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, d_model, device=device, dtype=dtype)

        self.attention = ScaledDotProductAttention()
        

        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPEEmbedding(
                theta,
                self.d_k,
                max_seq_len,
                device
            )

    def _create_sliding_window_mask(self, seq_len, device):
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)

        if self.window_size == float("inf"):
            mask = i >= j
        else:
            mask = (i >= j) & ((i - j) <= self.window_size)

        mask = ~mask

        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

    def forward(self, x, token_positions=None):

        seq_len = x.size(-2)

        causal_mask = self._create_sliding_window_mask(seq_len, x.device)
        
        q = einops.rearrange(self.Q(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = einops.rearrange(self.K(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = einops.rearrange(self.V(x), "b t (h d) -> b h t d", h=self.num_heads)

        if self.use_rope:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        
        attn = self.attention(q, k, v, causal_mask)

        attn = einops.rearrange(attn, "b h t d -> b t (h d)")

        output = self.linear(attn)

        return output
    

class MultiQueryAttention(nn.Module):
    """ 
    MQA have multiple queries，but just one dimension for keys and values
    """
    def __init__(
            self,
            d_model,
            num_heads,
            use_rope = False,
            theta = 1000.0,
            max_seq_len = 2048,
            window_size = float("inf"),
            device = None,
            dtype = None
    ) -> None:
        super().__init__()

        from cs336_basics.modules.linear import Linear
        from cs336_basics.modules.RoPE import RoPEEmbedding

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.window_size = window_size

        self.Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.K = Linear(d_model, self.d_k, device=device, dtype=dtype)
        self.V = Linear(d_model, self.d_v, device=device, dtype=dtype)
        self.linear = Linear(d_model, d_model, device=device, dtype=dtype)

        self.attention = ScaledDotProductAttention()

        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPEEmbedding(
                theta,
                self.d_k,
                max_seq_len,
                device
            )


    def _create_sliding_window_mask(self, seq_len, device):
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)

        if self.window_size == float("inf"):
            mask = i >= j
        else:
            mask = (i >= j) & ((i - j) <= self.window_size)

        mask = ~mask

        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
    

    def forward(self, x, token_positions=None):

        seq_len = x.size(-2)

        causal_mask = self._create_sliding_window_mask(seq_len, x.device)
        
        q = einops.rearrange(self.Q(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = einops.rearrange(self.K(x), "b t d -> b 1 t d")
        v = einops.rearrange(self.V(x), "b t d -> b 1 t d")

        if self.use_rope:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        
        attn = self.attention(q, k, v, causal_mask)

        attn = einops.rearrange(attn, "b h t d -> b t (h d)")

        output = self.linear(attn)

        return output


class GroupedQueryAttention(nn.Module):
    """
    GQA Grouped Query Attention (GQA) is an attention mechanism 
    that shares the same set of key/value heads among multiple query heads, 
    thereby significantly reducing the computation and caching overhead of key-value pairs 
    while achieving near-multi-head attention performance.
    """
    def __init__(
            self,
            d_model,
            num_heads,
            use_rope = False,
            theta = 1000.0,
            max_seq_len = 2048,
            group_size = 2,
            window_size = float("inf"),
            device = None,
            dtype = None
    ) -> None:
        super().__init__()

        from cs336_basics.modules.linear import Linear
        from cs336_basics.modules.RoPE import RoPEEmbedding

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % group_size == 0, "num_heads must be divisible by group_size"

        self.d_k = d_model // num_heads 
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_groups = num_heads // group_size
        self.window_size = window_size

        self.Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.K = Linear(d_model, self.num_groups * self.d_k, device=device, dtype=dtype)
        self.V = Linear(d_model, self.num_groups * self.d_v, device=device, dtype=dtype)
        self.linear = Linear(d_model, d_model, device=device, dtype=dtype)

        self.attention = ScaledDotProductAttention()

        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPEEmbedding(
                theta,
                self.d_k,
                max_seq_len,
                device
            )

    def _create_sliding_window_mask(self, seq_len, device):
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)

        if self.window_size == float("inf"):
            mask = i >= j
        else:
            mask = (i >= j) & ((i - j) <= self.window_size)

        mask = ~mask

        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
    

    def forward(self, x, token_positions=None):

        seq_len = x.size(-2)

        causal_mask = self._create_sliding_window_mask(seq_len, x.device)
        
        q = einops.rearrange(self.Q(x), "b t (g s d) -> b g s t d", g=self.num_groups, s=self.group_size)
        k = einops.rearrange(self.K(x), "b t (g d) -> b g 1 t d", g=self.num_groups)
        v = einops.rearrange(self.V(x), "b t (g d) -> b g 1 t d", g=self.num_groups)

        if self.use_rope:
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        
        attn = self.attention(q, k, v, causal_mask)

        attn = einops.rearrange(attn, "b g s t d -> b t (g s d)")

        output = self.linear(attn)

        return output