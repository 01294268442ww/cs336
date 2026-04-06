import torch.nn as nn
from .RMSNorm import RMSNorm
from .attention import CausalMultiHeadAttention, MultiQueryAttention, GroupedQueryAttention
from .FFN import FFN
from .MoE import MoE

class TransformerBlock(nn.Module):
    def __init__(
            self, 
            d_model, 
            num_heads, 
            d_ff, 
            max_seq_len, 
            theta, 
            use_rope = True, 
            use_moe = False, 
            num_experts = 2, 
            top_k = 2, 
            num_shared_experts = 1, 
            router_jitter = 0.0, 
            z_loss_coef = 1e-3, 
            lb_loss_coef = 1e-1, 
            mode = "causal",
            device = None, 
            dtype = None
    ) -> None:
        super().__init__()
        """
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        """

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_moe = use_moe
        
        if mode == "causal":
            self.mha = CausalMultiHeadAttention(
                d_model,
                num_heads, 
                use_rope=use_rope, 
                theta=theta, 
                max_seq_len=max_seq_len
            )
        elif mode == "multQuery":
            self.mha = MultiQueryAttention(
                d_model, 
                num_heads, 
                use_rope=use_rope, 
                theta=theta, 
                max_seq_len=max_seq_len
            )
        elif mode == "groupQuery":
            self.mha = GroupedQueryAttention(
                d_model,
                num_heads,
                use_rope=use_rope,
                theta=theta,
                max_seq_len=max_seq_len
            )
        
        
        if use_moe:
            self.ffn = MoE(
                d_model,
                d_ff,
                num_experts,
                top_k,
                num_shared_experts,
                router_jitter,
                z_loss_coef,
                lb_loss_coef,
                device,
                dtype
            )
        else:
            self.ffn = FFN(d_model, d_ff)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x, token_positions):

        x = x + self.mha(self.norm1(x), token_positions)
        aux = {}

        if self.use_moe:
            out, aux = self.ffn(self.norm2(x))
            x = x + out
        else:
            x = x + self.ffn(self.norm2(x))

        return x, aux
