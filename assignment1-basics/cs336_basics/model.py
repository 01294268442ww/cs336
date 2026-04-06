import torch
import torch.nn as nn
from .modules.transformer_block import TransformerBlock
from .modules.linear import Linear
from .modules.RMSNorm import RMSNorm
from .modules.utility import softmax
from .modules.embedding import Embedding

class Transformer(nn.Module):
    def __init__(
            self, 
            vocab_size, 
            context_length, 
            d_model, 
            num_layers, 
            num_heads, 
            d_ff, 
            theta, 
            use_rope = True, 
            use_moe = False, 
            num_experts = 2, 
            top_k = 2, 
            num_shared_experts = 1, 
            router_jitter = 0.0, 
            z_loss_coef = 1e-3, 
            lb_loss_coef = 1e-1, 
            device = None, 
            dtype = None
    ) -> None:
        super().__init__()
        """
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token
        embedding matrix.
        context_length: int The maximum context length, necessary for determining the dimensionality of
        the position embedding matrix.
        num_layers: int The number of Transformer blocks to use.
        """

        self.use_moe = use_moe

        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, 
                num_heads, 
                d_ff, 
                context_length, 
                theta, 
                mode= "groupQuery",
                use_rope=use_rope, 
                use_moe=use_moe, 
                num_experts=num_experts, 
                top_k=top_k, 
                num_shared_experts=num_shared_experts, 
                router_jitter=router_jitter, 
                z_loss_coef=z_loss_coef, 
                lb_loss_coef=lb_loss_coef, 
                device=device, 
                dtype=dtype
            ) for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        
        self.lm_head = Linear(d_model, vocab_size)

    
    def forward(self, x, token_positions=None):

        total_z_scaled = x.new_zeros(())
        tokens_per_expert_all = []
        total_lb_loss_scaled = x.new_zeros(())
        moe_layers = 0

        x = self.embedding(x)
        
        for block in self.blocks:
            x, aux = block(x, token_positions)
            if self.use_moe:
                total_z_scaled = total_z_scaled + aux["z_loss_scaled"]
                total_lb_loss_scaled = total_lb_loss_scaled + aux["lb_loss_scaled"]
                tokens_per_expert_all.append(aux["tokens_per_expert"])
                moe_layers += 1
        
        x = self.norm(x)

        logits = self.lm_head(x)

        aux_out = {
            "z_loss_scaled":total_z_scaled,
            "moe_layers":moe_layers,
            "token_per_expert":tokens_per_expert_all,
            "lb_loss_scaled":total_lb_loss_scaled
        }

        return logits, aux_out