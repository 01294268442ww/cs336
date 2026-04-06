import torch
import torch.nn as nn
import torch.nn.functional as F
from cs336_basics.modules.linear import Linear

class Router(nn.Module):
    def __init__(self, d_model, num_experts, device=None, dtype=None) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.linear = Linear(d_model, num_experts, device=device, dtype=dtype)

    def forward(self, x):
        logits = self.linear(x)
        return logits


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None) -> None:
        super().__init__()

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def _silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        return self.W2(self._silu(self.W1(x)) * self.W3(x))
    

class MoE(nn.Module):
    def __init__(
            self, 
            d_model, 
            d_ff, 
            num_experts, 
            top_k = 2, 
            num_shared_experts = 1, 
            router_jitter = 0.0, 
            z_loss_coef = 1e-3, 
            lb_loss_coef = 1e-1, 
            device=None, dtype=None
        ) -> None:
        super().__init__()

        assert num_shared_experts < num_experts, "shared_expert must be less than num_experts" 


        self.num_experts = num_experts
        self.top_k = top_k
        self.router_jitter = router_jitter
        self.z_loss_coef = z_loss_coef
        self.lb_loss_coef = lb_loss_coef

        self.num_shared_experts = num_shared_experts

        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, device=device, dtype=dtype) for _ in range(num_experts)]
        )

        self.shared_experts = nn.ModuleList(
            [Expert(d_model, d_ff, device=device, dtype=dtype) for _ in range(num_shared_experts)]
        )
    
    @staticmethod
    def _z_loss(logits):
        log_sum_exp = torch.logsumexp(logits, dim=-1)
        z_loss = torch.mean(log_sum_exp**2)

        return z_loss
    
    @staticmethod
    def _load_balance_loss(router_probs, topk_indices, num_experts):
        # (batch_size, seq_len, num_experts / k)
        p = router_probs.mean(dim=(0, 1))

        dispatch = F.one_hot(topk_indices, num_classes=num_experts).to(router_probs.dtype)
        f = dispatch.mean(dim=(0, 1, 2))

        return num_experts * torch.sum(p * f)

    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        logits = self.router(x) # batch_size, seq_len, num_epxerts

        # avoid all token to one expert
        if self.router_jitter > 0.0 and self.training:
            noise = torch.randn_like(logits) * self.router_jitter
            logits = logits + noise
        
        z_loss = self._z_loss(logits)

        router_probs = torch.softmax(logits, dim=-1)

        topk_logits, topk_indices = torch.topk(
            logits,
            self.top_k,
            dim=-1
        )

        if self.top_k == 1:
            topk_gates = router_probs.gather(-1, topk_indices)
        else:
            topk_gates = torch.softmax(topk_logits, -1)
        
        lb_loss = self._load_balance_loss(router_probs, topk_indices, self.num_experts)
        # flatten x (batch_size seq_len) d_model
        x_flat = x.reshape(batch_size * seq_len, d_model)
        out_flat = x_flat.new_zeros((batch_size * seq_len, d_model))

        if self.top_k == 1:

            expert_ids = topk_indices.reshape(batch_size * seq_len)
            gate_flat = topk_gates.reshape(batch_size * seq_len)

            for e in range(self.num_experts):
                pos = (expert_ids == e).nonzero(as_tuple=False).squeeze(1)
                if pos.numel() == 0:
                    continue

                x_e = x_flat.index_select(0, pos)
                y_e = self.experts[e](x_e)
                y_e = y_e * gate_flat.index_select(0, pos).unsqueeze(1)

                out_flat.index_add_(0, pos, y_e)
            
            counts = torch.bincount(expert_ids, minlength=self.num_experts).to(x.dtype)
            tokens_per_expert = counts / (batch_size * seq_len)
        
        else:
            token_ids = (
                torch.arange(batch_size * seq_len, device=x.device)
                .unsqueeze(1)
                .expand(batch_size * seq_len, self.top_k)
                .reshape(-1)
            )

            expert_ids = topk_indices.reshape(-1)
            gate_flat = topk_gates.reshape(-1)

            for e in range(self.num_experts):
                sel = (expert_ids == e).nonzero(as_tuple=False).squeeze(1)
                if sel.numel() == 0:
                    continue

                tok = token_ids.index_select(0, sel)
                x_e = x_flat.index_select(0, tok)
                y_e = self.experts[e](x_e)
                y_e = y_e * gate_flat.index_select(0, sel).unsqueeze(1)
                # every token have k expert
                out_flat.index_add_(0, tok, y_e)

            counts = torch.bincount(expert_ids, minlength=self.num_experts).to(x.dtype)
            tokens_per_expert = counts / (batch_size * seq_len * self.top_k)

            
        expert_outputs = out_flat.view(batch_size, seq_len, d_model)

        # through shared experts
        for expert in self.shared_experts:
            expert_outputs = expert(expert_outputs)
        

        return expert_outputs, {
            "tokens_per_expert":tokens_per_expert,
            "z_loss":z_loss,
            "z_loss_scaled": z_loss * self.z_loss_coef,
            "lb_loss":lb_loss,
            "lb_loss_scaled":lb_loss * self.lb_loss_coef
        }