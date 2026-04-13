import torch
import math
from einops import rearrange, einsum


class PytorchFlashAttention(torch.autograd.Function):

    @staticmethod # forward backward 没有共享变量，所以必须通过 ctx 来传递信息。
    def forward(ctx, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, is_causal : bool = False):
        have_head = q.size() == 4
        if have_head:
            B, H, T, D = q.size()
            q = rearrange(q, "b h t d -> (b h) t d")
            k = rearrange(k, "b h t d -> (b h) t d")
            v = rearrange(v, "b h t d -> (b h) t d")
            Bh =  B * H
        else:
            Bh, T, D = q.size()

        device = q.device
        scale = 1.0 / math.sqrt(D)

        Bc, Br = 64, 64
        Tr = math.ceil(T / Br)
        Tc = math.ceil(T / Bc)

        O = torch.zeros((Bh, T, D), device=device, dtype=torch.float32)
        L = torch.zeros((Bh, T, 1), device=device, dtype=torch.float32)

        for i in range(Tr):
            q_start, q_end = i * Br, min((i + 1) * Br, T)

            Q_i = q[:, q_start:q_end, :]
            Tr_i = Q_i.size(1)

            m_prev = torch.full((Bh, Tr_i), -float("inf"), device=device, dtype=torch.float32)
            l_prev = q.new_zeros((Bh, Tr_i), dtype=torch.float32)
            o_prev = q.new_zeros((Bh, Tr_i, D), dtype=torch.float32)

            start_j = 0
            if is_causal:
                start_j = i

            for j in range(start_j, Tc):
                kv_start, kv_end = j * Bc, min((j + 1) * Bc, T)

                K_j = k[:, kv_start:kv_end, :]
                V_j = v[:, kv_start:kv_end, :]
                
                S_ij = torch.einsum("b i d, b j d  -> b i j", Q_i, K_j) * scale

                if is_causal:
                    mask = torch.ones(Tr_i, K_j.size(1), device=device, dtype=torch.bool).triu(diagonal=1)
                    S_ij = S_ij.masked_fill(mask, -float("inf"))

                
                m_curr = torch.max(S_ij, dim=-1).values
                m_new = torch.max(m_prev, m_curr)
                exp_old = torch.exp(m_prev - m_new)
                P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))
                l_curr = exp_old * l_prev + P_ij.sum(dim=-1)
                o_prev = (exp_old.unsqueeze(-1) * o_prev) + torch.einsum("b i j, b j d -> b i d", P_ij, V_j)  

                m_prev, l_prev = m_new, l_curr
            
            O[:, q_start:q_end] = o_prev / l_prev.unsqueeze(-1)
            L[:, q_start:q_end] = (m_prev + torch.log(l_prev)).unsqueeze(-1)

        if have_head:
            O = rearrange(O, "(b h) t d -> b h t d", b=B, h=H)
            L = rearrange(L, "(b h) t 1 -> b h t 1", b=B, h=H)

        ctx.save_for_backward(q, k, v, O, L.squeeze(-1))
        ctx.is_causal = is_causal
        ctx.have_head = have_head
        ctx.B = B if have_head else None
        ctx.H = H if have_head else None
        
        return O
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, O, L = ctx.saved_tensors
        L = L.unsqueeze(-1)
        is_causal = ctx.is_causal
        have_head = ctx.have_head

        device = q.device
        scale = 1.0 / math.sqrt(q.size(-1))

        if have_head:
            B, H, T, D = ctx.B, ctx.H, q.size(1), q.size(2)
            q = rearrange(q, "b h t d -> (b h) t d")
            k = rearrange(k, "b h t d -> (b h) t d")
            v = rearrange(v, "b h t d -> (b h) t d")
            do = rearrange(do, "b h t d -> (b h) t d")
            O = rearrange(O, "b h t d -> (b h) t d")
            L = rearrange(L, "b h t 1 -> (b h) t 1")
            Bh = B * H
        else:
            Bh, T, D = q.size()
        
        Br, Bc = 64, 64
        Tr = math.ceil(T / Br)
        Tc = math.ceil(T / Bc)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        for j in range(Tc):
            kv_start, kv_end = j * Bc, min((j+1) * Bc, T)

            K_j = k[:, kv_start:kv_end, :]
            v_j = v[:, kv_start:kv_end, :]
            dk_j = torch.zeros_like(K_j)
            dv_j = torch.zeros_like(v_j)

            start_i = 0
            if is_causal:
                start_i = j
            
            for i in range(start_i, Tr):
                q_start, q_end = i * Br, min((i + 1) * Br, T)
                Q_i = q[:, q_start:q_end, :]
                O_i = O[:, q_start:q_end, :]
                do_i = do[:, q_start:q_end, :]
                L_i = L[:, q_start:q_end, :]

                S_ij = torch.einsum("b r d, b c d -> b r c", Q_i, K_j) * scale

                if is_causal:
                    mask = torch.ones(Q_i.size(1), K_j.size(1), device=device).triu(1)
                    S_ij = S_ij.masked_fill(mask, -float("inf"))

                P_ij = torch.exp(S_ij - L_i) # b r c
                dv_ij = torch.einsum("b r c, b r d -> b c d", P_ij, do_i)
                dp_ij = torch.einsum("b r d, b c d -> b r c", do_i, v_j)
                D_i =  torch.sum(do_i * O_i, dim=-1, keepdim=True)
                ds_ij = P_ij * (dp_ij - D_i)

                dq_ij = torch.einsum("b r c, b c d -> b r d", ds_ij, K_j) * scale
                dk_ij = torch.einsum("b r c, b r d -> b c d", ds_ij, Q_i) * scale

                dq[:, q_start:q_end, :] += dq_ij
                dk_j += dk_ij
                dv_j += dv_ij

            dk[:, kv_start:kv_end, :] = dk_j
            dv[:, kv_start:kv_end, :] = dv_j
    
        if have_head:
            dq = rearrange(dq, "(b h) t d -> b h t d", b=B, h=H)
            dk = rearrange(dk, "(b h) t d -> b h t d", b=B, h=H)
            dv = rearrange(dv, "(b h) t d -> b h t d", b=B, h=H)
        
        return dq, dk, dv, None