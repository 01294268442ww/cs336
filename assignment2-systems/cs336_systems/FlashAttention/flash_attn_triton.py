import torch
import triton
import math
import triton.language as tl
from einops import rearrange, einsum

# @triton.autotune(
#     [
#         triton.Config(
#             {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
#             num_stages=num_stages,
#             num_warps=num_warps,
#         )
#         for BLOCK_SIZE_Q in [64, 128]
#         for BLOCK_SIZE_KV in [32, 64]
#         for num_stages in ([3, 4, 7])
#         for num_warps in [2, 4]
#     ],
#     key=["SEQ_LEN", "HEAD_DIM"],
# )

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    scale,
    BLOCK_SIZE_Q : tl.constexpr,
    BLOCK_SIZE_KV : tl.constexpr,
    STAGE : tl.constexpr,
    offs_q : tl.constexpr,
    offs_kv : tl.constexpr,
    SEQ_LEN : tl.constexpr
):
    if STAGE == 1:
        # 左边K全部可见
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # 部分可见 因果attention
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # 非因果
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) # K是转置
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        K_block = tl.load(K_block_ptr)
        Qk_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            Qk_block = Qk_block * scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(Qk_block, 1))
            Qk_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(Qk_block, 1) * scale)
            Qk_block = Qk_block * scale - m_ij[:, None]
        
        P_block = tl.math.exp(Qk_block)
        l_ij = tl.sum(P_block, 1)

        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        # P_block = P_block.to(tl.float16)

        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    
    return O_block, l_i, m_i

@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)

    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS

    qkv_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    Q_block_ptr = tl.make_block_ptr(
        base= Q + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0) 
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        base= K + qkv_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1)
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0)
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    Q_block = tl.load(Q_block_ptr)

    if STAGE == 1 or STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN
        )
    if STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN
        )
    
    m_i += tl.math.log(l_i)

    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, is_causal : bool = False):
        have_head = q.size() == 4
        if have_head:
            B, H, T, D = q.size()
        else:
            # add num_head
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
            B, H, T, D = q.size()
        
        O = torch.empty_like(q)
        stage = 3 if is_causal else 1

        args = {
            "BLOCK_SIZE_Q" : 64,
            "BLOCK_SIZE_KV": 32
        }

        grid =  (
            triton.cdiv(T, args["BLOCK_SIZE_Q"]),
            B * H,
            1
        )

        scale = 1.0 / math.sqrt(D)

        M = torch.empty(
            (B, H, T), device=q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=q,
            K=k,
            V=v,
            scale=scale,
            M=M,
            O=O,
            stride_Q_batch=q.stride(0),
            stride_Q_head=q.stride(1),
            stride_Q_seq=q.stride(2),
            stride_Q_dim=q.stride(3),
            stride_K_batch=k.stride(0),
            stride_K_head=k.stride(1),
            stride_K_seq=k.stride(2),
            stride_K_dim=k.stride(3),
            stride_V_batch=v.stride(0),
            stride_V_head=v.stride(1),
            stride_V_seq=v.stride(2),
            stride_V_dim=v.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=B,
            NUM_HEADS=H,
            SEQ_LEN=T,
            HEAD_DIM=D,
            STAGE=stage,
            BLOCK_SIZE_Q=args["BLOCK_SIZE_Q"],
            BLOCK_SIZE_KV=args["BLOCK_SIZE_KV"]
        )

        if not have_head:
            O = O.squeeze(1)
            M = M.squeeze(1)

        ctx.save_for_backward(q, k, v, O, M)
        ctx.grid = grid
        ctx.scale = scale
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