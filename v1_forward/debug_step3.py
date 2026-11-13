# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.autotune(
    configs=[triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4)],
    key=['max_seqlen', 'num_heads', 'HEAD_DIM'],
)
@triton.jit
def flash_attn_fwd_varlen_debug(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    cu_seqlens_ptr,
    max_seqlen,
    num_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    head_idx = bh_idx % num_heads
    seq_idx = bh_idx // num_heads

    start_pos = tl.load(cu_seqlens_ptr + seq_idx)
    end_pos = tl.load(cu_seqlens_ptr + seq_idx + 1)
    seqlen = end_pos - start_pos

    q_offs = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + seq_idx * stride_qb + head_idx * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=q_offs[:, None] < seqlen, other=0.0)

    m = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    o = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    for block_n in range(0, (block_m_idx + 1) * BLOCK_M, BLOCK_N):
        n_offs = block_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + seq_idx * stride_kb + head_idx * stride_kh + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=n_offs[:, None] < seqlen, other=float('-inf'))

        v_ptrs = V_ptr + seq_idx * stride_vb + head_idx * stride_vh + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=n_offs[:, None] < seqlen, other=0.0)

        score = tl.dot(q, tl.trans(k)) * scale

        q_pos = q_offs[:, None]
        k_pos = n_offs[None, :]
        causal_mask = q_pos >= k_pos
        score = tl.where(causal_mask, score, float('-inf'))

        m_new = tl.maximum(m, tl.max(score, axis=-1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new[:, None])
        l = alpha * l + tl.sum(p, axis=1)
        o = alpha[:, None] * o + tl.dot(p.to(v.dtype), v)
        m = m_new

    o = o / l[:, None]

    o_ptrs = O_ptr + seq_idx * stride_ob + head_idx * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_ok
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=q_offs[:, None] < seqlen)


def flash_attn_triton_varlen_debug(Q, K, V, cu_seqlens):
    B, H, max_seqlen, D_head = Q.shape
    O = torch.zeros_like(Q)
    cu_seqlens = cu_seqlens.contiguous().to(Q.device)
    grid = lambda META: (triton.cdiv(max_seqlen, META['BLOCK_M']), B * H)
    flash_attn_fwd_varlen_debug[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        cu_seqlens,
        max_seqlen,
        num_heads=H,
        HEAD_DIM=D_head,
    )
    return O


def flash_attn_reference_varlen(Q, K, V, cu_seqlens):
    scale = Q.shape[-1] ** -0.5
    B, H, max_seqlen, D_head = Q.shape
    O = torch.zeros_like(Q)
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seqlen = end - start
        q = Q[i, :, :seqlen, :]
        k = K[i, :, :seqlen, :]
        v = V[i, :, :seqlen, :]
        mask = torch.triu(torch.ones(seqlen, seqlen, device=Q.device), diagonal=1).bool()
        score = torch.matmul(q, k.transpose(-2, -1)) * scale
        score = score.masked_fill(mask, float('-inf'))
        attn = torch.softmax(score, dim=-1)
        out = torch.matmul(attn, v)
        O[i, :, :seqlen, :] = out
    return O


if __name__ == "__main__":
    torch.manual_seed(42)
    D = 128
    H = 8
    D_head = D // H

    # Simple test: 2 batches, seqlen 8 and 4
    seqlens = [8, 4]
    B = len(seqlens)
    max_seqlen = max(seqlens)
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0)
    print(f"cu_seqlens: {cu_seqlens}")

    Q = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32) * 0.1
    K = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32) * 0.1
    V = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32) * 0.1

    ref = flash_attn_reference_varlen(Q, K, V, cu_seqlens)
    tri = flash_attn_triton_varlen_debug(Q, K, V, cu_seqlens)

    print(f"\nref[0,0,:8,:4] = \n{ref[0,0,:8,:4]}")
    print(f"\ntri[0,0,:8,:4] = \n{tri[0,0,:8,:4]}")
    print(f"\nref[1,0,:4,:4] = \n{ref[1,0,:8,:4]}")
    print(f"\ntri[1,0,:4,:4] = \n{tri[1,0,:8,:4]}")

    err = (ref - tri).abs().max().item()
    print(f"\nMax error: {err}")