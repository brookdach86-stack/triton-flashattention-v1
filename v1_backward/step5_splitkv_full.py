import torch
import triton
import triton.language as tl
import torch.nn.functional as F

'''
┌─────────────────────────────────────────────────────────┐
│ FlashAttention Backward Kernel                          │
│                                                         │
│ "以算代存"：Forward 扔了 P (N×N 注意力矩阵)               │
│ Backward 现场重算 P，省显存、带宽换计算                   │
│                                                         │
│ Forward 存的：O (输出), M (每行 max), L (每行 softmax 分母)│
│ Backward 丢的：P (现在重算)                              │
├─────────────────────────────────────────────────────────┤
│ 段 1: 加载 Q_block, dO_block, O_block, M_block, L_block │
│ 段 2: Di = rowsum(dO * O)，初始化 dq=0, scale=1/√D      │
│ 段 3: for block_n (KV block loop):                      │
│       k, v = load(K,V)                                  │
│       score = Q @ K^T * scale                           │  ← 重算 S
│       p = exp(score - M) / L                            │  ← 重算 P (用 M+L)
│                                                         │
│       [5 步菜谱]                                         │
│       ① dV = P^T @ dO  → atomic_add                    │
│       ② dP = dO @ V^T                                   │
│       ③ dS = P * (dP - Di)                              │
│       ④ dQ += dS @ K                                    │
│       ⑤ dK = dS^T @ Q → atomic_add                     │
│                                                         │
│ 段 4: store(dQ)                                         │
└─────────────────────────────────────────────────────────┘
'''


@triton.jit
def _flash_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    dO_ptr,
    partial_dQ_ptr, partial_dK_ptr, partial_dV_ptr,
    M_ptr, L_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_mb, stride_mh, stride_ms,
    stride_lb, stride_lh, stride_ls,
    num_q_blocks,
    num_kv_blocks,
    seqlen,
    num_heads,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    block_n_idx = tl.program_id(1)
    pid_bh = tl.program_id(2)
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    q_offs = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q_ptr + batch_idx * stride_qb + head_idx * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qk,
        mask=q_offs[:, None] < seqlen, other=0.0
    )

    do = tl.load(
        dO_ptr + batch_idx * stride_dob + head_idx * stride_doh + q_offs[:, None] * stride_dom + d_offs[None, :] * stride_dok,
        mask=q_offs[:, None] < seqlen, other=0.0
    )

    o_loaded = tl.load(
        O_ptr + batch_idx * stride_ob + head_idx * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_ok,
        mask=q_offs[:, None] < seqlen, other=0.0
    )

    Li = tl.load(
        L_ptr + batch_idx * stride_lb + head_idx * stride_lh + q_offs * stride_ls,
        mask=q_offs < seqlen, other=1.0
    )
    Li = tl.where(Li == 0.0, 1.0, Li)

    Mi = tl.load(
        M_ptr + batch_idx * stride_mb + head_idx * stride_mh + q_offs * stride_ms,
        mask=q_offs < seqlen, other=float('-inf')
    )

    Di = tl.sum(do * o_loaded, axis=1)

    q_f16 = q.to(tl.float16)
    do_f16 = do.to(tl.float16)

    scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    n_offs = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    kv_mask = n_offs[:, None] < seqlen
    local_n = tl.arange(0, BLOCK_N)

    k = tl.load(
        K_ptr + batch_idx * stride_kb + head_idx * stride_kh + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kk,
        mask=kv_mask, other=0.0
    )
    v = tl.load(
        V_ptr + batch_idx * stride_vb + head_idx * stride_vh + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vk,
        mask=kv_mask, other=0.0
    )

    k_f16 = k.to(tl.float16)
    v_f16 = v.to(tl.float16)

    s = tl.dot(q_f16, tl.trans(k_f16)) * scale

    p = tl.exp(s - Mi[:, None]) / Li[:, None]

    dv = tl.dot(tl.trans(p).to(tl.float16), do_f16)

    dp = tl.dot(do_f16, tl.trans(v_f16))
    ds = p * (dp - Di[:, None])

    dq_partial = tl.dot(ds.to(tl.float16), k_f16)
    dk_partial = tl.dot(tl.trans(ds).to(tl.float16), q_f16)

    # partial_dQ: (BH, num_q_blocks, num_kv_blocks, BLOCK_M, HEAD_DIM)
    p_dq_ptr = partial_dQ_ptr + pid_bh * (num_q_blocks * num_kv_blocks * BLOCK_M * HEAD_DIM) \
               + block_m_idx * (num_kv_blocks * BLOCK_M * HEAD_DIM) \
               + block_n_idx * (BLOCK_M * HEAD_DIM) \
               + tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM + d_offs[None, :]
    tl.store(p_dq_ptr, dq_partial.to(tl.float32), mask=q_offs[:, None] < seqlen)

    # partial_dK: (BH, num_q_blocks, num_kv_blocks, BLOCK_N, HEAD_DIM)
    p_dk_ptr = partial_dK_ptr + pid_bh * (num_q_blocks * num_kv_blocks * BLOCK_N * HEAD_DIM) \
               + block_m_idx * (num_kv_blocks * BLOCK_N * HEAD_DIM) \
               + block_n_idx * (BLOCK_N * HEAD_DIM) \
               + local_n[:, None] * HEAD_DIM + d_offs[None, :]
    tl.store(p_dk_ptr, dk_partial.to(tl.float32), mask=kv_mask)

    # partial_dV: (BH, num_q_blocks, num_kv_blocks, BLOCK_N, HEAD_DIM)
    p_dv_ptr = partial_dV_ptr + pid_bh * (num_q_blocks * num_kv_blocks * BLOCK_N * HEAD_DIM) \
               + block_m_idx * (num_kv_blocks * BLOCK_N * HEAD_DIM) \
               + block_n_idx * (BLOCK_N * HEAD_DIM) \
               + local_n[:, None] * HEAD_DIM + d_offs[None, :]
    tl.store(p_dv_ptr, dv.to(tl.float32), mask=kv_mask)


@triton.jit
def _reduce_kernel(
    partial_ptr, out_ptr,
    num_q_blocks, num_kv_blocks,
    seqlen, num_heads,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # grid: (num_kv_blocks, B*H)
    block_n_idx = tl.program_id(0)
    pid_bh = tl.program_id(1)
    n_offs = block_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    local_n = tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, HEAD_DIM)

    acc = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    for q in range(num_q_blocks):
        ptr = partial_ptr + pid_bh * (num_q_blocks * num_kv_blocks * BLOCK_N * HEAD_DIM) \
               + q * (num_kv_blocks * BLOCK_N * HEAD_DIM) \
               + block_n_idx * (BLOCK_N * HEAD_DIM) \
               + local_n[:, None] * HEAD_DIM + d_offs[None, :]
        acc += tl.load(ptr, mask=n_offs[:, None] < seqlen, other=0.0)

    out_ptrs = out_ptr + pid_bh * (seqlen * HEAD_DIM) + n_offs[:, None] * HEAD_DIM + d_offs[None, :]
    tl.store(out_ptrs, acc, mask=n_offs[:, None] < seqlen)


@triton.jit
def _reduce_dq_kernel(
    partial_ptr, out_ptr,
    num_q_blocks, num_kv_blocks,
    seqlen, num_heads,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    pid_bh = tl.program_id(1)
    q_offs = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for k in range(num_kv_blocks):
        ptr = partial_ptr + pid_bh * (num_q_blocks * num_kv_blocks * BLOCK_M * HEAD_DIM) \
               + block_m_idx * (num_kv_blocks * BLOCK_M * HEAD_DIM) \
               + k * (BLOCK_M * HEAD_DIM) \
               + tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM + d_offs[None, :]
        acc += tl.load(ptr, mask=q_offs[:, None] < seqlen, other=0.0)

    out_ptrs = out_ptr + pid_bh * (seqlen * HEAD_DIM) + q_offs[:, None] * HEAD_DIM + d_offs[None, :]
    tl.store(out_ptrs, acc, mask=q_offs[:, None] < seqlen)


def flash_attn_backward(Q, K, V, O, dO, M, L):
    B, H, S, D_head = Q.shape

    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    if D_head <= 32:
        BM, BN, NW = 32, 32, 2
    else:
        BM, BN, NW = 64, 32, 4

    num_q_blocks = triton.cdiv(S, BM)
    num_kv_blocks = triton.cdiv(S, BN)
    BH = B * H

    partial_dQ = torch.empty(BH, num_q_blocks, num_kv_blocks, BM, D_head, device='cuda', dtype=torch.float32)
    partial_dK = torch.empty(BH, num_q_blocks, num_kv_blocks, BN, D_head, device='cuda', dtype=torch.float32)
    partial_dV = torch.empty(BH, num_q_blocks, num_kv_blocks, BN, D_head, device='cuda', dtype=torch.float32)

    grid1 = (num_q_blocks, num_kv_blocks, BH)
    _flash_attn_bwd_kernel[grid1](
        Q, K, V, O, dO,
        partial_dQ, partial_dK, partial_dV,
        M, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        num_q_blocks,
        num_kv_blocks,
        S, H,
        BLOCK_M=BM, BLOCK_N=BN, HEAD_DIM=D_head, num_warps=NW, num_stages=2,
    )

    grid_q = (num_q_blocks, BH)
    _reduce_dq_kernel[grid_q](
        partial_dQ, dQ,
        num_q_blocks, num_kv_blocks,
        S, H,
        BLOCK_M=BM, HEAD_DIM=D_head, num_warps=2, num_stages=1,
    )

    grid_kv = (num_kv_blocks, BH)
    _reduce_kernel[grid_kv](
        partial_dK, dK,
        num_q_blocks, num_kv_blocks,
        S, H,
        BLOCK_N=BN, HEAD_DIM=D_head, num_warps=2, num_stages=1,
    )
    _reduce_kernel[grid_kv](
        partial_dV, dV,
        num_q_blocks, num_kv_blocks,
        S, H,
        BLOCK_N=BN, HEAD_DIM=D_head, num_warps=2, num_stages=1,
    )

    return dQ, dK, dV


def flash_attn_reference_backward(Q, K, V, dO):
    scale = Q.shape[-1] ** -0.5
    score = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(score, dim=-1)
    O = torch.matmul(attn, V)

    dV = torch.matmul(attn.transpose(-2, -1), dO)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dS = attn * (dP - (dP * attn).sum(dim=-1, keepdim=True))
    dQ = torch.matmul(dS, K)
    dK = torch.matmul(dS.transpose(-2, -1), Q)

    return dQ, dK, dV


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=== FlashAttention Backward (Multi-Head) ===")
    print("验证反向传播梯度: dQ, dK, dV")
    print()

    for S in [128, 256, 512]:
        B, H, D_head = 2, 8, 16
        Q = torch.randn(B, H, S, D_head, device='cuda', dtype=torch.float32, requires_grad=True)
        K = torch.randn(B, H, S, D_head, device='cuda', dtype=torch.float32, requires_grad=True)
        V = torch.randn(B, H, S, D_head, device='cuda', dtype=torch.float32, requires_grad=True)

        dO = torch.randn_like(Q)

        ref_dQ, ref_dK, ref_dV = flash_attn_reference_backward(Q, K, V, dO)

        with torch.no_grad():
            scale = D_head ** -0.5
            score = torch.matmul(Q, K.transpose(-2, -1)) * scale
            M_val = score.max(dim=-1)[0]
            exp_score = torch.exp(score - M_val.unsqueeze(-1))
            L_val = exp_score.sum(dim=-1)
            p = exp_score / L_val.unsqueeze(-1)
            O_val = torch.matmul(p, V)

            tri_dQ, tri_dK, tri_dV = flash_attn_backward(Q, K, V, O_val, dO, M_val, L_val)

        err_dQ = (ref_dQ - tri_dQ).abs().max().item()
        err_dK = (ref_dK - tri_dK).abs().max().item()
        err_dV = (ref_dV - tri_dV).abs().max().item()
        status = "✅" if max(err_dQ, err_dK, err_dV) < 5e-2 else "❌"

        print(f"S={S:4d} | dQ err={err_dQ:.2e} | dK err={err_dK:.2e} | dV err={err_dV:.2e} | {status}")