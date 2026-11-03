import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    dO_ptr, dQ_ptr,
    partial_dK_ptr, partial_dV_ptr,
    M_ptr, L_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_dqb, stride_dqh, stride_dqm, stride_dqk,
    stride_dkb, stride_dkh, stride_dkn, stride_dkk,
    stride_dvb, stride_dvh, stride_dvn, stride_dvk,
    stride_mb, stride_mh, stride_ms,
    stride_lb, stride_lh, stride_ls,
    cu_seqlens_ptr,
    num_q_blocks,
    num_kv_blocks,
    seqlen,
    num_heads,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    start_pos = tl.load(cu_seqlens_ptr + batch_idx)
    end_pos = tl.load(cu_seqlens_ptr + batch_idx + 1)
    seq_len = end_pos - start_pos

    if block_m_idx * BLOCK_M >= seq_len:
        return

    q_offs = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q_ptr + batch_idx * stride_qb + head_idx * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qk,
        mask=q_offs[:, None] < seq_len, other=0.0
    )

    do = tl.load(
        dO_ptr + batch_idx * stride_dob + head_idx * stride_doh + q_offs[:, None] * stride_dom + d_offs[None, :] * stride_dok,
        mask=q_offs[:, None] < seq_len, other=0.0
    )

    o_loaded = tl.load(
        O_ptr + batch_idx * stride_ob + head_idx * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_ok,
        mask=q_offs[:, None] < seq_len, other=0.0
    )

    Li = tl.load(
        L_ptr + batch_idx * stride_lb + head_idx * stride_lh + q_offs * stride_ls,
        mask=q_offs < seq_len, other=1.0
    )
    Li = tl.where(Li == 0.0, 1.0, Li)

    Mi = tl.load(
        M_ptr + batch_idx * stride_mb + head_idx * stride_mh + q_offs * stride_ms,
        mask=q_offs < seq_len, other=float('-inf')
    )

    Di = tl.sum(do * o_loaded, axis=1)

    q_f16 = q.to(tl.float16)
    do_f16 = do.to(tl.float16)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    bni = 0
    for block_n in range(0, seq_len, BLOCK_N):
        n_offs = block_n + tl.arange(0, BLOCK_N)
        kv_mask = n_offs[:, None] < seq_len
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

        causal_mask = n_offs[None, :] <= q_offs[:, None]
        s = tl.where(causal_mask, s, float('-inf'))

        p = tl.exp(s - Mi[:, None]) / Li[:, None]

        dv = tl.dot(tl.trans(p).to(tl.float16), do_f16)
        p_dv_ptr = partial_dV_ptr + pid_bh * (num_q_blocks * num_kv_blocks * BLOCK_N * HEAD_DIM) \
                   + block_m_idx * (num_kv_blocks * BLOCK_N * HEAD_DIM) \
                   + bni * (BLOCK_N * HEAD_DIM) \
                   + local_n[:, None] * HEAD_DIM + d_offs[None, :]
        tl.store(p_dv_ptr, dv.to(tl.float32), mask=kv_mask)

        dp = tl.dot(do_f16, tl.trans(v_f16))
        ds = p * (dp - Di[:, None])

        dq += tl.dot(ds.to(tl.float16), k_f16)

        dk = tl.dot(tl.trans(ds).to(tl.float16), q_f16)
        p_dk_ptr = partial_dK_ptr + pid_bh * (num_q_blocks * num_kv_blocks * BLOCK_N * HEAD_DIM) \
                   + block_m_idx * (num_kv_blocks * BLOCK_N * HEAD_DIM) \
                   + bni * (BLOCK_N * HEAD_DIM) \
                   + local_n[:, None] * HEAD_DIM + d_offs[None, :]
        tl.store(p_dk_ptr, dk.to(tl.float32), mask=kv_mask)

        bni += 1

    dq_ptrs = dQ_ptr + batch_idx * stride_dqb + head_idx * stride_dqh + q_offs[:, None] * stride_dqm + d_offs[None, :] * stride_dqk
    tl.store(dq_ptrs, dq.to(dQ_ptr.dtype.element_ty), mask=q_offs[:, None] < seq_len)


@triton.jit
def _reduce_kernel(
    partial_ptr, out_ptr,
    num_q_blocks, num_kv_blocks,
    seqlen, num_heads,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
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


def flash_attn_backward(Q, K, V, O, dO, M, L, cu_seqlens):
    B, H, max_seqlen, D_head = Q.shape
    cu_seqlens = cu_seqlens.contiguous().to(Q.device)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    if D_head <= 32:
        BM, BN, NW = 32, 32, 2
    else:
        BM, BN, NW = 64, 32, 4

    num_q_blocks = triton.cdiv(max_seqlen, BM)
    num_kv_blocks = triton.cdiv(max_seqlen, BN)
    BH = B * H

    partial_dK = torch.empty(BH, num_q_blocks, num_kv_blocks, BN, D_head, device='cuda', dtype=torch.float32)
    partial_dV = torch.empty(BH, num_q_blocks, num_kv_blocks, BN, D_head, device='cuda', dtype=torch.float32)

    grid1 = (num_q_blocks, BH)
    _flash_attn_bwd_kernel[grid1](
        Q, K, V, O, dO, dQ,
        partial_dK, partial_dV,
        M, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        cu_seqlens,
        num_q_blocks,
        num_kv_blocks,
        max_seqlen, H,
        BLOCK_M=BM, BLOCK_N=BN, HEAD_DIM=D_head, num_warps=NW, num_stages=2,
    )

    grid2 = (num_kv_blocks, BH)
    _reduce_kernel[grid2](
        partial_dK, dK,
        num_q_blocks, num_kv_blocks,
        max_seqlen, H,
        BLOCK_N=BN, HEAD_DIM=D_head, num_warps=2, num_stages=1,
    )
    _reduce_kernel[grid2](
        partial_dV, dV,
        num_q_blocks, num_kv_blocks,
        max_seqlen, H,
        BLOCK_N=BN, HEAD_DIM=D_head, num_warps=2, num_stages=1,
    )

    return dQ, dK, dV


def flash_attn_reference_backward(Q, K, V, dO, cu_seqlens):
    scale = Q.shape[-1] ** -0.5
    B, H, max_seqlen, D_head = Q.shape
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        seq_len = end - start

        q = Q[i, :, :seq_len, :]
        k = K[i, :, :seq_len, :]
        v = V[i, :, :seq_len, :]
        do = dO[i, :, :seq_len, :]

        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        score = torch.matmul(q, k.transpose(-2, -1)) * scale
        score = score.masked_fill(mask, float('-inf'))
        attn = torch.softmax(score, dim=-1)

        dv = torch.matmul(attn.transpose(-2, -1), do)
        dp = torch.matmul(do, v.transpose(-2, -1))
        ds = attn * (dp - (dp * attn).sum(dim=-1, keepdim=True))
        dq = torch.matmul(ds, k)
        dk = torch.matmul(ds.transpose(-2, -1), q)

        dQ[i, :, :seq_len, :] = dq
        dK[i, :, :seq_len, :] = dk
        dV[i, :, :seq_len, :] = dv

    return dQ, dK, dV


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=== FlashAttention Backward (Causal + VarLen + SplitKV) ===")
    print()

    import sys
    sys.path.insert(0, r'd:\AI HPC\FlashAttention')
    from v1_forward.step3_varlen import flash_attn_triton_varlen

    H, D_head = 8, 16

    for seqlens in [[128, 256], [64, 128, 192], [256, 128, 64]]:
        B = len(seqlens)
        max_seqlen = max(seqlens)
        cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0)

        Q = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32)
        dO = torch.randn_like(Q)

        with torch.no_grad():
            tri_O, tri_M, tri_L = flash_attn_triton_varlen(Q, K, V, cu_seqlens)
            ref_dQ, ref_dK, ref_dV = flash_attn_reference_backward(Q, K, V, dO, cu_seqlens)
            tri_dQ, tri_dK, tri_dV = flash_attn_backward(Q, K, V, tri_O, dO, tri_M, tri_L, cu_seqlens)

            valid_mask = torch.zeros_like(Q, dtype=torch.bool)
            for i in range(B):
                valid_mask[i, :, :seqlens[i], :] = True

            err_dQ = (ref_dQ[valid_mask] - tri_dQ[valid_mask]).abs().max().item()
            err_dK = (ref_dK[valid_mask] - tri_dK[valid_mask]).abs().max().item()
            err_dV = (ref_dV[valid_mask] - tri_dV[valid_mask]).abs().max().item()

        status = "✅" if max(err_dQ, err_dK, err_dV) < 1e-1 else "❌"
        print(f"seqlens={seqlens} | dQ={err_dQ:.2e} dK={err_dK:.2e} dV={err_dV:.2e} | {status}")