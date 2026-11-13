# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
import torch
import triton
import triton.language as tl
import torch.nn.functional as F


'''
FlashAttention Forward Kernel (Causal + 跳块优化)
┌─────────────────────────────────────────────────────┐
│ 准备阶段:                                            │
│   1. program_id → block_m_idx, batch_idx            │
│   2. 加载 Q_block [BLOCK_M, D] 到 SRAM              │
│   3. 初始化 m=-inf, l=0, o=zeros, scale=1/√D        │
├─────────────────────────────────────────────────────┤
│ 循环阶段: for block_n in range(0, (block_m_idx+1)*BLOCK_M, BLOCK_N): │
│                                                     │
│   ★ 跳块优化: 只遍历 Q_block 能看到的 K block        │
│                                                     │
│   4. 加载 K_block [BLOCK_N, D]                      │
│   5. 加载 V_block [BLOCK_N, D]                      │
│   6. score = Q @ K^T * scale  [BLOCK_M, BLOCK_N]    │
│   7. causal_mask 屏蔽对角线 K block 内部无效位置     │
│   8. online_softmax 更新:                           │
│      m_new = max(m, max(score, axis=1))             │
│      α = exp(m - m_new)                             │
│      p = exp(score - m_new[:, None])                │
│      l = α * l + sum(p, axis=1)                    │
│      o = α[:,None] * o + dot(p, v)                 │
│      m = m_new                                     │
├─────────────────────────────────────────────────────┤
│ 收尾阶段:                                            │
│   9. o = o / l[:, None]  (归一化)                   │
│  10. tl.store(O_ptr, o)  (写回 HBM)                  │
└─────────────────────────────────────────────────────┘

每次迭代:

  HBM ──load──→ SRAM           SRAM 内计算           ──store──→ HBM
  ┌───┐         ┌───┐                                  ┌───┐
  │K,V│ ──────→ │k,v│ ─→ score=Q@K^T ─→ softmax ─→ o+=P@V │ ──→ │O │
  └───┘         └───┘                                  └───┘
                     ↑ q 一直在 SRAM 不动!
'''


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=4, num_warps=8),
    ],
    key=['seqlen', 'HEAD_DIM'],
)
@triton.jit
def flash_attn_fwd_causal(
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr, L_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_mb, stride_mh, stride_ms,
    stride_lb, stride_lh, stride_ls,
    seqlen,
    num_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    block_m_idx = tl.program_id(0)
    bh_idx = tl.program_id(1)
    head_idx = bh_idx % num_heads
    batch_idx = bh_idx // num_heads

    q_offs = block_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh + q_offs[:, None] * stride_qm + d_offs[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=q_offs[:, None] < seqlen, other=0.0)

    m = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    o = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    for block_n in range(0, (block_m_idx + 1) * BLOCK_M, BLOCK_N):
        n_offs = block_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + batch_idx * stride_kb + head_idx * stride_kh + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=n_offs[:, None] < seqlen, other=float('-inf'))

        v_ptrs = V_ptr + batch_idx * stride_vb + head_idx * stride_vh + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vk
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

    o_ptrs = O_ptr + batch_idx * stride_ob + head_idx * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_ok
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=q_offs[:, None] < seqlen)

    m_ptrs = M_ptr + batch_idx * stride_mb + head_idx * stride_mh + q_offs * stride_ms
    tl.store(m_ptrs, m, mask=q_offs < seqlen)

    l_ptrs = L_ptr + batch_idx * stride_lb + head_idx * stride_lh + q_offs * stride_ls
    tl.store(l_ptrs, l, mask=q_offs < seqlen)


def flash_attn_triton_causal(Q, K, V, need_aux=True):
    B, H, S, D_head = Q.shape
    O = torch.empty_like(Q)
    M = torch.empty(B, H, S, device=Q.device, dtype=torch.float32)
    L = torch.empty(B, H, S, device=Q.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(S, META['BLOCK_M']), B * H)
    flash_attn_fwd_causal[grid](
        Q, K, V, O, M, L,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        M.stride(0), M.stride(1), M.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        S,
        num_heads=H,
        HEAD_DIM=D_head,
    )
    if need_aux:
        return O, M, L
    return O


def flash_attn_reference_causal(Q, K, V):
    scale = Q.shape[-1] ** -0.5
    S = Q.shape[-2]
    mask = torch.triu(torch.ones(S, S, device=Q.device), diagonal=1).bool()
    score = torch.matmul(Q, K.transpose(-2, -1)) * scale
    score = score.masked_fill(mask, float('-inf'))
    attn = torch.softmax(score, dim=-1)
    return torch.matmul(attn, V)


def roofline_bw_pct(tflops, ai, peak_bw_gb):
    return tflops * 1000 / (ai * peak_bw_gb) * 100


def benchmark(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


if __name__ == "__main__":
    torch.manual_seed(42)
    PEAK_BW_GB = 384
    D = 128
    H = 8
    D_head = D // H

    print(f"{'S':>6} | {'Tri ms':>8} | {'Tri TFLOPS':>11} | {'Tri BW%':>8} | {'SDPA ms':>8} | {'SDPA TFLOPS':>11} | {'Speedup':>7}")
    print("-" * 85)

    for S in [512, 1024, 2048, 4096]:
        B = 1
        Q_single = torch.randn(B, S, D, device='cuda', dtype=torch.float16)
        K_single = torch.randn(B, S, D, device='cuda', dtype=torch.float16)
        V_single = torch.randn(B, S, D, device='cuda', dtype=torch.float16)
        Q = Q_single.view(B, S, H, D_head).transpose(1, 2)
        K = K_single.view(B, S, H, D_head).transpose(1, 2)
        V = V_single.view(B, S, H, D_head).transpose(1, 2)

        ms_tri = benchmark(flash_attn_triton_causal, Q, K, V)
        ms_sdpa = benchmark(lambda q,k,v: F.scaled_dot_product_attention(q, k, v, is_causal=True), Q, K, V)

        flops = 2 * B * H * S * S * D_head
        tflops_tri = flops / (ms_tri * 1e-3) / 1e12
        tflops_sdpa = flops / (ms_sdpa * 1e-3) / 1e12

        cfg = flash_attn_fwd_causal.best_config
        bm = cfg.kwargs['BLOCK_M']
        bp = 2
        N = triton.cdiv(S, bm)
        q_bytes = S * D_head * bp
        kv_rows = sum(min((i + 1) * bm, S) for i in range(N))
        kv_bytes = kv_rows * D_head * bp
        o_bytes = S * D_head * bp
        tri_bytes = B * H * (q_bytes + 2 * kv_bytes + o_bytes)
        bw_tri = tri_bytes / (ms_tri * 1e-3) / 1e9
        bw_pct_tri = bw_tri / PEAK_BW_GB * 100

        speedup = ms_sdpa / ms_tri

        print(f"{S:>6} | {ms_tri:>8.3f} | {tflops_tri:>11.2f} | {bw_pct_tri:>7.1f}% | "
              f"{ms_sdpa:>8.3f} | {tflops_sdpa:>11.2f} | {speedup:>7.3f}")
        print(f"    ↳ best: BM={bm}, BN={cfg.kwargs['BLOCK_N']}, "
              f"ns={cfg.num_stages}, nw={cfg.num_warps}")
        print(f"    ↳ HBM: {tri_bytes/1e6:.1f} MB (K/V {kv_bytes*B*H/1e6:.1f} MB), 实际带宽 {bw_tri:.1f} GB/s")

    print("\n--- 精度验证 ---")
    for S in [64, 128, 256]:
        B, D = 1, 128
        Q_single = torch.randn(B, S, D, device='cuda', dtype=torch.float32)
        K_single = torch.randn(B, S, D, device='cuda', dtype=torch.float32)
        V_single = torch.randn(B, S, D, device='cuda', dtype=torch.float32)
        Q = Q_single.view(B, S, H, D_head).transpose(1, 2)
        K = K_single.view(B, S, H, D_head).transpose(1, 2)
        V = V_single.view(B, S, H, D_head).transpose(1, 2)

        ref = flash_attn_reference_causal(Q, K, V)
        tri = flash_attn_triton_causal(Q, K, V)

        err = (ref - tri).abs().max().item()
        status = "✅" if err < 3e-3 else "❌"
        print(f"S={S:4d}  Causal Max error: {err:.2e}  {status}")