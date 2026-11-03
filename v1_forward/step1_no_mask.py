import torch
import triton
import triton.language as tl
import torch.nn.functional as F

'''
┌─────────────────────────────────────────────┐
│ 段 1: 加载 Q [BLOCK_M, D]                   │
│   q = tl.load(Q_ptr + ...)                  │
├─────────────────────────────────────────────┤
│ 段 2: 初始化状态                             │
│   m = -inf, l = 0, o = zeros, scale = 1/√D │
├─────────────────────────────────────────────┤
│ 段 3: for block_n in range(0, S, BLOCK_N):  │
│       k, v = tl.load(...)                    │
│       score = Q @ K^T * scale               │ ← 矩阵乘法 1
│       online softmax 更新 m, l              │
│       o += P @ V                            │ ← 矩阵乘法 2
├─────────────────────────────────────────────┤
│ 段 4: O = o / l, store(O)                   │
└─────────────────────────────────────────────┘
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
def flash_attn_fwd(
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
    q = tl.load(q_ptrs, mask = q_offs[:, None] < seqlen, other = 0.0)

    m = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    o = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    for block_n in range(0, seqlen, BLOCK_N):
        n_offs = block_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + batch_idx * stride_kb + head_idx * stride_kh + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kk
        k = tl.load(k_ptrs, mask = n_offs[:, None] < seqlen, other = float('-inf'))

        v_ptrs = V_ptr + batch_idx * stride_vb + head_idx * stride_vh + n_offs[:, None] * stride_vn + d_offs[None, :] * stride_vk
        v = tl.load(v_ptrs, mask = n_offs[:, None] < seqlen, other = 0.0)

        score = tl.dot(q,tl.trans(k)) * scale

        m_new = tl.maximum(m, tl.max(score,axis = -1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new[:, None])
        l = alpha * l + tl.sum(p,axis = 1)
        o = alpha[:, None] * o + tl.dot(p.to(v.dtype), v)
        m = m_new

    o = o / l[:, None]

    o_ptrs = O_ptr + batch_idx * stride_ob + head_idx * stride_oh + q_offs[:, None] * stride_om + d_offs[None, :] * stride_ok
    tl.store(o_ptrs, o.to(O_ptr.dtype.element_ty), mask=q_offs[:, None] < seqlen)

    m_ptrs = M_ptr + batch_idx * stride_mb + head_idx * stride_mh + q_offs * stride_ms
    tl.store(m_ptrs, m, mask=q_offs < seqlen)

    l_ptrs = L_ptr + batch_idx * stride_lb + head_idx * stride_lh + q_offs * stride_ls
    tl.store(l_ptrs, l, mask=q_offs < seqlen)


def flash_attn_triton(Q, K, V, need_aux=True):
    B, H, S, D_head = Q.shape
    O = torch.empty_like(Q)
    M = torch.empty(B, H, S, device=Q.device, dtype=torch.float32)
    L = torch.empty(B, H, S, device=Q.device, dtype=torch.float32)
    grid = lambda META:(triton.cdiv(S,META['BLOCK_M']), B * H)
    flash_attn_fwd[grid](
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


def flash_attn_reference(Q, K, V):
    scale = Q.shape[-1] ** -0.5
    score = torch.matmul(Q, K.transpose(-2, -1)) * scale
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

def actual_hbm_bytes_step1(B, H, S, D_head, BLOCK_M):
    """step1 无 mask 的实际 HBM 总字节（fp16）
    Q load 1 次，K/V 各被 S/BM 个 program 完整加载"""
    bp = 2
    N = triton.cdiv(S, BLOCK_M)
    q = S * D_head * bp
    kv = N * S * D_head * bp
    o = S * D_head * bp
    return B * H * (q + 2 * kv + o)

if __name__ == "__main__":
    torch.manual_seed(42)
    PEAK_BW_GB = 384
    D = 128
    H = 8
    D_head = D // H

    print(f"{'S':>6} | {'Tri ms':>8} | {'Tri TFLOPS':>11} | {'Tri BW%':>8} | {'SDPA ms':>8} | {'SDPA TFLOPS':>11} | {'SDPA BW%':>8} | {'Speedup':>7}")
    print("-" * 90)

    for S in [512, 1024, 2048, 4096]:
        B = 1
        Q_single = torch.randn(B, S, D, device='cuda', dtype=torch.float16)
        K_single = torch.randn(B, S, D, device='cuda', dtype=torch.float16)
        V_single = torch.randn(B, S, D, device='cuda', dtype=torch.float16)
        Q = Q_single.view(B, S, H, D_head).transpose(1, 2)
        K = K_single.view(B, S, H, D_head).transpose(1, 2)
        V = V_single.view(B, S, H, D_head).transpose(1, 2)

        ms_tri = benchmark(flash_attn_triton, Q, K, V)
        ms_sdpa = benchmark(F.scaled_dot_product_attention, Q, K, V)

        flops = 4 * B * H * S * S * D_head
        tflops_tri = flops / (ms_tri * 1e-3) / 1e12
        tflops_sdpa = flops / (ms_sdpa * 1e-3) / 1e12

        cfg = flash_attn_fwd.best_config
        bm = cfg.kwargs['BLOCK_M']
        tri_bytes = actual_hbm_bytes_step1(B, H, S, D_head, bm)
        bw_tri = tri_bytes / (ms_tri * 1e-3) / 1e9
        bw_pct_tri = bw_tri / PEAK_BW_GB * 100

        speedup = ms_sdpa / ms_tri

        print(f"{S:>6} | {ms_tri:>8.3f} | {tflops_tri:>11.2f} | {bw_pct_tri:>7.1f}% | "
              f"{ms_sdpa:>8.3f} | {tflops_sdpa:>11.2f} | {'—':>8} | {speedup:>7.3f}")
        print(f"    ↳ best: BM={bm}, BN={cfg.kwargs['BLOCK_N']}, "
              f"ns={cfg.num_stages}, nw={cfg.num_warps}")
        print(f"    ↳ HBM: {tri_bytes/1e6:.1f} MB, 实际带宽 {bw_tri:.1f} GB/s")