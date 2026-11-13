# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
"""
FlashAttention Triton 实现性能对比测试

对比 Triton 实现与 PyTorch SDPA 在不同序列长度、不同 mask 模式下的性能差异。
同时输出内核元信息和差距归因分析，用于理解 Triton 与 CUDA 实现的差距。

Usage:
    python test_v2_minimal.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1_forward.step1_no_mask import flash_attn_triton
from v1_forward.step2_causal import flash_attn_triton_causal
from v1_forward.step3_varlen import flash_attn_triton_varlen


def benchmark(fn, *args, warmup=10, iters=50):
    """Benchmark a function, return average latency in ms."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn(*args)
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def sdpa(q, k, v, causal=False):
    """PyTorch scaled dot-product attention (cuDNN backend)."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


# ============================================================
# Section 0: 环境信息
# ============================================================
def print_env():
    print("=" * 60)
    print("  FlashAttention Triton Benchmark")
    print("=" * 60)
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU      : {p.name}")
    print(f"  SM       : sm_{p.major}{p.minor}  |  SMs: {p.multi_processor_count}")
    print(f"  Threads  : {p.max_threads_per_block}/block")
    print(f"  Shared   : {p.shared_memory_per_block // 1024} KB/block")
    print(f"  VRAM     : {p.total_memory / 1024**3:.1f} GiB")
    print(f"  CUDA     : {torch.version.cuda}")
    print(f"  PyTorch  : {torch.__version__}")
    print()


# ============================================================
# Section 1: 正确性验证
# ============================================================
def check_correctness():
    print("=" * 60)
    print("  Section 1: Correctness — Triton vs SDPA (fp16, causal)")
    print("=" * 60)

    torch.manual_seed(42)
    D = 64

    for S in [128, 256, 512]:
        q = torch.randn(1, S, D, device='cuda', dtype=torch.float16)
        k = torch.randn(1, S, D, device='cuda', dtype=torch.float16)
        v = torch.randn(1, S, D, device='cuda', dtype=torch.float16)

        ref = sdpa(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), causal=True).squeeze(1)
        out = flash_attn_triton_causal(q, k, v)
        err = (ref.float() - out.float()).abs().max().item()
        ok = "PASS" if err < 1e-3 else "FAIL"
        print(f"  S={S:>4d}  max_err={err:.1e}  [{ok}]")

    print()


# ============================================================
# Section 2: 性能测试 — no mask
# ============================================================
def perf_no_mask():
    print("=" * 60)
    print("  Section 2: Performance — no mask, fp16")
    print("=" * 60)

    torch.manual_seed(42)
    D = 64
    print(f"  {'S':>6} | {'Triton ms':>10} | {'Triton TF':>9} | {'SDPA ms':>10} | {'SDPA TF':>9} | {'Ratio':>7}")
    print("  " + "-" * 60)

    for S in [512, 1024, 2048, 4096]:
        q = torch.randn(1, S, D, device='cuda', dtype=torch.float16)
        k = torch.randn(1, S, D, device='cuda', dtype=torch.float16)
        v = torch.randn(1, S, D, device='cuda', dtype=torch.float16)
        flops = 4 * 1 * S * S * D

        ms_tri = benchmark(flash_attn_triton, q, k, v)
        ms_sdpa = benchmark(lambda: sdpa(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), causal=False))

        tf = lambda ms: flops / (ms * 1e-3) / 1e12
        ratio = ms_sdpa / ms_tri if ms_tri > 0 else 0

        print(f"  {S:>6} | {ms_tri:>10.3f} | {tf(ms_tri):>9.2f} | "
              f"{ms_sdpa:>10.3f} | {tf(ms_sdpa):>9.2f} | {ratio:>6.2f}x")

    print()


# ============================================================
# Section 3: 性能测试 — causal
# ============================================================
def perf_causal():
    print("=" * 60)
    print("  Section 3: Performance — causal, fp16")
    print("=" * 60)

    torch.manual_seed(42)
    D = 64
    print(f"  {'S':>6} | {'Triton ms':>10} | {'Triton TF':>9} | {'SDPA ms':>10} | {'SDPA TF':>9} | {'Ratio':>7}")
    print("  " + "-" * 60)

    for S in [512, 1024, 2048, 4096]:
        q = torch.randn(1, S, D, device='cuda', dtype=torch.float16)
        k = torch.randn(1, S, D, device='cuda', dtype=torch.float16)
        v = torch.randn(1, S, D, device='cuda', dtype=torch.float16)
        flops = 2 * 1 * S * S * D

        ms_tri = benchmark(flash_attn_triton_causal, q, k, v)
        ms_sdpa = benchmark(lambda: sdpa(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), causal=True))

        tf = lambda ms: flops / (ms * 1e-3) / 1e12
        ratio = ms_sdpa / ms_tri if ms_tri > 0 else 0

        print(f"  {S:>6} | {ms_tri:>10.3f} | {tf(ms_tri):>9.2f} | "
              f"{ms_sdpa:>10.3f} | {tf(ms_sdpa):>9.2f} | {ratio:>6.2f}x")

    print()


# ============================================================
# Section 4: 性能测试 — varlen
# ============================================================
def perf_varlen():
    print("=" * 60)
    print("  Section 4: Performance — varlen causal, fp16 (Triton only)")
    print("=" * 60)

    torch.manual_seed(42)
    D = 64
    print(f"  {'max_S':>6} | {'seqlens':>20} | {'Triton ms':>10} | {'Triton TF':>9}")
    print("  " + "-" * 52)

    for seqlens in [[128, 128, 128], [256, 256, 256], [512, 256, 128], [1024, 512, 256]]:
        total = sum(seqlens)
        m = max(seqlens)
        cu = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0)
        q = torch.randn(total, D, device='cuda', dtype=torch.float16)
        k = torch.randn(total, D, device='cuda', dtype=torch.float16)
        v = torch.randn(total, D, device='cuda', dtype=torch.float16)
        flops = 2 * sum(s * s for s in seqlens) * D

        ms = benchmark(flash_attn_triton_varlen, q, k, v, cu.cuda())
        tf = flops / (ms * 1e-3) / 1e12
        print(f"  {m:>6} | {str(seqlens):>20} | {ms:>10.3f} | {tf:>9.2f}")

    print()


# ============================================================
# Section 5: 内核元信息 & 差距归因
# ============================================================
def kernel_meta():
    print("=" * 60)
    print("  Section 5: Kernel Meta & Gap Analysis")
    print("=" * 60)

    p = torch.cuda.get_device_properties(0)
    D = 64
    smem = p.shared_memory_per_block

    print(f"  Triton autotune search space: BM=[64,128] BN=[64,128] warps=[4,8] stages=[2,3,4]")
    print(f"  Total configs: 2x2x2x3 = 24, auto-selected at runtime")
    print(f"\n  Shared Memory Usage (D={D}):")
    for BM in [64, 128]:
        for BN in [64, 128]:
            t = (BM + BN + BN) * D * 2
            ok = "OK" if t <= smem else "OVERFLOW"
            print(f"    BM={BM:3d} BN={BN:3d}  total={t//1024:2d}KB  [{ok}]")

    print(f"\n  Gap Analysis — Triton vs FA2 (CUDA):")
    print(f"    1. cp.async / TMA          — async HBM-to-smem copy vs sync tl.load")
    print(f"    2. WMMA / wgmma            — warp-level MMA vs per-warp tl.dot")
    print(f"    3. Swizzle / padding       — bank conflict elimination in smem")
    print(f"    4. Double buffering         — load-compute overlap via kStages")
    print(f"    5. Compile-time heuristics  — vs runtime autotune search")
    print(f"    6. Split-K                  — long sequence partitioning across SMs")
    print()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print_env()
    check_correctness()
    perf_no_mask()
    perf_causal()
    perf_varlen()
    kernel_meta()