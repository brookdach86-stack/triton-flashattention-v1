import time
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v1_forward.step3_varlen import flash_attn_triton_varlen
from v1_backward.step4_varlen_causal import flash_attn_backward


def benchmark(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    torch.manual_seed(42)
    H = 8
    D_head = 64

    print(f"{'seqlens':<22} {'fwd_ms':>9} {'bwd_ms':>9} {'fwd+bwd_ms':>11} {'fwd%':>7} {'bwd%':>7}")
    print("-" * 70)

    test_cases = [
        [128],
        [256],
        [512],
        [1024],
        [2048],
        [128, 256],
        [256, 512],
        [128, 256, 512],
    ]

    for seqlens in test_cases:
        B = len(seqlens)
        max_seqlen = max(seqlens)
        cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0)

        Q = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32, requires_grad=True)
        K = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32, requires_grad=True)
        V = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32, requires_grad=True)
        dO = torch.randn_like(Q)

        with torch.no_grad():
            O, M, L = flash_attn_triton_varlen(Q, K, V, cu_seqlens)

            t_fwd = benchmark(flash_attn_triton_varlen, Q, K, V, cu_seqlens)
            t_bwd = benchmark(flash_attn_backward, Q, K, V, O, dO, M, L, cu_seqlens)

            def fwd_bwd_pair():
                O2, M2, L2 = flash_attn_triton_varlen(Q, K, V, cu_seqlens)
                flash_attn_backward(Q, K, V, O2, dO, M2, L2, cu_seqlens)

            t_pair = benchmark(fwd_bwd_pair)

        t_sum = t_fwd + t_bwd
        fwd_pct = t_fwd / t_pair * 100
        bwd_pct = t_bwd / t_pair * 100

        print(f"{str(seqlens):<22} {t_fwd:>9.3f} {t_bwd:>9.3f} {t_pair:>11.3f} {fwd_pct:>6.1f}% {bwd_pct:>6.1f}%")

        dQ, dK, dV = flash_attn_backward(Q, K, V, O, dO, M, L, cu_seqlens)
        total = sum(s * s for s in seqlens)
        flops_bwd = 30 * H * total * D_head
        flops_fwd = 4 * H * total * D_head
        tflops_bwd = flops_bwd / (t_bwd * 1e-3) / 1e12
        tflops_fwd = flops_fwd / (t_fwd * 1e-3) / 1e12
        print(f"                        fwd={tflops_fwd:.2f} TFLOPS  bwd={tflops_bwd:.2f} TFLOPS")


if __name__ == "__main__":
    main()