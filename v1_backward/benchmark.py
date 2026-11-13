# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
import time
import sys
import torch

sys.path.insert(0, r'd:\AI HPC\FlashAttention')
from v1_forward.step3_varlen import flash_attn_triton_varlen
from v1_backward.step4_varlen_causal import (
    flash_attn_backward as triton_bwd,
    flash_attn_reference_backward as ref_bwd,
)


def benchmark(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


print(f"{'seqlens':<20} {'triton(ms)':<12} {'ref_pure(ms)':<14} {'speedup':<8}")
print("-" * 58)

H, D = 8, 64

for seqlens in [[128], [256], [512], [1024], [2048],
                [128, 256], [256, 512], [128, 256, 512]]:
    B = len(seqlens)
    max_seqlen = max(seqlens)
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0)

    Q = torch.randn(B, H, max_seqlen, D, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, max_seqlen, D, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, max_seqlen, D, device='cuda', dtype=torch.float32)
    dO = torch.randn_like(Q)

    with torch.no_grad():
        tri_O, tri_M, tri_L = flash_attn_triton_varlen(Q, K, V, cu_seqlens)

        t_triton = benchmark(triton_bwd, Q, K, V, tri_O, dO, tri_M, tri_L, cu_seqlens)
        t_ref = benchmark(ref_bwd, Q, K, V, dO, cu_seqlens)

    speedup = t_ref / t_triton
    print(f"{str(seqlens):<20} {t_triton:<12.2f} {t_ref:<14.2f} {speedup:<8.2f}x")
