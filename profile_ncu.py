# @# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v1_forward.step3_varlen import flash_attn_triton_varlen
from v1_backward.step4_varlen_causal import flash_attn_backward


def run_profile_case():
    torch.manual_seed(42)
    H = 8
    D_head = 64
    seqlens = [2048]

    B = len(seqlens)
    max_seqlen = max(seqlens)
    cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32).cumsum(dim=0)

    Q = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, max_seqlen, D_head, device='cuda', dtype=torch.float32)
    dO = torch.randn_like(Q)

    for _ in range(3):
        with torch.no_grad():
            O, M, L = flash_attn_triton_varlen(Q, K, V, cu_seqlens)
            flash_attn_backward(Q, K, V, O, dO, M, L, cu_seqlens)

    torch.cuda.synchronize()


if __name__ == "__main__":
    run_profile_case()