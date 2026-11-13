# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
import torch
import triton
import triton.language as tl


# ┌─────────────────────────────────────────────────────────┐
# │ Online Softmax 单 program 版流程图                       │
# │                                                         │
# │  program_id(0) → row_idx                                │
# │                                                         │
# │  Pass 1:                                                │
# │    m = -inf                                             │
# │    l = 0.0                                              │
# │    for start in range(0, n_cols, BLOCK):                │
# │        x = tl.load(row_base + start + arange, mask)     │
# │        m_new = max(m, max(x))                           │
# │        alpha = exp(m - m_new)                           │
# │        l = alpha * l + sum(exp(x - m_new))              │
# │        m = m_new                                        │
# │    store m, l to global                                 │
# │                                                         │
# │  Pass 2:                                                │
# │    for start in range(0, n_cols, BLOCK):                │
# │        x = tl.load(row_base + start + arange, mask)     │
# │        out = exp(x - m) / l                             │
# │        tl.store(output + start + arange, out, mask)     │
# │                                                         │
# │  ⚠️ 为什么两遍: online 里 m 会变, 之前输出的 exp 基准错了 │
# │  ⚠️ FlashAttention 不需要两遍: o 在 SRAM 里可原地 alpha 缩放 │
# └─────────────────────────────────────────────────────────┘


@triton.jit
def softmax_online(
    input_ptr,
    output_ptr,
    m_ptr,
    l_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # ── Step 1: program_id → 行号 → 行基址 ──
    row_idx = tl.program_id(0) # program_id == Block ID (CUDA)
    row_base = row_idx * n_cols

    # ── Step 2: 初始化 online softmax 状态 ──
    m = -float('inf')
    l = 0.0

    # ── Step 3 之后: 循环 + Pass 2 —— 下一步写 ──

    # Pass 1: 分块扫描，online更新 m, l
    for start in range(0, n_cols,BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_t = tl.load(input_ptr + row_base + offsets, mask)

        # 沿行规约，更新 m 最大值
        m_new = tl.maximum(m, tl.max(x_t, axis=0))
        alpha = tl.exp(m - m_new)
        l = alpha * l + tl.sum(tl.exp(x_t - m_new), axis=0)
        m = m_new

    # 存最终 m,l 到global
    tl.store(m_ptr + row_idx, m)
    tl.store(l_ptr + row_idx, l)

    # Pass2: 用最终 m, l 算 softmax
    for start in range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x_t = tl.load(input_ptr + row_base + offsets, mask)
        out = tl.exp(x_t - m) / l
        tl.store(output_ptr + row_base + offsets, out, mask)

def softmax_online_pytorch(x):
    """Python wrapper —— 调 triton kernel"""
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    # 创建张量 m, l
    m = torch.empty(n_rows, device = x.device, dtype = torch.float32)
    l = torch.empty(n_rows, device = x.device, dtype = torch.float32)
    BLOCK_SIZE = min(triton.next_power_of_2(n_cols),1024)
    softmax_online[(n_rows,)](x, out, m, l, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out, m, l

if __name__ == "__main__":
    torch.manual_seed(42)

    for n_cols in [128,512,1024,2048,8192]:
        x = torch.randn(4, n_cols, device='cuda', dtype=torch.float32)
        ref = torch.softmax(x, dim=-1)
        tri, m, l = softmax_online_pytorch(x)
        err = (ref - tri).abs().max().item()
        status = "✅" if err < 1e-5 else "❌"
        print(f"n_cols={n_cols:5d}  max_err={err:.2e}  {status}")

    x_big = torch.tensor([[1000.0, 1001.0, 999.0, 998.0]], device='cuda')
    ref_big = torch.softmax(x_big, dim=-1)
    tri_big, m_big, l_big = softmax_online_pytorch(x_big)
    err_big = (ref_big - tri_big).abs().max().item()
    print(f"Big values: max_err={err_big:.2e}  m={m_big.item():.1f}  l={l_big.item():.4f}  {'✅' if err_big < 1e-5 else '❌'}")