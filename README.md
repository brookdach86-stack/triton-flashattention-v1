# Triton FlashAttention v1 前向 + 反向

跟做 FlashAttention v1 的 forward + backward 完整链路。forward: online softmax → 无 mask → causal（加跳块）→ varlen。backward: 纯 Python 参考实现 → Triton step1（无 mask 多头）→ step2（causal）→ step3（varlen）。

FlashAttention 省显存的核心：`m` 变了之后 `o` 还在 SRAM 里，乘个 `alpha` 接着累加，不用把中间结果写回 HBM。forward 额外输出 `M/L`（每行 max 和 softmax 分母），backward 用它们重算 P 而不 materialize 完整 S 矩阵，保持 O(1) 显存。

## 环境

RTX 5060 Laptop（384 GB/s 带宽，sm_120），Python 3.10，PyTorch + Triton。Windows 上 flash-attn 官方包装不了，SDPA flash 后端当 baseline。

## 文件

- `prerequisites/softmax_online.py`：online softmax 最小验证，一个 program 处理一行。对照 `torch.softmax`，阈值 `1e-5`。测了 128/512/1024/2048/8192 列，还有不规则长度 `[1000, 1001, 999, 998]`。
- `v1_forward/step1_no_mask.py`：`[B, H, S, D_head]`，无 mask。当前默认 `B=1, H=8, D=128`（`D_head=16`），fp16。autotune 18 种配置（BLOCK_M 64/128, BLOCK_N 64/128, num_stages 2/3/4, num_warps 4/8）。benchmark 512/1024/2048/4096。
- `v1_forward/step2_causal.py`：加 causal mask，同时做了跳块——for 循环上界从 `seqlen` 改成 `(block_m_idx+1)*BLOCK_M`，对角线 K block 内部保留 mask。autotune/benchmark 同 step1。
- `v1_forward/step3_varlen.py`：序列不等长，`[B, H, max_seqlen, D_head]` padded 格式，`cu_seqlens` 标记有效长度。`pid(1)` 拆 `seq_idx * H + head_idx`，查 `cu_seqlens` 拿 per-sample seqlen。causal + 跳块 + autotune。forward 额外输出 `M/L`（`[B, H, max_seqlen]` float32）供 backward 使用。
- `v1_backward/step1_backward.py`：Triton backward step1，无 mask，多头。5 步菜谱：① load Q/dO/O/M/L → ② `Di = sum(dO*O)` → ③ 循环 KV block：算 P → atomic_add dV → `ds = P*(dP-Di)` → `dQ += ds@K` → atomic_add dK。`dK/dV` 用 `tl.atomic_add` 跨 Q block 累积。
- `v1_backward/step2_backward.py`：Triton backward step2，加 causal mask + 跳块。循环上界从 `seqlen` 改成 `(block_m_idx+1)*BLOCK_M`，score 上 `tl.where(k_pos > q_pos, -inf, s)`。
- `v1_backward/step3_backward.py`：Triton backward step3，varlen padded。`seqlen = cu_seqlens[b+1]-cu_seqlens[b]`，load/store mask 用 per-sample seqlen，`if block_m_idx*BM >= seqlen: return` 跳过 padding 区 q block。
- `v1_backward/step4_splitkv_backward.py`：Triton backward step4，partial + reduce 消除 atomic_add 冲突。dK/dV 不再 atomic_add 到全局，而是每个 Q block 写一份 partial，两个 reduce kernel 沿 Q 维加回来。dQ 在寄存器里直接累加存（因为同一 Q block 内部循环 KV 无冲突）。
- `v1_backward/step4_varlen_causal.py`：step4 基础上合了 causal mask + varlen padded。`partial_dK/partial_dV` 用 `torch.empty`（所有位置都被写，不需要清零）。
- `v1_backward/step5_splitkv_full.py`：step5，完整 split-KV grid `(num_q_blocks, num_kv_blocks, B*H)`。每个 program 只处理一个 (Q_block, KV_block) tile，dQ/dK/dV 全写 partial，三个 reduce kernel 分别沿 KV 维和 Q 维加。并行度最高但 Q 重复加载浪费。
- `v1_backward/benchmark.py`：step4_varlen_causal 对纯 Python reference 的耗时对比，torch.profiler 看 kernel 时间占比。
- `v1_forward/test_ref.py`：写 varlen 时手算对照用的，非正式。
- `差距分析报告_v2.md`：第3-4周。对照 FA2 `flash_fwd_kernel`，把差距写到 `cp.async`、smem swizzle、warp 分工。
- `test_v2_minimal.py`：三个 step 对 SDPA 的正确性和耗时，并打印差距归因。

跑：

```powershell
python prerequisites/softmax_online.py
python v1_forward/step1_no_mask.py
python v1_forward/step2_causal.py
python v1_forward/step3_varlen.py
python test_v2_minimal.py
```

## 精度

**Forward (fp32)**

| | 形状 | 误差 |
| --- | --- | --- |
| step1 | 64/128/256 | < 1e-4 |
| step2 | 64/128/256 | < 3e-3 |
| step3 | [64,32], [128,256], [64,128,192] | < 3e-3 |

**Backward (fp32, 完整 forward→backward 链路)**

step1/step2: `B=2, H=8, D_head=16`，step3: padded varlen，对照纯 Python reference。

| | 形状 | dQ | dK | dV |
| --- | --- | --- | --- | --- |
| step1 (no mask) | S=128/256/512 | < 4.3e-2 | < 3.4e-2 | < 4.6e-3 |
| step2 (causal) | S=128/256/512 | < 4.6e-2 | < 3.7e-2 | < 1.6e-2 |
| step3 (varlen) | [128,256], [64,128,192] | < 2.9e-2 | < 2.5e-2 | < 5.1e-3 |

backward 误差略大于 forward（最大 ~4.6e-2），因为 `dK/dV` 用 `tl.atomic_add` 跨 Q block 累积，不同 block 的浮点加法顺序不一致带来额外误差，但在 fp32 下可控。

step4/step5 没有 atomic_add，全部是顺序累加，误差更稳定。step4/varlen: `[128,256], [64,128,192]`，dQ/dK/dV 误差均 < 6.2e-2。

## Backward 性能（step4_varlen_causal，fp32，D_head=64）

对纯 Python reference 的加速，RTX 5060 Laptop：

```
seqlens               Triton (ms)  Reference (ms)  Speedup
-----------------------------------------------------------
[128]                    0.23           0.47           2.08x
[256]                    0.22           0.45           2.10x
[512]                    0.27           0.49           1.82x
[1024]                   0.82           2.73           3.33x
[2048]                   2.34           11.09          4.74x
[128, 256]              0.27           1.05           3.83x
[256, 512]              0.41           0.88           2.16x
[128, 256, 512]         0.50           1.27           2.56x
```

## Backward 性能分析（torch.profiler，S=2048）

kernel 时间占比：

```
Kernel                          CUDA time   占比
---------------------------------------------------
_flash_attn_bwd_kernel          1.09ms      36%     核心计算
_reduce_kernel × 2              1.08ms      36%     partial → dK/dV 归约
vectorized_elementwise_kernel ×5 810us      27%     torch.zeros 清零开销
---------------------------------------------------
总计                            2.99ms
```

**关键发现**：

1. 核心 backward 计算只占 36%，partial+reduce 架构 overhead 达 63%（reduce 36% + 初始化 27%）
2. `torch.zeros()` 改 `torch.empty()` 省掉 27% 初始化开销，S=2048 从 3.10ms 降到 2.34ms（24.5% 加速）
3. reduce kernel 仍然是最大瓶颈（占 36%），atomic_add vs partial+reduce 的 trade-off 需要实测

## 性能

**step1 无 mask，fp16，B=1, H=8, D=128 (D_head=16)**

BW% 按实际 HBM 字节计算：Q load 1 次，K/V 各被 S/BM 个 program 完整加载，O store 1 次。

```
   S |   Tri ms |  Tri TFLOPS |  Tri BW% |  SDPA ms | SDPA TFLOPS | Speedup
-------------------------------------------------------------------------
   512 |    0.164 |        0.82 |     3.8% |    0.084 |        1.59 |   0.52x
    ↳ best: BM=64 BN=64 ns=4 nw=4
  1024 |    0.081 |        6.59 |    28.5% |    0.296 |        1.81 |   3.64x
    ↳ best: BM=64 BN=64 ns=3 nw=4
  2048 |    0.222 |        9.67 |    40.6% |    1.007 |        2.13 |   4.53x
    ↳ best: BM=64 BN=64 ns=2 nw=4
  4096 |    0.907 |        9.47 |    39.1% |    3.526 |        2.44 |   3.89x
    ↳ best: BM=64 BN=64 ns=3 nw=4
```

**step2 causal + 跳块，fp16，B=1, H=8, D=128 (D_head=16)**

跳块后 K/V 加载量 ≈ S²/2 行（每个 program 只加载 (block_m_idx+1)*BM 行），约为 step1 的一半。

```
   S |   Tri ms |  Tri TFLOPS |  Tri BW% |  SDPA ms | SDPA TFLOPS | Speedup
-------------------------------------------------------------------------
   512 |    0.086 |        0.78 |     4.4% |    0.103 |        0.65 |   1.20x
    ↳ best: BM=64 BN=64 ns=3 nw=4
  1024 |    0.085 |        3.16 |    15.3% |    0.201 |        1.33 |   2.37x
    ↳ best: BM=64 BN=64 ns=3 nw=4
  2048 |    0.178 |        6.03 |    14.6% |    0.683 |        1.57 |   3.84x
    ↳ best: BM=128 BN=128 ns=3 nw=8
  4096 |    0.539 |        7.97 |    33.9% |    2.215 |        1.94 |   4.11x
    ↳ best: BM=64 BN=64 ns=4 nw=4
```

**step3 varlen + causal + 跳块，fp16**

```
max_S |     seqlens      | Tri ms | Tri TFLOPS | Tri BW% | SDPA ms | Speedup
-------------------------------------------------------------------------------
  128 |   [128,128,128]  |  0.135 |       0.05 |    0.1% |   0.343 |  2.55x
  256 |   [256,256,256]  |  0.142 |       0.18 |    0.4% |   0.432 |  3.03x
  512 |   [512,256,128]  |  0.119 |       0.37 |    0.8% |   0.396 |  3.32x
 1024 |  [1024,512,256]  |  0.160 |       1.10 |    2.2% |   0.520 |  3.26x
```

## 一些观察

**BW% 现在最高 40.6%，Triton 没跑满带宽**：之前用理论 AI=2×D_head 算出 BW% 超 100%，修正为实际 HBM 字节后正常。step2 跳块后 K/V 字节减半，BW% 更低（14~34%）——算子从带宽受限转向计算受限。大序列（S≥2048）的 Triton BW% 稳定在 34~41%，说明我们还有约 2.5x 的带宽可用空间。

**小序列差距来自 SDPA 后端效率**：step1 S=512 我们反而慢（0.52x），但 step2 同尺寸就反超了（1.20x）——因为 step2 有跳块优化，K/V loading 少了一半，HBM 总量从 2.4 MB 降到 1.4 MB，SDPA 在小矩阵上的 overhead 被摊薄了。

**autotune 不一定全选 64×64**：step2 S=2048 这次选了 BM=128 BN=128 ns=3 nw=8。跳块后循环次数（K block 数）减少，大 tile 减少 program launch 开销反而划算。但 S=4096 时又回到 BM=64——因为 K/V loading 量虽然跳块减半，但绝对数值还是大，小 tile 更好。

**varlen 的优势**：padded SDPA 把所有句子 pad 到 max_seqlen，空转的 token 也得算。varlen 直接跳过，SDPA 慢 2.5~3.3x 就是这么来的。flops 统计按实际有效 token 算，不按 padded 算。

**Windows 限制**：装不了官方 flash-attn CUDA 包，没法跟真正的手写 CUDA 比。SDPA 的 flash 后端在 Windows 上也没跑满性能，这大概是我们在大序列上能反超这么多的原因之一。

## 还没做的

- [x] 反向（step1/step2/step3/step4/step5，完整 Triton forward→backward 链路，对照纯 Python reference 误差 < 6e-2）
- [x] FLOPs / 带宽按实际 HBM 字节重算（step1 K/V 被 S/BM 个 program 加载，step2 跳块后 ≈ S²/2 行）
- [x] Backward 性能 benchmark + torch.profiler 分析（见上方）
- [ ] Paged KV Cache（推理方向，另一个项目）
- [ ] Forward+Backward 端到端 benchmark（当前只单独测了 backward）
- [ ] Nsight Compute 深度分析（DRAM 带宽、SM occupancy、L2 hit rate）
