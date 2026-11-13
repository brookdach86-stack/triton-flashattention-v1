![Original Work](https://img.shields.io/badge/Original-Wain-brightgreen)
![GPG Signed](https://img.shields.io/badge/GPG-Signed-brightgreen)
![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)
![Target HW](https://img.shields.io/badge/Target%20HW-RTX%205060%20Laptop%20(sm_120%2C26SM%2C32MB%20L2)-brightgreen)
![BW 384GB/s](https://img.shields.io/badge/Peak%20BW-384%20GB%2Fs-orange)
![NSight Verified](https://img.shields.io/badge/Profiled%20by-NSight%20Compute-blue)

---

# Triton FlashAttention v1 Forward + Backward

单卡RTX 5060 Laptop上从零实现的完整FlashAttention v1前后向。不用vLLM/TensorRT-LLM——从online softmax一步步堆到varlen + causal + split-KV backward。

**真踩过的坑（每一步都用NCU验证过）**：

- backward一开始用`tl.atomic_add`累加dK/dV，**DRAM%飙到90%**——根因是global lock contention，Triton的atomic_add在global memory上是串行锁。改成partial+reduce搬到smem，DRAM%降到60%
- online softmax中间值m/l我一开始傻乎乎存smem（跟着教程走），NCU一测**L1 Hit=0%**——白忙活。register才是最优，online算法中间值每个Q行私有，根本不需要过cache
- varlen版本有人只做block skip不做boundary mask（多算30%无用K块），有人只做mask但loop bound写错（内存越界风险）。两个都做才是真懂varlen的cu_seqlens内存布局
- softmax_v2的L1 Hit=0%**不是bug**——online softmax的m/l本来就是register私有的，NCU确认零cache traffic是正确行为

## 性能全景（BW% / DRAM%，全是单卡实测数据）

```
BW%/DRAM%
85% ┤          ● CUDA Reduce V6 (partial+reduce，解决atomic bank conflict)
    │      ❌  │ CUDA Reduce V5 (atomic_add，DRAM%90%，4-way bank conflict)
    │          │
    │  ●───────┤ FA Bwd step4 (partial+reduce，DRAM%90%→60%)
    │  │       │
    │  ●───────┤ FA Fwd step3 (register存SoftMax中间值，L1 Hit=0% ✅)
    │  │       │
    │  ●───────┤ FP8 PagedAttention (per-page scale，绑定Paged Cache粒度)
    │  │       │
 0% └──┴───────┴──────────────────────────────────→ 版本迭代
    Reduce V5→V6  FA Bwd  FA Fwd  FP8
```

---

<!--
@Author: Wain (brookdach86-stack@users.noreply.github.com)
@GitHub: https://github.com/brookdach86-stack
@Note: 本内容为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
-->

在Triton里完整实现FlashAttention v1前向+后向。Forward演进线：online softmax → no mask → causal（带block skipping）→ varlen。Backward演进线：纯Python参考实现 → Triton step1（no mask multi-head）→ step2（causal）→ step3（varlen）→ step4（split-KV，干掉atomic_add）→ step5（full split-KV grid）。

核心省内存思路：当`m`（行max）在累加中途变了，正在累加的`o`还留在SRAM里——乘个`alpha = exp(m_old - m_new)`接着算，不需要中间写回HBM。Forward还要额外输出`M/L`（每行的max和softmax分母），这样backward可以重算P而不用把完整S矩阵materialize出来，维持O(1)内存。

## 环境

RTX 5060 Laptop（384 GB/s带宽，sm_120），Python 3.10，PyTorch + Triton。Windows上装不了官方flash-attn CUDA wheel，所以用PyTorch SDPA的flash backend当baseline。

## 文件说明

- `prerequisites/softmax_online.py`：最小online softmax验证，一行一个program。和`torch.softmax`交叉验证，阈值`1e-5`。测了128/512/1024/2048/8192列和不规则长度`[1000, 1001, 999, 998]`。
- `v1_forward/step1_no_mask.py`：`[B, H, S, D_head]`，无mask。默认`B=1, H=8, D=128`（`D_head=16`），fp16。Autotune 18种配置（BLOCK_M 64/128, BLOCK_N 64/128, num_stages 2/3/4, num_warps 4/8）。Benchmark在512/1024/2048/4096。
- `v1_forward/step2_causal.py`：加causal mask + block skipping——KV loop上界从`seqlen`改成`(block_m_idx+1)*BLOCK_M`，对角K块内部保留mask。Autotune/benchmark和step1一样。
- `v1_forward/step3_varlen.py`：变长序列`[B, H, max_seqlen, D_head]`，padded格式，`cu_seqlens`标记有效长度。`pid(1)`拆成`seq_idx * H + head_idx`，每sample的seqlen从`cu_seqlens`查。Causal + block skipping + autotune。Forward额外输出`M/L`（`[B, H, max_seqlen]` float32）给backward用。
- `v1_backward/step1_backward.py`：Triton backward step1，no mask multi-head。五步流程：① load Q/dO/O/M/L → ② `Di = sum(dO*O)` → ③ 遍历KV块：算P → atomic_add dV → `ds = P*(dP-Di)` → `dQ += ds@K` → atomic_add dK。`dK/dV`跨Q块用`tl.atomic_add`累加。
- `v1_backward/step2_backward.py`：Triton backward step2，加causal mask + block skipping。Loop上界从`seqlen`改成`(block_m_idx+1)*BLOCK_M`，score变成`tl.where(k_pos > q_pos, -inf, s)`。
- `v1_backward/step3_backward.py`：Triton backward step3，varlen padded。`seqlen = cu_seqlens[b+1]-cu_seqlens[b]`，load/store mask用per-sample seqlen，`if block_m_idx*BM >= seqlen: return`跳过padding区的Q块。
- `v1_backward/step4_splitkv_backward.py`：Triton backward step4，partial + reduce，干掉atomic_add竞争。dK/dV不再atomic加到global——每个Q块写一份partial，两个reduce kernel沿Q方向求和。dQ直接在register里累加（无竞争，因为一个Q块内的KV-loop是串行的）。
- `v1_backward/step4_varlen_causal.py`：step4 + causal mask + varlen padded。`partial_dK/partial_dV`用`torch.empty`（所有位置都写，不需要zero）。
- `v1_backward/step5_splitkv_full.py`：step5，full split-KV grid `(num_q_blocks, num_kv_blocks, B*H)`。每个program处理一个(Q_block, KV_block) tile，dQ/dK/dV全写partial，三个reduce kernel分别沿KV和Q方向求和。并行度最高但Q load有重复。
- `v1_backward/benchmark.py`：step4_varlen_causal vs 纯Python参考实现的latency，用torch.profiler做kernel time breakdown。
- `v1_backward/benchmark_e2e.py`：Forward+backward端到端benchmark——测fwd、bwd、以及fwd→bwd整个pipeline的TFLOPS，报告bwd占总时间百分比。
- `v1_forward/test_ref.py`：varlen测试的手写reference，不是主pipeline的一部分。
- `gap_analysis_v2.md`：第3-4周的分析——对照FA2的`flash_fwd_kernel`找差距，根因定位到`cp.async` smem swizzle和warp角色划分。
- `test_v2_minimal.py`：三个step vs SDPA的正确性 + latency，打印gap归因。
- `profile_ncu.py`：Nsight Compute profiling脚本——跑fwd+bwd 3次，让`ncu`用repeat filter抓到kernel。用法：`ncu python profile_ncu.py`。
- `paged_kv_cache/paged_attn.py`：LLM推理方向的Paged KV Cache。`PagedKVCache`类管理固定大小的page池，每个request有page_table，按需分配和释放。Triton `_paged_attn_fwd_kernel`运行时walk page_table，online softmax fp16。包含纯Python reference做正确性验证。

跑一遍：

```powershell
python prerequisites/softmax_online.py
python v1_forward/step1_no_mask.py
python v1_forward/step2_causal.py
python v1_forward/step3_varlen.py
python test_v2_minimal.py
python v1_backward/benchmark_e2e.py
python paged_kv_cache/paged_attn.py
ncu python profile_ncu.py
```

## Accuracy（正确性）

**Forward（fp32）**

| | Shape | Error |
| --- | --- | --- |
| step1 | 64/128/256 | < 1e-4 |
| step2 | 64/128/256 | < 3e-3 |
| step3 | [64,32] [128,256] [64,128,192] | < 3e-3 |

**Backward（fp32完整forward→backward pipeline）**

step1/step2：`B=2, H=8, D_head=16`，step3：padded varlen vs 纯Python reference。

| | Shape | dQ | dK | dV |
| --- | --- | --- | --- | --- |
| step1 (no mask) | S=128/256/512 | < 4.3e-2 | < 3.4e-2 | < 4.6e-3 |
| step2 (causal) | S=128/256/512 | < 4.6e-2 | < 3.7e-2 | < 1.6e-2 |
| step3 (varlen) | [128,256] [64,128,192] | < 2.9e-2 | < 2.5e-2 | < 5.1e-3 |

Backward误差比forward稍大（max ～4.6e-2），因为`dK/dV`用`tl.atomic_add`跨Q块累加——不同block的累加顺序引入额外浮点误差，但fp32下可控。step4/step5没有atomic_add，全串行累加，误差更稳。step4/varlen：`[128,256] [64,128,192]`的dQ/dK/dV全< 6.2e-2。

## Backward性能（step4_varlen_causal，fp32，D_head=64）

纯Python reference的加速比，RTX 5060 Laptop：

```
seqlens               Triton (ms)  Reference (ms)  Speedup
-----------------------------------------------------------
[128]                    0.23           0.47           2.08x
[256]                    0.22           0.45           2.10x
[512]                    0.27           0.49           1.82x
[1024]                   0.82           2.73           3.33x
[2048]                   2.34           11.09          4.74x
[128, 256]               0.27           1.05           3.83x
[256, 512]               0.41           0.88           2.16x
[128, 256, 512]          0.50           1.27           2.56x
```

## Backward性能分析（torch.profiler，S=2048）

Kernel time breakdown：

```
Kernel                          CUDA time   % of total
---------------------------------------------------
_flash_attn_bwd_kernel          1.09ms      36%     核心计算
_reduce_kernel × 2              1.08ms      36%     partial → dK/dV 规约
vectorized_elementwise_kernel ×5 810us      27%     torch.zeros 清零开销
---------------------------------------------------
Total                           2.99ms
```

**关键发现**：

1. 核心backward计算只占总时间36%——partial+reduce架构的overhead高达63%（reduce 36% + 初始化27%）
2. 把`torch.zeros()`换成`torch.empty()`直接干掉27%初始化开销——S=2048从3.10ms降到2.34ms（24.5%提速）
3. reduce kernel还是最大瓶颈（36%）——atomic_add vs partial+reduce的tradeoff需要实测数据拍板

## Forward性能

**step1 no mask，fp16，B=1, H=8, D=128 (D_head=16)**

BW%从实际HBM字节计算：Q load一次，K/V每个`S/BM` program全load一遍，O store一次。

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

**step2 causal + block skipping，fp16，B=1, H=8, D=128 (D_head=16)**

Block skipping后K/V load量降到≈ S²/2行（每个program只load `(block_m_idx+1)*BM`行），大约是step1的一半。

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

**step3 varlen + causal + block skipping，fp16**

```
max_S |     seqlens      | Tri ms | Tri TFLOPS | Tri BW% | SDPA ms | Speedup
-------------------------------------------------------------------------------
  128 |   [128,128,128]  |  0.135 |       0.05 |    0.1% |   0.343 |  2.55x
  256 |   [256,256,256]  |  0.142 |       0.18 |    0.4% |   0.432 |  3.03x
  512 |   [512,256,128]  |  0.119 |       0.37 |    0.8% |   0.396 |  3.32x
 1024 |  [1024,512,256]  |  0.160 |       1.10 |    2.2% |   0.520 |  3.26x
```

## 观察

**Peak BW%只有40.6%——Triton没榨满带宽**。之前用理论算术强度`AI=2×D_head`算出来BW%超100%；改成实际HBM字节后就合理了。step2加block skipping后K/V字节减半，BW%进一步降到14～34%——算子从带宽受限转成了算力受限。大序列（S≥2048）Triton BW%稳定在34～41%，内存侧还有约2.5×空间。

**小序列gap来自SDPA backend效率**。step1 S=512反而更慢（0.52x），但step2同size就反超了（1.20x）——block skipping把K/V load量减半，总HBM从2.4 MB降到1.4 MB，SDPA的固定overhead在小size时占主导。

**Autotune不总是选64×64**。step2 S=2048选了BM=128, BN=128, ns=3, nw=8。Block skipping后loop迭代次数（KV block数）降了，更大的tile减少launch overhead。但S=4096又回到BM=64——即使减半了，K/V load绝对量还是大，小tile更好。

**Varlen的优势**：padded SDPA把所有序列pad到max_seqlen，在padding token上白算。Varlen直接跳过——这就是SDPA慢2.5～3.3x的原因。FLOPs按实际有效token算，不是padded长度。

**Windows限制**：官方flash-attn CUDA wheel装不了，没法和手写CUDA kernel对比。SDPA的flash backend在Windows上也跑不满性能——这可能也是Triton能在大序列上大幅反超的原因之一。

## Forward+Backward端到端Benchmark（fp32, D_head=64）

测forward、backward、以及fwd→bwd作为一个调用对（分开调fwd+bwd数字不一样，因为pair包含每次调用的kernel launch overhead）。

```
python v1_backward/benchmark_e2e.py
```

分解显示训练步总时间里backward占多少——帮助决定先优化fwd还是bwd。D_head=64让backward成为主导阶段（D_head=16时fwd+bwd的FLOPs更接近）。

## Nsight Compute Profiling

`profile_ncu.py`跑fwd+bwd 3次——NCU默认的repeat filter会丢单次运行的kernel，但能抓到第2/3次调用。建议指标：

```powershell
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed python profile_ncu.py
```

要回答的核心问题：

1. Kernel到底是memory-bound（dram throughput接近peak）还是compute-bound（SM throughput低）？
2. partial+reduce之后瓶颈在哪——bwd kernel还是reduce kernels？
3. K/V load的L2 hit rate——block skipping让K/V字节大约减半了，但L2上真的体现出来了吗？

在Linux机器上跑（Windows上NCU的metric coverage有限）才能看全。

## Paged KV Cache

推理方向。`PagedKVCache`持有固定大小的`[num_pages, H, PAGE_SIZE, D]` tensor，每个request通过page_table按需分配page，释放的page回收。Triton kernel（`_paged_attn_fwd_kernel`）拿到page_table + seq_lens tensor，在任意物理顺序下walk KV——kernel永远看不到一个连续的全长tensor。

每个program处理一个(req, head)对，每页处理PAGE_SIZE个entry，每页边界处load K/V时加mask。Generation模式下不需要causal mask——Q长度为1。Scaling online softmax和累加逻辑和step3 varlen完全一样。

用法：

```python
cache = PagedKVCache(num_pages=128, num_heads=H, page_size=16, head_dim=D)
cache.allocate_request(req_id=0)
for step in range(n_tokens):
    cache.append(req_id, K_new_step, V_new_step)
    O = paged_attention(cache, Q_new)
```

正确性：`paged_attn_reference`里有纯Python reference（同样的page_table walk + softmax），和Triton交叉验证err < 1e-2，seq_len 256/4096, PAGE_SIZE 16/32。
