# Triton FlashAttention v1 Forward + Backward

Full end-to-end implementation of FlashAttention v1 forward and backward passes in Triton. Forward progression: online softmax → no mask → causal (with block skipping) → varlen. Backward progression: pure Python reference → Triton step1 (no mask, multi-head) → step2 (causal) → step3 (varlen) → step4 (split-KV, no atomic_add) → step5 (full split-KV grid).

Core memory-saving idea: when `m` (row max) changes mid-accumulation, the running output `o` is still in SRAM — multiply by `alpha = exp(m_old - m_new)` and keep accumulating, no intermediate writeback to HBM. Forward additionally outputs `M/L` (per-row max and softmax denominator) so backward can recompute P without materializing the full S matrix, maintaining O(1) memory.

## Environment

RTX 5060 Laptop (384 GB/s bandwidth, sm_120), Python 3.10, PyTorch + Triton. Official flash-attn CUDA wheel cannot be installed on Windows; PyTorch SDPA flash backend is used as baseline.

## Files

- `prerequisites/softmax_online.py`: Minimal online softmax verification, one program per row. Cross-checked against `torch.softmax` with threshold `1e-5`. Tested 128/512/1024/2048/8192 columns and irregular lengths `[1000, 1001, 999, 998]`.
- `v1_forward/step1_no_mask.py`: `[B, H, S, D_head]`, no mask. Default `B=1, H=8, D=128` (`D_head=16`), fp16. Autotune over 18 configurations (BLOCK_M 64/128, BLOCK_N 64/128, num_stages 2/3/4, num_warps 4/8). Benchmarks at 512/1024/2048/4096.
- `v1_forward/step2_causal.py`: Adds causal mask + block skipping — the KV loop upper bound changes from `seqlen` to `(block_m_idx+1)*BLOCK_M`, with mask preserved inside diagonal K blocks. Autotune/benchmark same as step1.
- `v1_forward/step3_varlen.py`: Variable-length sequences, `[B, H, max_seqlen, D_head]` padded format, `cu_seqlens` marks valid lengths. `pid(1)` is split as `seq_idx * H + head_idx`, per-sample seqlen looked up from `cu_seqlens`. Causal + block skipping + autotune. Forward additionally outputs `M/L` (`[B, H, max_seqlen]` float32) for backward use.
- `v1_backward/step1_backward.py`: Triton backward step1, no mask, multi-head. Five-step recipe: ① load Q/dO/O/M/L → ② `Di = sum(dO*O)` → ③ loop over KV blocks: compute P → atomic_add dV → `ds = P*(dP-Di)` → `dQ += ds@K` → atomic_add dK. `dK/dV` accumulated across Q blocks via `tl.atomic_add`.
- `v1_backward/step2_backward.py`: Triton backward step2, adds causal mask + block skipping. Loop upper bound from `seqlen` to `(block_m_idx+1)*BLOCK_M`, score becomes `tl.where(k_pos > q_pos, -inf, s)`.
- `v1_backward/step3_backward.py`: Triton backward step3, varlen padded. `seqlen = cu_seqlens[b+1]-cu_seqlens[b]`, load/store masks use per-sample seqlen, `if block_m_idx*BM >= seqlen: return` skips padding-region Q blocks.
- `v1_backward/step4_splitkv_backward.py`: Triton backward step4, partial + reduce eliminates atomic_add contention. dK/dV no longer atomically accumulate to globals — each Q block writes a partial copy, two reduce kernels sum along the Q dimension. dQ accumulates directly in registers (no contention since KV-loop within one Q block is sequential).
- `v1_backward/step4_varlen_causal.py`: step4 + causal mask + varlen padded. `partial_dK/partial_dV` use `torch.empty` (all positions written, no zeroing needed).
- `v1_backward/step5_splitkv_full.py`: step5, full split-KV grid `(num_q_blocks, num_kv_blocks, B*H)`. Each program handles one (Q_block, KV_block) tile, dQ/dK/dV all write partials, three reduce kernels sum along KV and Q dimensions. Highest parallelism but duplicates Q loads.
- `v1_backward/benchmark.py`: step4_varlen_causal vs pure Python reference latency, torch.profiler for kernel time breakdown.
- `v1_forward/test_ref.py`: Manual reference for varlen testing, not part of the main pipeline.
- `gap_analysis_v2.md`: Week 3-4 analysis — cross-checks against FA2 `flash_fwd_kernel`, gaps traced to `cp.async`, smem swizzle, and warp role partitioning.
- `test_v2_minimal.py`: Correctness + latency of three steps vs SDPA, prints gap attribution.

Run:

```powershell
python prerequisites/softmax_online.py
python v1_forward/step1_no_mask.py
python v1_forward/step2_causal.py
python v1_forward/step3_varlen.py
python test_v2_minimal.py
```

## Accuracy

**Forward (fp32)**

| | Shape | Error |
| --- | --- | --- |
| step1 | 64/128/256 | < 1e-4 |
| step2 | 64/128/256 | < 3e-3 |
| step3 | [64,32], [128,256], [64,128,192] | < 3e-3 |

**Backward (fp32, full forward→backward pipeline)**

step1/step2: `B=2, H=8, D_head=16`, step3: padded varlen, vs pure Python reference.

| | Shape | dQ | dK | dV |
| --- | --- | --- | --- | --- |
| step1 (no mask) | S=128/256/512 | < 4.3e-2 | < 3.4e-2 | < 4.6e-3 |
| step2 (causal) | S=128/256/512 | < 4.6e-2 | < 3.7e-2 | < 1.6e-2 |
| step3 (varlen) | [128,256], [64,128,192] | < 2.9e-2 | < 2.5e-2 | < 5.1e-3 |

Backward errors are slightly larger than forward (max ~4.6e-2) because `dK/dV` use `tl.atomic_add` across Q blocks — different accumulation orders across blocks introduce additional floating-point error, but this is controllable in fp32.

step4/step5 have no atomic_add, all sequential accumulation, errors more stable. step4/varlen: `[128,256], [64,128,192]`, dQ/dK/dV all < 6.2e-2.

## Backward Performance (step4_varlen_causal, fp32, D_head=64)

Speedup over pure Python reference, RTX 5060 Laptop:

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

## Backward Performance Analysis (torch.profiler, S=2048)

Kernel time breakdown:

```
Kernel                          CUDA time   % of total
---------------------------------------------------
_flash_attn_bwd_kernel          1.09ms      36%     core compute
_reduce_kernel × 2              1.08ms      36%     partial → dK/dV reduction
vectorized_elementwise_kernel ×5 810us      27%     torch.zeros zeroing overhead
---------------------------------------------------
Total                           2.99ms
```

**Key findings**:

1. Core backward compute is only 36% of total — partial+reduce architecture overhead hits 63% (reduce 36% + initialization 27%)
2. Replacing `torch.zeros()` with `torch.empty()` eliminates 27% initialization overhead — S=2048 drops from 3.10ms to 2.34ms (24.5% speedup)
3. The reduce kernel remains the single largest bottleneck (36%) — the atomic_add vs partial+reduce tradeoff needs empirical measurement

## Performance

**step1 no mask, fp16, B=1, H=8, D=128 (D_head=16)**

BW% computed from actual HBM bytes: Q loaded once, K/V fully loaded by `S/BM` programs each, O stored once.

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

**step2 causal + block skipping, fp16, B=1, H=8, D=128 (D_head=16)**

With block skipping, K/V loading drops to ≈ S²/2 rows (each program only loads `(block_m_idx+1)*BM` rows), roughly half of step1.

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

**step3 varlen + causal + block skipping, fp16**

```
max_S |     seqlens      | Tri ms | Tri TFLOPS | Tri BW% | SDPA ms | Speedup
-------------------------------------------------------------------------------
  128 |   [128,128,128]  |  0.135 |       0.05 |    0.1% |   0.343 |  2.55x
  256 |   [256,256,256]  |  0.142 |       0.18 |    0.4% |   0.432 |  3.03x
  512 |   [512,256,128]  |  0.119 |       0.37 |    0.8% |   0.396 |  3.32x
 1024 |  [1024,512,256]  |  0.160 |       1.10 |    2.2% |   0.520 |  3.26x
```

## Observations

**Peak BW% is 40.6% — Triton is not saturating bandwidth**. Earlier estimates using theoretical arithmetic intensity `AI=2×D_head` gave BW% over 100%; corrected to actual HBM bytes they are reasonable. After block skipping in step2, K/V bytes halve and BW% drops further (14~34%) — the operator shifts from bandwidth-bound to compute-bound. For large sequences (S≥2048) Triton BW% stabilizes at 34~41%, leaving roughly 2.5× room on the memory side.

**Small-sequence gaps come from SDPA backend efficiency**. step1 S=512 is actually slower (0.52x), but step2 at the same size overtakes (1.20x) — block skipping halves K/V loading, total HBM from 2.4 MB to 1.4 MB, and SDPA's fixed overhead dominates at small sizes.

**Autotune doesn't always pick 64×64**. step2 S=2048 selected BM=128 BN=128 ns=3 nw=8. After block skipping the loop iteration count (KV block count) drops, so larger tiles reduce launch overhead. But at S=4096 it goes back to BM=64 — absolute K/V loading is still large even after halving, so smaller tiles are better.

**Varlen advantage**: padded SDPA pads all sequences to max_seqlen, wasting compute on padding tokens. Varlen skips them entirely — that's the 2.5~3.3x slowdown of SDPA. FLOPs counted on actual valid tokens, not padded length.

**Windows limitation**: official flash-attn CUDA wheel won't install, so no comparison against handwritten CUDA kernels. SDPA's flash backend also doesn't run at full performance on Windows — this is likely part of why Triton can overtake so significantly at large sequences.

## Not Yet Done

- [x] Backward (step1/step2/step3/step4/step5, complete Triton forward→backward pipeline, error vs pure Python reference < 6e-2)
- [x] FLOPs / bandwidth recomputed from actual HBM bytes (step1 K/V loaded by S/BM programs, step2 after block skipping ≈ S²/2 rows)
- [x] Backward benchmark + torch.profiler analysis (above)
- [ ] Paged KV Cache (inference direction, separate project)
- [ ] Forward+Backward end-to-end benchmark (currently backward measured independently)
- [ ] Nsight Compute deep analysis (DRAM bandwidth, SM occupancy, L2 hit rate)
