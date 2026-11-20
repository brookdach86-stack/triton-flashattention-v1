# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 基于 README 中实测数据生成展示图表

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C_BASE = "#4C72B0"
C_ACC = "#DD8452"
C_OK = "#2E7D32"

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "DejaVu Sans"],
    "font.size": 11,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------
# 图 1：Forward speedup vs PyTorch SDPA（step1 / step2）
# ---------------------------------------------------------------
S = [512, 1024, 2048, 4096]
s1 = [0.52, 3.64, 4.53, 3.89]
s2 = [1.20, 2.37, 3.84, 4.11]

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(S, s1, marker="o", markersize=7, linewidth=2, color=C_BASE,
        markerfacecolor="white", markeredgewidth=2, label="step1 no-mask")
ax.plot(S, s2, marker="s", markersize=7, linewidth=2, color=C_ACC,
        markerfacecolor="white", markeredgewidth=2, label="step2 causal + block-skip")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="parity (1x)")
for x, y in zip(S, s1):
    ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                xytext=(-16, 10) if y < 1 else (0, 9),
                ha="center", fontsize=8.5, color=C_BASE)
for x, y in zip(S, s2):
    ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                xytext=(18, 8) if y < 2 else (0, -14),
                ha="center", fontsize=8.5, color=C_ACC)
ax.set_xscale("log", base=2)
ax.set_xticks(S)
ax.set_xticklabels([str(s) for s in S])
ax.set_xlabel("Sequence length")
ax.set_ylabel("Speedup vs PyTorch SDPA")
ax.set_title("FlashAttention Forward: Triton vs SDPA (fp16, B=1 H=8 D_head=16)",
             fontweight="bold")
ax.set_ylim(0, 5.6)
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.text(0.55, 0.30, "block skipping 让 K/V load 减半\n大序列稳定 3.8~4.1x", transform=ax.transAxes,
        fontsize=8.5, color=C_ACC, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec=C_ACC, alpha=0.9))
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_forward_speedup.png"), bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------
# 图 2：Backward speedup + kernel time breakdown
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))

seqs = ["[128]", "[256]", "[512]", "[1024]", "[2048]", "[128,256]", "[256,512]", "[128,256,512]"]
sp = [2.08, 2.10, 1.82, 3.33, 4.74, 3.83, 2.16, 2.56]
colors = [C_BASE] * len(seqs)
colors[4] = C_OK
colors[5] = C_OK
bars = axes[0].barh(range(len(seqs)), sp, color=colors, height=0.6)
axes[0].set_yticks(range(len(seqs)))
axes[0].set_yticklabels(seqs, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlabel("Speedup vs Python reference")
axes[0].set_title("Backward Speedup (step4 varlen)", fontweight="bold", fontsize=10.5)
axes[0].set_xlim(0, 5.6)
for b, v in zip(bars, sp):
    axes[0].text(v + 0.1, b.get_y() + b.get_height()/2, f"{v:.2f}x", va="center", fontsize=8.5,
                 fontweight="bold")

kernels = ["_flash_attn_bwd\n(core compute)", "_reduce_kernel x2\n(partial->dK/dV)",
           "elementwise x5\n(torch.zeros init)"]
pct = [36, 36, 27]
colors2 = [C_OK, C_ACC, "#B0BEC5"]
wedges, _ = axes[1].pie(pct, colors=colors2, startangle=90,
                        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
axes[1].text(0, 0, "2.99 ms\nS=2048", ha="center", va="center", fontsize=11, fontweight="bold")
axes[1].legend(wedges, [f"{k}  {p}%" for k, p in zip(kernels, pct)],
               fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.05))
axes[1].set_title("Kernel Time Breakdown", fontweight="bold", fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_backward.png"), bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------
# 图 0：架构图
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.0)
ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07,rounding_size=0.13",
                                facecolor=fc, edgecolor=ec, linewidth=1.5))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontweight="bold")


def arrow(x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.4, color=color))


box(0.2, 2.1, 1.5, 1.0, "Q / K / V\n\n[B,H,S,D]", "#ECEFF1", "#90A4AE", fs=9)
arrow(1.7, 2.6, 2.15, 2.6)

ax.add_patch(FancyBboxPatch((2.15, 0.75), 8.65, 3.6,
                            boxstyle="round,pad=0.08,rounding_size=0.18",
                            facecolor="#FAFCFF", edgecolor=C_BASE, linewidth=1.7, alpha=0.5))
ax.text(2.35, 4.18, "Triton FlashAttention v1 (Forward + Backward)", fontsize=9.5,
        fontweight="bold", color=C_BASE)

box(2.35, 3.15, 1.75, 0.85, "Forward\n\nonline softmax\nno HBM roundtrip", "#E8F5E9", C_OK, fs=8)
box(2.35, 1.95, 1.75, 0.85, "step1 no-mask\nstep2 causal+skip\nstep3 varlen", "#E3F2FD", C_BASE, fs=8)
box(2.35, 0.95, 1.75, 0.75, "Paged KV Cache\npage_table walk", "#F3E5F5", "#7B1FA2", fs=8)

box(4.45, 3.15, 1.85, 0.85, "Backward\n\n5-step chain\nstep1 -> step5", "#FFF3E0", C_ACC, fs=8)
box(4.45, 1.95, 1.85, 0.85, "atomic_add\n-> partial+reduce\n(DRAM 90% -> 60%)", "#FFF3E0", C_ACC, fs=8)
box(4.45, 0.95, 1.85, 0.75, "split-KV grid\n(Q_blk, KV_blk, B*H)\nmax parallelism", "#FFF8E1", "#F9A825", fs=7.5)

box(6.65, 3.15, 1.75, 0.85, "M / L 输出\n(per-row max\n+ softmax denom)", "#EDE7F6", "#5E35B1", fs=8)
box(6.65, 1.95, 1.75, 0.85, "O(1) memory\nrecompute P\nin backward", "#EDE7F6", "#5E35B1", fs=8)
box(6.65, 0.95, 1.75, 0.75, "correctness\n< 1e-4 fwd\n< 6.2e-2 bwd", "#E8F5E9", C_OK, fs=7.5)

arrow(8.6, 2.35, 8.85, 2.35)
box(8.85, 1.7, 1.75, 1.3, "PyTorch SDPA\nbaseline\n\nspeedup\n2~4.7x", "#ECEFF1", "#90A4AE", fs=8)

ax.text(5.5, 4.82, "Triton FlashAttention v1: Forward + Backward", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#263238")
ax.text(5.5, 4.52, "from online softmax to split-KV backward, on RTX 5060 Laptop (sm_120)",
        ha="center", va="center", fontsize=9, color="#546E7A", style="italic")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig0_architecture.png"), bbox_inches="tight", dpi=170)
plt.close(fig)

print("已生成:")
for f in sorted(os.listdir(FIG_DIR)):
    print(f"  {f}  ({os.path.getsize(os.path.join(FIG_DIR, f))} bytes)")