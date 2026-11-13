# @Author: Wain (brookdach86-stack@users.noreply.github.com)
# @GitHub: https://github.com/brookdach86-stack
# @Note: 本代码为原创，转载请注明来源。基于学习路径迭代：CSAPP Lab → CUDA Kernel → Triton → LLM Inference
import torch

torch.manual_seed(42)

N, D_MODEL = 8, 4
BLOCK = 2

Q = torch.randn(N, D_MODEL)
K = torch.randn(N, D_MODEL)
V = torch.randn(N, D_MODEL)

scale = D_MODEL ** -0.5

ref_score = Q @ K.T * scale
ref_attn = torch.softmax(ref_score, dim=-1)
ref_O = ref_attn @ V

print("=== 参考结果（torch 全矩阵计算）===")
print(f"O[0]: {ref_O[0]}")
print()

print("=== Forward：分块 online-softmax（block_size=2）===")

O = torch.zeros(N, D_MODEL)
m = torch.full((N,), float('-inf'))
l = torch.zeros(N)

for j_start in range(0, N, BLOCK):
    j_end = min(j_start + BLOCK, N)
    Kj = K[j_start:j_end]
    Vj = V[j_start:j_end]

    Sij = Q @ Kj.T * scale
    mij = Sij.max(dim=-1).values

    m_new = torch.maximum(m, mij)
    alpha = torch.exp(m - m_new)
    p = torch.exp(Sij - m_new.unsqueeze(-1))

    l = alpha * l + p.sum(dim=-1)
    O = alpha.unsqueeze(-1) * O + p @ Vj
    m = m_new

O_final = O / l.unsqueeze(-1)

print(f"O[0]: {O_final[0]}")
print(f"Max error: {(ref_O - O_final).abs().max().item():.2e}")
print("✅ Forward match!" if (ref_O - O_final).abs().max().item() < 1e-5 else "❌ Mismatch")

print()
print("=== Backward：分块重算 P，倒推梯度 ===")

dO = torch.randn(N, D_MODEL)
dQ = torch.zeros(N, D_MODEL)
dK = torch.zeros(N, D_MODEL)
dV = torch.zeros(N, D_MODEL)

Di = (dO * ref_O).sum(dim=-1)

for j_start in range(0, N, BLOCK):
    j_end = min(j_start + BLOCK, N)
    Kj = K[j_start:j_end]
    Vj = V[j_start:j_end]

    Sij = Q @ Kj.T * scale
    p = torch.exp(Sij - m.unsqueeze(-1)) / l.unsqueeze(-1)

    dV[j_start:j_end] += p.T @ dO

    dP = dO @ Vj.T
    dS = p * (dP - Di.unsqueeze(-1))

    dQ += dS @ Kj
    dK[j_start:j_end] += dS.T @ Q

ref_dV = ref_attn.T @ dO
ref_dP = dO @ V.T
ref_dS = ref_attn * (ref_dP - (ref_dP * ref_attn).sum(dim=-1, keepdim=True))
ref_dQ = ref_dS @ K
ref_dK = ref_dS.T @ Q

print(f"dQ max error: {(ref_dQ - dQ).abs().max().item():.2e}")
print(f"dK max error: {(ref_dK - dK).abs().max().item():.2e}")
print(f"dV max error: {(ref_dV - dV).abs().max().item():.2e}")
all_ok = (ref_dQ - dQ).abs().max().item() < 1e-5 and \
         (ref_dK - dK).abs().max().item() < 1e-5 and \
         (ref_dV - dV).abs().max().item() < 1e-5
print("✅ Backward match!" if all_ok else "❌ Mismatch")