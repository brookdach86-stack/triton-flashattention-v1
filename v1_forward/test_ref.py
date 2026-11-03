import torch

def flash_attn_reference(Q, K, V, cu_seqlens):
    scale = Q.shape[-1] ** -0.5
    O = torch.zeros_like(Q)

    for i in range(len(cu_seqlens)-1):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i+1].item()
        seqlen = end - start
        print(f'  Processing seq {i}: start={start}, end={end}, seqlen={seqlen}')

        q = Q[start:end]
        k = K[start:end]
        v = V[start:end]

        print(f'  q.shape={q.shape}, k.shape={k.shape}')

        mask = torch.triu(torch.ones(seqlen, seqlen, device=Q.device), diagonal=1).bool()
        score = torch.matmul(q,k.transpose(-2,-1))*scale
        print(f'  score[0,:3]={score[0,:3]}')

        attn = torch.softmax(score, dim=-1)
        print(f'  attn[0,:3]={attn[0,:3]}')

        out = torch.matmul(attn,v)
        print(f'  out[0,:3]={out[0,:3]}')

        O[start:end] = out

    return O

torch.manual_seed(42)
D = 4
S = 3

Q = torch.randn(S, D, device='cuda')
K = torch.randn(S, D, device='cuda')
V = torch.randn(S, D, device='cuda')
cu_seqlens = torch.tensor([0, S], dtype=torch.int32).cuda()

print('=== Running reference function ===')
ref = flash_attn_reference(Q, K, V, cu_seqlens)

print('\n=== Manual calculation ===')
scale = D ** -0.5
score = torch.matmul(Q, K.T) * scale
print(f'score[0,:3]={score[0,:3]}')
mask = torch.triu(torch.ones(S, S, device='cuda'), diagonal=1).bool()
score_masked = score.masked_fill(mask, float('-inf'))
attn = torch.softmax(score_masked, dim=-1)
print(f'attn[0,:3]={attn[0,:3]}')
out = torch.matmul(attn, V)
print(f'out[0,:3]={out[0,:3]}')

print('\n=== Comparison ===')
print('Ref:', ref)
print('Manual:', out)