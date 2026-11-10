import torch
import triton
import triton.language as tl


class PagedKVCache:
    def __init__(self, num_pages, num_heads, page_size, head_dim, dtype=torch.float16, device='cuda'):
        self.num_pages = num_pages
        self.num_heads = num_heads
        self.page_size = page_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.k_cache = torch.empty(num_pages, num_heads, page_size, head_dim, dtype=dtype, device=device)
        self.v_cache = torch.empty(num_pages, num_heads, page_size, head_dim, dtype=dtype, device=device)
        self.free_pages = list(range(num_pages))
        self.page_tables = {}
        self.seq_lens = {}

    def allocate_request(self, req_id):
        self.page_tables[req_id] = []
        self.seq_lens[req_id] = 0

    def free_request(self, req_id):
        for pid in self.page_tables.pop(req_id):
            self.free_pages.append(pid)
        self.seq_lens.pop(req_id)

    def append(self, req_id, k_new, v_new):
        seq_len = self.seq_lens[req_id]
        page_idx = seq_len // self.page_size
        offset = seq_len % self.page_size

        if page_idx >= len(self.page_tables[req_id]):
            if not self.free_pages:
                raise RuntimeError("No free pages")
            self.page_tables[req_id].append(self.free_pages.pop(0))

        page_id = self.page_tables[req_id][page_idx]
        self.k_cache[page_id, :, offset, :] = k_new.to(self.dtype)
        self.v_cache[page_id, :, offset, :] = v_new.to(self.dtype)
        self.seq_lens[req_id] += 1

    def build_page_table_tensor(self, max_seq_len):
        B = len(self.seq_lens)
        max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        pt = torch.full((B, max_pages), -1, dtype=torch.int32, device=self.device)
        for i, (req_id, pages) in enumerate(self.page_tables.items()):
            for j, pid in enumerate(pages):
                if j < max_pages:
                    pt[i, j] = pid
        return pt

    def get_seq_lens_tensor(self):
        return torch.tensor(list(self.seq_lens.values()), dtype=torch.int32, device=self.device)


@triton.jit
def _paged_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    page_table_ptr, seq_lens_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kh, stride_kn, stride_kk,
    stride_vp, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_od,
    stride_ptb, stride_ptn,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    req = tl.program_id(0)
    head = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + req)
    if seq_len == 0:
        return

    d_offs = tl.arange(0, HEAD_DIM)
    q = tl.load(Q_ptr + req * stride_qb + head * stride_qh + d_offs * stride_qd).to(tl.float32)

    m = float('-inf')
    l = 0.0
    o = tl.zeros([HEAD_DIM], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    n_pages = tl.cdiv(seq_len, PAGE_SIZE)

    for pi in range(0, n_pages):
        page_id = tl.load(page_table_ptr + req * stride_ptb + pi * stride_ptn)
        if page_id < 0:
            continue
        slot = tl.arange(0, PAGE_SIZE)
        pos = pi * PAGE_SIZE + slot
        valid = pos < seq_len

        k_ptrs = K_ptr + page_id * stride_kp + head * stride_kh + slot[:, None] * stride_kn + d_offs[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)

        score = tl.sum(q[None, :] * k, axis=1) * scale
        score = tl.where(valid, score, float('-inf'))

        m_new = tl.maximum(m, tl.max(score))
        alpha = tl.exp(m - m_new)
        p = tl.exp(score - m_new)

        v_ptrs = V_ptr + page_id * stride_vp + head * stride_vh + slot[:, None] * stride_vn + d_offs[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)

        l = alpha * l + tl.sum(p)
        o = alpha * o + tl.sum(p[:, None] * v, axis=0)
        m = m_new

    o = o / l
    tl.store(O_ptr + req * stride_ob + head * stride_oh + d_offs * stride_od,
             o.to(O_ptr.dtype.element_ty))


def paged_attention(cache, Q_new):
    B, H, D = Q_new.shape
    O = torch.empty(B, H, D, dtype=cache.dtype, device=cache.device)

    max_seq_len = max(cache.seq_lens.values())
    page_table = cache.build_page_table_tensor(max_seq_len)
    seq_lens = cache.get_seq_lens_tensor()

    _paged_attn_fwd_kernel[(B, H)](
        Q_new, cache.k_cache, cache.v_cache, O,
        page_table, seq_lens,
        Q_new.stride(0), Q_new.stride(1), Q_new.stride(2),
        cache.k_cache.stride(0), cache.k_cache.stride(1), cache.k_cache.stride(2), cache.k_cache.stride(3),
        cache.v_cache.stride(0), cache.v_cache.stride(1), cache.v_cache.stride(2), cache.v_cache.stride(3),
        O.stride(0), O.stride(1), O.stride(2),
        page_table.stride(0), page_table.stride(1),
        PAGE_SIZE=cache.page_size, HEAD_DIM=D,
    )
    return O


def paged_attention_reference(cache, Q_new):
    B, H, D = Q_new.shape
    scale = D ** -0.5
    O = torch.empty(B, H, D, dtype=torch.float32, device=cache.device)

    for i, (req_id, seq_len) in enumerate(cache.seq_lens.items()):
        pages = cache.page_tables[req_id]
        n_pages = (seq_len + cache.page_size - 1) // cache.page_size

        K_list, V_list = [], []
        for pi in range(n_pages):
            page_id = pages[pi]
            n_valid = min(seq_len - pi * cache.page_size, cache.page_size)
            K_list.append(cache.k_cache[page_id, :, :n_valid, :].float())
            V_list.append(cache.v_cache[page_id, :, :n_valid, :].float())

        K_full = torch.cat(K_list, dim=1) if K_list else torch.empty(H, 0, D, device=cache.device)
        V_full = torch.cat(V_list, dim=1) if V_list else torch.empty(H, 0, D, device=cache.device)

        q = Q_new[i].float()
        if seq_len == 0:
            O[i] = 0
            continue
        scores = torch.einsum('hd,hsd->hs', q, K_full) * scale
        weights = torch.softmax(scores, dim=-1)
        O[i] = torch.einsum('hs,hsd->hd', weights, V_full)

    return O.to(cache.dtype)


if __name__ == "__main__":
    torch.manual_seed(42)

    H = 8
    D = 64
    PAGE_SIZE = 16
    NUM_PAGES = 64

    cache = PagedKVCache(NUM_PAGES, H, PAGE_SIZE, D, dtype=torch.float16)
    cache.allocate_request(0)
    cache.allocate_request(1)

    for _ in range(40):
        k = torch.randn(H, D, dtype=torch.float16, device='cuda')
        v = torch.randn(H, D, dtype=torch.float16, device='cuda')
        cache.append(0, k, v)

    for _ in range(25):
        k = torch.randn(H, D, dtype=torch.float16, device='cuda')
        v = torch.randn(H, D, dtype=torch.float16, device='cuda')
        cache.append(1, k, v)

    Q_new = torch.randn(2, H, D, dtype=torch.float16, device='cuda')

    tri = paged_attention(cache, Q_new)
    ref = paged_attention_reference(cache, Q_new)

    for i in range(2):
        err = (ref[i].float() - tri[i].float()).abs().max().item()
        print(f"req{i} seq_len={cache.seq_lens[i]}  max_err={err:.2e}  {'PASS' if err < 1e-2 else 'FAIL'}")

    print(f"\nreq0 pages={cache.page_tables[0]}")
    print(f"req1 pages={cache.page_tables[1]}")
    print(f"free_pages={len(cache.free_pages)}/{NUM_PAGES}")