"""GPU tests: Nenya's Triton kernels against their PyTorch references.

    pytest -q tests/test_kernels.py
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda"


def test_rms_norm_fp32w():
    from nenya_vllm.layers.mhc import rms_norm_fp32
    from nenya_vllm.ops.small_kernels import rms_norm_fp32w

    torch.manual_seed(0)
    x = torch.randn(37, 960, device=DEV, dtype=torch.bfloat16) * 3
    w = torch.rand(960, device=DEV) + 0.5
    ref = rms_norm_fp32(x, w, 1e-5)
    got32 = rms_norm_fp32w(x, w, 1e-5, torch.float32)
    got16 = rms_norm_fp32w(x, w, 1e-5, torch.bfloat16)
    assert torch.allclose(got32, ref, rtol=1e-5, atol=1e-5)
    assert torch.equal(got16, ref.to(torch.bfloat16)) or (got16.float() - ref).abs().max() < 2e-2
    # strided rows (a column slice of a wider tensor)
    wide = torch.randn(11, 400, device=DEV, dtype=torch.bfloat16)
    got = rms_norm_fp32w(wide[:, :320], w[:320], 1e-6, torch.bfloat16)
    assert (got.float() - rms_norm_fp32(wide[:, :320], w[:320], 1e-6)).abs().max() < 2e-2


def test_gdn2_gates():
    from nenya_vllm.ops.small_kernels import gdn2_gates

    proj = torch.randn(9, 7936, device=DEV, dtype=torch.bfloat16) * 4
    kd = 1536
    b, w = gdn2_gates(proj, 3 * kd, 4 * kd, kd)
    assert torch.equal(b, torch.sigmoid(proj[:, 3 * kd:4 * kd]))
    assert torch.equal(w, torch.sigmoid(proj[:, 4 * kd:5 * kd]))


def test_moe_route():
    from nenya_vllm.ops.small_kernels import moe_route

    torch.manual_seed(1)
    logits = torch.randn(64, 120, device=DEV) * 2
    bias = torch.randn(120, device=DEV) * 0.5
    w, ids = moe_route(logits, bias, 6)
    ref_ids = torch.topk(logits + bias, 7, dim=-1).indices[..., :6]
    ref_w = torch.softmax(torch.gather(logits, -1, ref_ids), dim=-1)
    assert torch.equal(ids.long(), ref_ids)
    assert torch.allclose(w, ref_w, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mhc_kernels_vs_reference(dtype, monkeypatch):
    import nenya_vllm.layers.mhc as mhc

    torch.manual_seed(2)
    m = mhc.NenyaMHC(4, 960, 20).to(DEV)
    with torch.no_grad():
        m.theta.weight.normal_(0, 0.02)
        m.alpha_pre.fill_(0.7); m.alpha_post.fill_(0.4); m.alpha_res.fill_(3.0)
        m.beta_pre.normal_(); m.beta_post.normal_(); m.beta_res.normal_(0, 3.0)
    m.finalize()
    x = (torch.randn(23, 4, 960, device=DEV) * 2).to(dtype)
    w = torch.rand(960, device=DEV) + 0.5
    layer = torch.randn(23, 960, device=DEV, dtype=torch.bfloat16)
    monkeypatch.setattr(mhc, "_IMPL", "torch")
    r32, r16, rcoef = m.pre(x, w, 1e-5)
    rpost = m.post(x, layer, rcoef)
    monkeypatch.setattr(mhc, "_IMPL", "triton")
    k32, k16, kcoef = m.pre(x, w, 1e-5)
    kpost = m.post(x, layer, kcoef)
    assert (k32 - r32).abs().max() < 1e-4
    assert (kcoef - rcoef).abs().max() < 1e-5          # post gates and the Sinkhorn matrix
    # rows come out exactly normalised (the last Sinkhorn step); columns only approximately after 20 iterations
    H = kcoef[:, 4:].view(-1, 4, 4)
    assert (H.sum(-1) - 1).abs().max() < 1e-5
    assert (kpost.float() - rpost.float()).abs().max() < 3e-2


def test_ngram_rows_kernel_vs_reference():
    from nenya_vllm.ops.ngram_rows import nenya_ngram_rows, nenya_ngram_rows_reference

    torch.manual_seed(3)
    eot = 100257
    token_map = torch.randint(0, 62450, (100352,), device=DEV)
    token_map[eot] = 62355
    mult = torch.tensor([121595318837705, 7679631734805, 57058700551945], device=DEV)
    sizes = torch.tensor([275003, 275027, 275039, 275047, 275053, 275059, 275083, 275087,
                          275129, 275131, 275147, 275153, 275159, 275161, 275167, 275183], device=DEV)
    offs = torch.cumsum(torch.cat([sizes.new_zeros(1), sizes[:-1]]), 0)
    # 4 requests: a fresh prompt, a continued chunk, a decode token, one with <|endoftext|> inside
    lens = [7, 5, 1, 9]
    ids = torch.randint(0, 100000, (sum(lens),), device=DEV)
    ids[13 + 3] = eot
    qsl = torch.tensor([0, 7, 12, 13, 22] + [22] * 4, device=DEV, dtype=torch.int32)   # padded to 8 requests
    ctx = torch.full((8, 2), eot, device=DEV, dtype=torch.int32)
    ctx[1] = torch.tensor([11, 22]); ctx[2] = torch.tensor([33, eot]); ctx[3] = torch.tensor([44, 55])
    got = nenya_ngram_rows(ids, qsl, ctx, token_map, mult, sizes, offs, eot, 62355, 8)
    ref = nenya_ngram_rows_reference(ids, qsl, ctx, token_map, mult, sizes, offs, eot, 62355, 8)
    assert torch.equal(got, ref)


def test_moe_route_garbage_rows():
    """CUDA-graph padding rows can hold NaN/inf: ids must stay valid experts and weights finite."""
    from nenya_vllm.ops.small_kernels import moe_route

    logits = torch.randn(4, 120, device=DEV)
    logits[1] = float("nan")
    logits[2, :60] = float("inf")
    logits[3, 7] = float("nan")
    w, ids = moe_route(logits, torch.zeros(120, device=DEV), 6)
    assert int(ids.min()) >= 0 and int(ids.max()) < 120
    assert torch.isfinite(w).all()
    assert torch.allclose(w.sum(-1), torch.ones(4, device=DEV))


@pytest.mark.parametrize("T", [1, 3, 5, 9, 16, 32, 33, 64, 100])
def test_moe_in_route(T):
    """[expert_down | fc1 | fc2] GEMV + fp32 router GEMV + routing in one kernel == bf16 GEMM, torch's fp32 matmul and
    the release's selection (ties aside); counters left at zero."""
    from nenya_vllm.ops.decode_gemv import _COUNTERS
    from nenya_vllm.ops.moe_small import moe_in_route

    torch.manual_seed(4 + T)
    x = torch.randn(T, 960, device=DEV)
    w = torch.randn(120, 960, device=DEV) * 0.05
    bias = torch.randn(120, device=DEV) * 0.1
    w_in = (torch.randn(1920, 960, device=DEV) * 960 ** -0.5).to(torch.bfloat16)
    for _ in range(2):
        lat_in, sh, logits, wts, ids = moe_in_route(x, w, bias, w_in, 6, 384)
    mid = torch.cat([lat_in, sh], dim=1)
    assert torch.allclose(logits.double(), x.double() @ w.double().t(), rtol=1e-5, atol=1e-5)
    logits = x.double() @ w.double().t()
    ref_ids = torch.topk(logits + bias.double(), 7, dim=-1).indices[..., :6]
    ref_w = torch.softmax(torch.gather(logits, -1, ref_ids), dim=-1)
    assert torch.equal(ids.long(), ref_ids)
    assert torch.allclose(wts.double(), ref_w, rtol=1e-5, atol=1e-6)
    ref_mid = x.to(torch.bfloat16).double() @ w_in.double().t()
    tol, atol = (2 ** -8, 1e-6) if T < 5 else (2 ** -7, 1e-4)      # GEMV vs tensor-core accumulation
    assert ((mid.double() - ref_mid).abs() <= tol * ref_mid.abs() + atol).all()
    torch.cuda.synchronize()
    assert all(int(c.abs().sum()) == 0 for c in _COUNTERS.values())


@pytest.mark.parametrize("T", [1, 3])
@pytest.mark.parametrize("gate_first", [True, False])
def test_moe_experts_small(gate_first, T):
    """routed experts (gate/up, SiLU-mul, down, weighted sum) + shared SiLU-mul vs an fp32 PyTorch reference."""
    from nenya_vllm.ops.moe_small import moe_experts_small

    torch.manual_seed(5)
    E, H, LAT, HS, K = 120, 384, 384, 768, 6
    w13 = (torch.randn(E, 2 * H, LAT, device=DEV) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, LAT, H, device=DEV) * 0.05).to(torch.bfloat16)
    mid = torch.randn(T, LAT + 2 * HS, device=DEV).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=DEV)[:K] for _ in range(T)]).to(torch.int32)
    wts = torch.softmax(torch.randn(T, K, device=DEV), dim=-1)
    buf = torch.empty(T, LAT + HS, device=DEV, dtype=torch.bfloat16)
    moe_experts_small(mid[:, :LAT].contiguous(), mid[:, LAT:].contiguous(), ids, wts, w13, w2, gate_first, buf)
    x = mid[:, :LAT].float()
    ref = torch.zeros(T, LAT, device=DEV)
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            g_rows = slice(0, H) if gate_first else slice(H, 2 * H)
            u_rows = slice(H, 2 * H) if gate_first else slice(0, H)
            g = w13[e, g_rows].float() @ x[t]
            u = w13[e, u_rows].float() @ x[t]
            ref[t] += wts[t, k] * (w2[e].float() @ (torch.nn.functional.silu(g) * u))
    shared = torch.nn.functional.silu(mid[:, LAT:LAT + HS].float()) * mid[:, LAT + HS:].float()
    got_r, got_s = buf[:, :LAT].float(), buf[:, LAT:].float()
    assert (got_r - ref).abs().max() <= 1e-2 * ref.abs().max()
    assert (got_r - ref.to(torch.bfloat16).float()).abs().max() <= 2 ** -7 * ref.abs().max()
    assert torch.equal(got_s, shared.to(torch.bfloat16).float())


@pytest.mark.parametrize("T", [9, 16, 33, 64])
@pytest.mark.parametrize("gate_first", [True, False])
def test_moe_experts_grouped(gate_first, T):
    """experts grouped by expert (9..64 tokens): w_out's fp32 input == an fp64 reference of the same maths (fp32
    accumulation, hi + residual MMAs for the down projection); a few experts get > 16 pairs (several chunks)"""
    from nenya_vllm.ops.moe_small import moe_experts_grouped

    torch.manual_seed(6 + T)
    E, H, LAT, HS, K = 120, 384, 384, 768, 6
    w13 = (torch.randn(E, 2 * H, LAT, device=DEV) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, LAT, H, device=DEV) * 0.05).to(torch.bfloat16)
    lat_in = torch.randn(T, LAT, device=DEV).to(torch.bfloat16)
    sh = torch.randn(T, 2 * HS, device=DEV).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, device=DEV)[:K] for _ in range(T)]).to(torch.int32)
    ids[:, 0] = 7                                   # expert 7 in every token: T pairs (several chunks of 16)
    ids[: T // 2, 1] = 3                            # expert 3: T/2 pairs
    for t in range(T):                              # keep each token's experts distinct
        seen = set()
        for k in range(K):
            while int(ids[t, k]) in seen:
                ids[t, k] = (int(ids[t, k]) + 11) % E
            seen.add(int(ids[t, k]))
    wts = torch.softmax(torch.randn(T, K, device=DEV), dim=-1)
    got = moe_experts_grouped(lat_in, sh, ids, wts, w13, w2, gate_first)
    x = lat_in.double()
    ref = torch.zeros(T, LAT, device=DEV, dtype=torch.float64)
    g_rows = slice(0, H) if gate_first else slice(H, 2 * H)
    u_rows = slice(H, 2 * H) if gate_first else slice(0, H)
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k])
            g = w13[e, g_rows].double() @ x[t]
            u = w13[e, u_rows].double() @ x[t]
            ref[t] += wts[t, k].double() * (w2[e].double() @ (torch.nn.functional.silu(g) * u))
    shared = torch.nn.functional.silu(sh[:, :HS].double()) * sh[:, HS:].double()
    assert got.dtype == torch.float32 and got.shape == (T, LAT + HS)
    assert (got[:, :LAT].double() - ref).abs().max() <= 1e-5 * ref.abs().max()
    assert (got[:, LAT:].double() - shared).abs().max() <= 1e-6 * shared.abs().max()
    assert torch.equal(moe_experts_grouped(lat_in, sh, ids, wts, w13, w2, gate_first), got)


def _mhc(seed, res_scale=3.0):
    import nenya_vllm.layers.mhc as mhc

    torch.manual_seed(seed)
    m = mhc.NenyaMHC(4, 960, 20).to(DEV)
    with torch.no_grad():
        m.theta.weight.normal_(0, 0.02)
        m.alpha_pre.fill_(0.7); m.alpha_post.fill_(0.4); m.alpha_res.fill_(3.0)
        m.beta_pre.normal_(); m.beta_post.normal_(); m.beta_res.normal_(0, res_scale)
    m.finalize()
    return m


def _bf16_close(a, b, frac=1e-3, atol=1e-30):
    """bf16 tensors equal up to rare one-ulp differences (fp32 rounding order before the bf16 rounding); atol covers
    results that cancel to near zero (a post's terms are O(1), their fp32 rounding O(1e-7))."""
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    assert (diff <= 2 ** -7 * b.abs() + atol).all(), f"max rel diff {(diff / b.abs().clamp_min(1e-30)).max()}"
    assert (diff > 0).float().mean() <= frac


def _as_matrix(coef):
    return coef


def coef_has_logits(T):
    """True for the token counts that take the fused boundary kernel"""
    from nenya_vllm.ops import mhc_boundary as mb

    return 0 < T <= mb.SMALL_T


def _pre_ref(m, x, w, fused):
    """today's pre: one token at a time for the fused sizes (the kernel computes every token alike, like a one-token
    batch), the whole batch at once for the sizes that take the three kernels themselves"""
    if not fused:
        return m.pre(x, w, 1e-5)
    singles = [m.pre(x[i:i + 1], w, 1e-5) for i in range(x.shape[0])]
    return tuple(torch.cat([o[k] for o in singles]) for k in range(3))


@pytest.mark.parametrize("T,res_scale", [(1, 3.0), (2, 3.0), (3, 3.0), (5, 3.0), (32, 3.0), (128, 3.0), (130, 3.0), (7, 40.0)])
def test_mhc_boundary_vs_unfused(T, res_scale, monkeypatch):
    """post(sublayer i) + pre(sublayer i+1) in one kernel (decode sizes; 37 tokens take the three kernels) vs theta
    GEMM + pre kernel + post kernel. theta's outputs are rounded to bf16 after an fp32 sum, so they flip by one ulp
    whenever the summation order changes: today's path alone differs between batch sizes (cuBLAS GEMM for 37 tokens
    vs GEMV for one: 41% of the rounded outputs flip, coefficients move by up to 5e-3). The fused kernel computes
    every token the same way at any batch size; it is compared with today's one-token path, which it tracks to
    ~1e-4."""
    import nenya_vllm.layers.mhc as mhc
    from nenya_vllm.ops.mhc_boundary import mhc_boundary
    from nenya_vllm.ops.mhc_kernels import mhc_post

    monkeypatch.setattr(mhc, "_IMPL", "triton")
    fused = coef_has_logits(T)
    a, b = _mhc(10), _mhc(11, res_scale)
    torch.manual_seed(12)
    x = (torch.randn(T, 4, 960, device=DEV) * 2).to(torch.bfloat16)
    layer = torch.randn(T, 960, device=DEV).to(torch.bfloat16)
    wa, wb = torch.rand(960, device=DEV) + 0.5, torch.rand(960, device=DEV) + 0.5
    _, _, coef_a = _pre_ref(a, x, wa, fused)
    x_new_ref = mhc_post(x, layer, coef_a)
    r32, _, rcoef = _pre_ref(b, x_new_ref, wb, fused)
    # the fused kernel takes the previous sublayer's coefficients in its own format (Sinkhorn logits)
    _, _, _, coef_a_k = mhc_boundary(x, None, None, a, wa, 1e-5)
    assert (_as_matrix(coef_a_k) - coef_a).abs().max() < 1e-3
    x_new, k32, k16, kcoef = mhc_boundary(x, layer, coef_a_k, b, wb, 1e-5)
    # the post itself, given the same coefficients (theirs differ from today's by the theta flips above)
    _bf16_close(x_new, mhc_post(x, layer, _as_matrix(coef_a_k)), frac=1e-3 if fused else 0, atol=1e-5)
    assert (_as_matrix(kcoef) - rcoef).abs().max() < 1e-3
    assert (k32 - r32).abs().max() <= 2e-3 * r32.abs().max()
    assert torch.equal(k16, k32.to(torch.bfloat16))


@pytest.mark.parametrize("T", [1, 21, 130])
@pytest.mark.parametrize("res_scale", [3.0, 40.0])
def test_mhc_boundary_exact(res_scale, T, monkeypatch):
    """With theta = 0 nothing can flip: the gates, the Sinkhorn (multiplicative below a spread of 60, log domain
    above) and the read + RMSNorm must match an fp64 reference to fp32 accuracy."""
    import nenya_vllm.layers.mhc as mhc
    from nenya_vllm.ops.mhc_boundary import mhc_boundary
    from nenya_vllm.ops.mhc_kernels import mhc_post

    monkeypatch.setattr(mhc, "_IMPL", "triton")
    a, b = _mhc(15), _mhc(16, res_scale)
    with torch.no_grad():
        b.theta.weight.zero_()
    torch.manual_seed(17)
    x = (torch.randn(T, 4, 960, device=DEV) * 2).to(torch.bfloat16)
    layer = torch.randn(T, 960, device=DEV).to(torch.bfloat16)
    wa, wb = torch.rand(960, device=DEV) + 0.5, torch.rand(960, device=DEV) + 0.5
    _, _, _, coef_a = mhc_boundary(x, None, None, a, wa, 1e-5)
    x_new, k32, k16, kcoef = mhc_boundary(x, layer, coef_a, b, wb, 1e-5)
    _bf16_close(x_new, mhc_post(x, layer, coef_a), atol=1e-5)
    # fp64 reference from the kernel's own new streams
    xd = x_new.double()
    h_pre = torch.sigmoid(b.beta_pre.double()).expand(T, 4)
    h_post = 2 * torch.sigmoid(b.beta_post.double()).expand(T, 4)
    z = b.beta_res.double().view(4, 4)
    for _ in range(20):
        z = z - torch.logsumexp(z, dim=0, keepdim=True)
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
    H = z.exp().expand(T, 4, 4)
    xin = (h_pre.unsqueeze(-1) * xd).sum(1)
    ref32 = xin * torch.rsqrt(xin.pow(2).mean(-1, keepdim=True) + 1e-5) * wb.double()
    kcoef = _as_matrix(kcoef)
    assert (kcoef[:, :4].double() - h_post).abs().max() < 1e-6
    assert (kcoef[:, 4:].double().view(T, 4, 4) - H).abs().max() < 2e-6
    assert (k32.double() - ref32).abs().max() <= 2e-6 * ref32.abs().max()


@pytest.mark.parametrize("T", [1, 9, 130])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mhc_boundary_pre_only(dtype, T, monkeypatch):
    """no post (the model's first sublayer: fp32 streams; after an Engram: bf16) == theta GEMM + pre kernel."""
    import nenya_vllm.layers.mhc as mhc
    from nenya_vllm.ops.mhc_boundary import mhc_boundary

    monkeypatch.setattr(mhc, "_IMPL", "triton")
    b = _mhc(13)
    torch.manual_seed(14)
    x = (torch.randn(T, 4, 960, device=DEV) * 2).to(dtype)
    w = torch.rand(960, device=DEV) + 0.5
    r32, _, rcoef = _pre_ref(b, x, w, coef_has_logits(T))
    x_same, k32, k16, kcoef = mhc_boundary(x, None, None, b, w, 1e-5)
    assert x_same.data_ptr() == x.data_ptr()
    assert (_as_matrix(kcoef) - rcoef).abs().max() < 1e-3
    assert (k32 - r32).abs().max() <= 2e-3 * r32.abs().max()


@pytest.mark.parametrize("T", [1, 3, 130])
def test_mhc_post_final(T):
    """last post + sum of the streams + final RMSNorm == post kernel, bf16 sum, RMSNorm."""
    from nenya_vllm.ops.mhc_boundary import mhc_boundary, mhc_post_final
    from nenya_vllm.ops.mhc_kernels import mhc_post

    a = _mhc(40, 40.0)
    torch.manual_seed(41)
    x = (torch.randn(T, 4, 960, device=DEV) * 2).to(torch.bfloat16)
    layer = torch.randn(T, 960, device=DEV).to(torch.bfloat16)
    w = torch.rand(960, device=DEV) + 0.5
    _, _, _, coef = mhc_boundary(x, None, None, a, w, 1e-5)
    ref = mhc_post(x, layer, coef).float().sum(1).to(torch.bfloat16).float()
    ref = (ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + 1e-5) * w).to(torch.bfloat16)
    _bf16_close(mhc_post_final(x, layer, coef, w, 1e-5), ref, frac=2e-2)


@pytest.mark.parametrize("T", [1, 4, 6, 128])
def test_mhc_boundary_deterministic(T):
    """The cross-program reduction (last program of a token block sums the chunk partials in a fixed order) gives
    bitwise-identical results on every launch and leaves the shared counters at zero."""
    import nenya_vllm.ops.mhc_boundary as mb

    a, b = _mhc(20), _mhc(21)
    torch.manual_seed(22)
    x = (torch.randn(T, 4, 960, device=DEV) * 2).to(torch.bfloat16)
    layer = torch.randn(T, 960, device=DEV).to(torch.bfloat16)
    w = torch.rand(960, device=DEV) + 0.5
    _, _, _, coef_a = mb.mhc_boundary(x, None, None, a, w, 1e-5)
    first = mb.mhc_boundary(x, layer, coef_a, b, w, 1e-5)
    for _ in range(5):
        again = mb.mhc_boundary(x, layer, coef_a, b, w, 1e-5)
        for u, v in zip(first, again):
            assert torch.equal(u, v)
    torch.cuda.synchronize()
    from nenya_vllm.ops.decode_gemv import _COUNTERS

    assert all(int(c.abs().sum()) == 0 for c in _COUNTERS.values())


@pytest.mark.parametrize("T", [1, 3, 8, 17, 64, 100])
@pytest.mark.parametrize("N,K,bias", [(7936, 960, False), (960, 1536, False), (1536, 128, True), (130, 960, True)])
def test_decode_linear(N, K, bias, T):
    """fp32 accumulation, one rounding: within one bf16 ulp of the exact product (cuBLAS: the same up to order)."""
    from nenya_vllm.ops.decode_gemv import decode_linear

    from nenya_vllm.ops.decode_gemv import GEMM_MIN_T

    torch.manual_seed(N + K + T)
    w = (torch.randn(N, K, device=DEV) * K ** -0.5).to(torch.bfloat16)
    x = torch.randn(T, K, device=DEV).to(torch.bfloat16)
    b = torch.randn(N, device=DEV).to(torch.bfloat16) if bias else None
    ref = x.double() @ w.double().t() + (b.double() if bias else 0)
    y = decode_linear(x, w, b)
    assert y.dtype == torch.bfloat16 and y.shape == (T, N)
    # the GEMV path sums in order in fp32 (within the bf16 rounding); tensor-core MMAs sum in their own order (like
    # cuBLAS), which can push a rounding a hair past half an ulp
    tol, atol = (2 ** -8, 1e-6) if T < GEMM_MIN_T else (2 ** -7, 1e-4)
    assert ((y.double() - ref).abs() <= tol * ref.abs() + atol).all()
    # strided rows (a column slice of a wider tensor)
    xs = torch.randn(T, K + 64, device=DEV).to(torch.bfloat16)[:, 32:32 + K]
    ref = xs.double() @ w.double().t() + (b.double() if bias else 0)
    assert ((decode_linear(xs, w, b).double() - ref).abs() <= tol * ref.abs() + atol).all()


@pytest.mark.parametrize("T", [1, 3, 5, 16, 40])
def test_decode_linear_prologues(T):
    """The fused input transforms against fp64 references of the same maths, without intermediate rounding: the GEMV
    path computes the input in fp32, the tensor-core path feeds it as two bf16 MMAs (hi + residual)."""
    from nenya_vllm.ops import decode_gemv as dg

    torch.manual_seed(30 + T)
    w = (torch.randn(960, 1536, device=DEV) * 1536 ** -0.5).to(torch.bfloat16)
    mma = T >= dg.GEMM_MIN_T

    def check(y, xin):
        ref = xin.double() @ w.double().t()
        # one bf16 rounding of the output (tensor-core sums: a hair more)
        tol, atol = (2 ** -7, 1e-4) if mma else (2 ** -8, 1e-5)
        assert ((y.double() - ref).abs() <= tol * ref.abs() + atol).all()

    a = torch.randn(T, 2 * 1536, device=DEV).to(torch.bfloat16)
    g, u = a[:, :1536].double(), a[:, 1536:].double()
    check(dg.decode_linear(a, w, prologue=dg.SILU_MUL), g * torch.sigmoid(g) * u)
    a, gate = torch.randn(T, 1536, device=DEV).to(torch.bfloat16), torch.randn(T, 1536, device=DEV).to(torch.bfloat16)
    check(dg.decode_linear(a, w, prologue=dg.SIGMOID_GATE, aux=gate), a.double() * torch.sigmoid(gate.double()))
    nw = torch.rand(128, device=DEV) + 0.5
    a3 = a.double().view(T, 12, 128)
    xin = (a3 * torch.rsqrt(a3.pow(2).mean(-1, keepdim=True) + 1e-5) * nw.double()).view(T, 1536)
    check(dg.decode_linear(a, w, prologue=dg.GATED_RMSNORM, aux=gate, norm_weight=nw, eps=1e-5, group=128),
          xin * torch.sigmoid(gate.double()))
    rows = torch.randint(0, 4000, (T, 1536 // 96), device=DEV)
    table = torch.randn(4000, 96, device=DEV).to(torch.bfloat16)
    check(dg.decode_linear(rows, w, prologue=dg.GATHER, aux=table, group=96), table[rows].reshape(T, 1536).double())


def _gdn2_inputs(T, exact, seed):
    """GDN-2 decode inputs. exact: the inputs of every GEMV (x, f_a, g_a) and their weights on coarse binary grids
    so every dot product is exact in fp32 -- cuBLAS and the kernels then round identically and only fp32 summation
    order in the recurrence remains."""
    torch.manual_seed(seed)
    H, K, D = 12, 128, 960
    KD = H * K
    NOUT = 5 * KD + 2 * K
    slots = 2 * T + 3
    x = torch.randn(T, D, device=DEV)
    w_in = torch.randn(NOUT, D, device=DEV) * D ** -0.5
    fbw = torch.randn(KD, K, device=DEV) * K ** -0.5
    gbw = torch.randn(KD, K, device=DEV) * K ** -0.5
    if exact:
        x = torch.randint(-4, 5, (T, D), device=DEV).float() / 4
        w_in = torch.randint(-8, 9, (NOUT, D), device=DEV).float() / 256
        w_in[5 * KD:] = torch.randint(-8, 9, (2 * K, D), device=DEV).float() / 64     # f_a, g_a land on a grid
        fbw = torch.randint(-8, 9, (KD, K), device=DEV).float() / 512
        gbw = torch.randint(-8, 9, (KD, K), device=DEV).float() / 512
    d = dict(x=x.to(torch.bfloat16), w_in=w_in.to(torch.bfloat16),
             conv_state=torch.randn(slots, 3 * KD, 3, device=DEV).to(torch.bfloat16),
             conv_weight=torch.randn(3 * KD, 4, device=DEV) * 0.5,
             fbw=fbw.to(torch.bfloat16), gbw=gbw.to(torch.bfloat16),
             gbb=(torch.randn(KD, device=DEV) * 0.5).to(torch.bfloat16),
             A_log=torch.rand(H, device=DEV) * 2 - 1, dt_bias=torch.randn(KD, device=DEV) * 0.5,
             ssm=torch.randn(slots, H, K, K, device=DEV) * 0.1)
    # distinct slots, slot 0 = vLLM's null block for a padded last row
    idx = (torch.randperm(slots - 1, device=DEV)[:T] + 1).to(torch.int32)
    if T > 1:
        idx[-1] = 0
    d["idx"] = idx
    return d


def _gdn2_reference(d, n):
    """today's unfused decode path: cuBLAS in_proj / f_b / g_b, the gate kernel, vLLM's conv update, fla"""
    import torch.nn.functional as F
    from fla.ops.gdn2.fused_recurrent import fused_recurrent_gdn2_fwd
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    from nenya_vllm.ops.small_kernels import gdn2_gates

    H, K = 12, 128
    KD = H * K
    idx = d["idx"][:n]
    conv, ssm = d["conv_state"].clone(), d["ssm"].clone()
    proj = F.linear(d["x"], d["w_in"])
    f = F.linear(proj[:, 5 * KD:5 * KD + K], d["fbw"])
    gate = F.linear(proj[:, 5 * KD + K:], d["gbw"], d["gbb"])
    b, w = gdn2_gates(proj, 3 * KD, 4 * KD, KD)
    x = causal_conv1d_update(proj[:n, :3 * KD], conv, d["conv_weight"], None, "silu", conv_state_indices=idx,
                             validate_data=False)
    qkv = x.reshape(n, 3, H, K).permute(1, 0, 2, 3).contiguous()
    core = torch.zeros(proj.shape[0], H, K, dtype=torch.bfloat16, device=DEV)
    fused_recurrent_gdn2_fwd(qkv[0].unsqueeze(1), qkv[1].unsqueeze(1), qkv[2].unsqueeze(1), f[:n].view(n, 1, H, K),
                             b[:n].view(n, 1, H, K), w[:n].view(n, 1, H, K), initial_state=ssm, scale=K ** -0.5,
                             inplace_final_state=True, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True,
                             out=core[:n].view(n, 1, H, K), use_gate_in_kernel=True, A_log=d["A_log"],
                             dt_bias=d["dt_bias"])
    return core.view(-1, KD), gate, conv, ssm


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("T", [1, 4, 16, 33])
def test_gdn2_decode_step(T, exact):
    """in_proj + short convs (one kernel) and the recurrence with f_b, g_b and the gates (one kernel) == today's
    eight kernels. 33 tokens take cuBLAS in_proj + vLLM's conv kernel in front of the recurrent kernel, as the layer
    does above GEMV_MAX_T."""
    import torch.nn.functional as F
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

    from nenya_vllm.ops.decode_gemv import GEMV_MAX_T
    from nenya_vllm.ops.gdn2_decode import gdn2_in_proj, gdn2_recurrent_step

    d = _gdn2_inputs(T, exact, 50 + T)
    real = d["idx"] != 0
    core_r, gate_r, conv_r, ssm_r = _gdn2_reference(d, T)
    conv, ssm = d["conv_state"].clone(), d["ssm"].clone()
    if T <= GEMV_MAX_T:
        proj = gdn2_in_proj(d["x"], d["w_in"], conv, d["conv_weight"], d["idx"], T)
    else:
        proj = F.linear(d["x"], d["w_in"])
        causal_conv1d_update(proj[:, :3 * 1536], conv, d["conv_weight"], None, "silu", conv_state_indices=d["idx"],
                             validate_data=False)
    core, gate = gdn2_recurrent_step(proj, d["fbw"], d["gbw"], d["gbb"], d["A_log"], d["dt_bias"], ssm, d["idx"], T,
                                     12, 128)
    if exact:
        assert torch.equal(gate, gate_r)
    else:
        # g_b on this proj: on an RTX PRO 6000 (T = 3, 4) cuBLAS's in_proj rounds one g_a element of the last row
        # differently from the exact result (the fused in_proj matches it), which moves that row's whole reference
        # gate; the fused gate equals the fp64 g_b(g_a) + bias of its own proj (3 of 3
        # re-runs failed against the cuBLAS reference, the fused proj and gate equal to fp64)
        _bf16_close(gate, F.linear(proj[:, 5 * 1536 + 128:], d["gbw"], d["gbb"]), frac=2e-2, atol=1e-5)
        # the g_a columns themselves against cuBLAS's in_proj: one-ulp differences allowed (that box: 1 of 512)
        _bf16_close(proj[:, 5 * 1536 + 128:], F.linear(d["x"], d["w_in"])[:, 5 * 1536 + 128:], frac=2e-2, atol=1e-5)
    # null-block rows: zeros (the reference leaves garbage there; both are discarded downstream)
    assert torch.equal(core[~real], torch.zeros_like(core[~real]))
    _bf16_close(core[real], core_r[real], frac=2e-3 if exact else 5e-2, atol=1e-4 if exact else 2e-3)
    sl = d["idx"][real].long()
    if exact:
        assert torch.equal(conv[sl], conv_r[sl])
    else:
        _bf16_close(conv[sl], conv_r[sl], frac=2e-2, atol=1e-5)
    untouched = torch.ones(conv.shape[0], dtype=torch.bool, device=DEV)
    untouched[sl] = False
    untouched[0] = False
    assert torch.equal(conv[untouched], d["conv_state"][untouched])
    assert torch.equal(ssm[untouched], d["ssm"][untouched])
    tol = 1e-5 if exact else 5e-2
    assert ((ssm[sl] - ssm_r[sl]).abs() <= tol * ssm_r[sl].abs().amax()).all()


@pytest.mark.parametrize("cfg", ["32,64,3,1", "32,64,3,3", "64,64,3,5", "32,128,2,4"])
@pytest.mark.parametrize("T", [5, 16, 32, 64])
def test_gdn2_in_proj_splitk(T, cfg, monkeypatch):
    """in_proj's tensor-core path with split-K: the same output, convolutions and history as the reference (exact
    inputs: every partial sum exact, so bitwise)"""
    import nenya_vllm.ops.gdn2_decode as gd

    monkeypatch.setattr(gd, "IN_PROJ_CONFIG", cfg)
    d = _gdn2_inputs(T, True, 80 + T)
    real = d["idx"] != 0
    _, _, conv_r, _ = _gdn2_reference(d, T)
    conv = d["conv_state"].clone()
    proj = gd.gdn2_in_proj(d["x"], d["w_in"], conv, d["conv_weight"], d["idx"], T)
    monkeypatch.setattr(gd, "IN_PROJ_CONFIG", "32,64,3,1")
    conv1 = d["conv_state"].clone()
    proj1 = gd.gdn2_in_proj(d["x"], d["w_in"], conv1, d["conv_weight"], d["idx"], T)
    assert torch.equal(proj[real], proj1[real]) and torch.equal(conv, conv1)
    sl = d["idx"][real].long()
    assert torch.equal(conv[sl], conv_r[sl])


@pytest.mark.parametrize("T", [1, 5])
def test_mla_decode_prep(T):
    """kv_norm + latent cache row + absorbed query (one kernel) == rms_norm_fp32w, vLLM's concat_and_cache_mla and
    the absorbed-query bmm + concat of vLLM's MLA decode path."""
    import vllm._custom_ops as ops

    from nenya_vllm.ops.mla_decode import mla_decode_prep
    from nenya_vllm.ops.small_kernels import rms_norm_fp32w

    torch.manual_seed(60 + T)
    H, P, R, L, page, blocks = 12, 80, 32, 320, 16, 8
    kqg = torch.randn(T, L + R + H * (P + R) + 960, device=DEV).to(torch.bfloat16)
    kvn = torch.rand(L, device=DEV) + 0.5
    W_UK = (torch.randn(L, H, P, device=DEV) * 0.05).to(torch.bfloat16)
    W_UK_T = W_UK.permute(1, 2, 0)                                 # vLLM's [N, P, L] view
    slots = torch.randperm(blocks * page, device=DEV)[:T].long()
    if T > 1:
        slots[-1] = -1                                             # padding row: not written
    cache = torch.randn(blocks, page, L + R, device=DEV).to(torch.bfloat16)
    ref_cache = cache.clone()
    kv_c = rms_norm_fp32w(kqg[:, :L], kvn, 1e-6, torch.bfloat16)
    ops.concat_and_cache_mla(kv_c, kqg[:, L:L + R], ref_cache, slots, "auto", torch.tensor(1.0, device=DEV))
    q = kqg[:, L + R:L + R + H * (P + R)].reshape(T, H, P + R)
    ql = torch.bmm(q[..., :P].transpose(0, 1), W_UK_T).transpose(0, 1)
    ref_q = torch.cat([ql, q[..., P:]], dim=-1)
    got_q = mla_decode_prep(kqg, kvn, 1e-6, cache, slots, W_UK_T, L, R, H, P)
    assert torch.equal(cache, ref_cache)
    _bf16_close(got_q, ref_q, frac=1e-2, atol=1e-5)


@pytest.mark.parametrize("splits", [1, 4])
def test_mla_stage2_vup(splits):
    """stage 2 (merge of the splits) + value up-projection == vLLM's stage-2 kernel + the W_UV bmm."""
    from vllm.v1.attention.ops.triton_decode_attention import _decode_softmax_reducev_fwd

    from nenya_vllm.ops.mla_decode import mla_stage2_vup

    torch.manual_seed(70 + splits)
    B, H, L, V = 6, 12, 320, 80
    logits = torch.randn(B, H, splits, L + 1, device=DEV)
    seq_lens = torch.randint(1, 3000, (B,), device=DEV, dtype=torch.int32)
    W_UV = (torch.randn(L, H, V, device=DEV) * 0.05).to(torch.bfloat16).transpose(0, 1)      # vLLM's [N, L, V] view
    o = torch.empty(B, H, L, dtype=torch.bfloat16, device=DEV)
    lse = torch.empty(B, H, dtype=torch.bfloat16, device=DEV)
    q_dummy = torch.empty(B, H, L + 32, dtype=torch.bfloat16, device=DEV)
    v_dummy = torch.empty(1, 1, 1, L, dtype=torch.bfloat16, device=DEV)
    _decode_softmax_reducev_fwd(logits, q_dummy, o, lse, v_dummy, seq_lens, splits)
    ref = torch.bmm(o.transpose(0, 1), W_UV).transpose(0, 1).reshape(B, H * V)
    out = torch.empty(B, H * V, dtype=torch.bfloat16, device=DEV)
    mla_stage2_vup(logits, seq_lens, W_UV, out, L, H, V)
    _bf16_close(out, ref, frac=1e-2, atol=1e-5)


@pytest.mark.parametrize("post", [True, False])
@pytest.mark.parametrize("T", [1, 5])
def test_engram_decode(T, post):
    """table-row gather + [W_K | W_V] GEMV, then post + gate + norms + dilated conv + residual add in one kernel ==
    F.embedding, cuBLAS, mhc_post and the release's ops with vLLM's ple_conv (layers/engram.py)."""
    import math

    import torch.nn.functional as F

    from nenya_vllm.layers.mhc import rms_norm_fp32
    from nenya_vllm.ops.decode_gemv import GATHER, decode_linear
    from nenya_vllm.ops.dilated_conv import ple_conv
    from nenya_vllm.ops.engram_decode import engram_decode_combine
    from nenya_vllm.ops.mhc_kernels import mhc_post

    torch.manual_seed(80 + T)
    n, d, G, KS, dil = 4, 960, 60, 4, 3
    eps = torch.finfo(torch.float32).eps
    slots, nrows = 2 * T + 3, 5000
    tables = (torch.randn(nrows, G, device=DEV) * 0.5).to(torch.bfloat16)
    rows = torch.randint(0, nrows, (T, d // G), device=DEV)
    w_kv = (torch.randn((n + 1) * d, d, device=DEV) * d ** -0.5).to(torch.bfloat16)
    b_kv = (torch.randn((n + 1) * d, device=DEV) * 0.1).to(torch.bfloat16)
    x_old = (torch.randn(T, n, d, device=DEV) * 2).to(torch.bfloat16)
    layer = torch.randn(T, d, device=DEV).to(torch.bfloat16)
    coef = torch.cat([torch.rand(T, n, device=DEV) * 2, torch.softmax(torch.randn(T, n * n, device=DEV), -1)], -1)
    nh, nk, nv = (torch.rand(d, device=DEV) + 0.5 for _ in range(3))
    cw = torch.randn(n * d, KS, device=DEV) * 0.5
    state = torch.randn(slots, n * d, (KS - 1) * dil, device=DEV)
    idx = (torch.randperm(slots - 1, device=DEV)[:T] + 1).to(torch.int32)
    if T > 1:
        idx[-1] = 0
    has_init = torch.ones(T, dtype=torch.bool, device=DEV)

    # reference: today's ops
    kv_r = F.linear(F.embedding(rows, tables).reshape(T, -1), w_kv, b_kv)
    xr = mhc_post(x_old, layer, coef) if post else x_old
    key, value = kv_r[:, :n * d].view(T, n, d), kv_r[:, n * d:]
    z = (rms_norm_fp32(xr, nh, eps) * rms_norm_fp32(key, nk, eps)).sum(-1, keepdim=True) / math.sqrt(d)
    z = z.abs().clamp_min(1e-6).sqrt() * z.sign()
    vg = torch.sigmoid(z) * value.float().unsqueeze(1)
    c = rms_norm_fp32(vg, nv, eps).reshape(T, n * d).contiguous()
    y = torch.zeros(T, n * d, device=DEV)
    st_r = state.clone()
    ple_conv(inputs=c, residual=y, conv_state=st_r, conv_weights=cw, state_indices=idx, outer_residual=None,
             mode="decode", dilation=dil, has_initial_states=has_init)
    ref = (xr + (y.view(T, n, d) + vg).to(torch.bfloat16)).to(torch.bfloat16)

    kv = decode_linear(rows, w_kv, b_kv, prologue=GATHER, aux=tables, group=G)
    _bf16_close(kv, kv_r, frac=1e-2, atol=1e-5)
    st = state.clone()
    got = engram_decode_combine(x_old, layer if post else None, coef if post else None, kv_r, nh, nk, nv, cw, st, idx,
                                has_init, T, eps, dil)
    _bf16_close(got, ref, frac=1e-2, atol=1e-4)
    real = idx != 0
    sl = idx[real].long()
    assert ((st[sl] - st_r[sl]).abs() <= 1e-5 * st_r[sl].abs().amax()).all()
    untouched = torch.ones(slots, dtype=torch.bool, device=DEV)
    untouched[sl] = False
    untouched[0] = False
    assert torch.equal(st[untouched], state[untouched])


@pytest.mark.parametrize("T", [5, 16, 32, 64])
@pytest.mark.parametrize("splitk", [1, 2, 3, 5, 12])
def test_decode_linear_splitk(T, splitk):
    """split-K tensor-core path: every slice count within the bf16 output rounding of the fp64 product (prologues
    included, slices of whole norm groups), bitwise reproducible, counters left at zero, bias and -inf mask applied
    once"""
    from nenya_vllm.ops import decode_gemv as dg

    torch.manual_seed(40 + T + splitk)
    N, K = 960, 1536
    w = (torch.randn(N, K, device=DEV) * K ** -0.5).to(torch.bfloat16)
    b = torch.randn(N, device=DEV).to(torch.bfloat16)
    a = torch.randn(T, K, device=DEV).to(torch.bfloat16)
    gate = torch.randn(T, K, device=DEV).to(torch.bfloat16)
    nw = torch.rand(128, device=DEV) + 0.5
    a3 = a.double().view(T, 12, 128)
    xin = (a3 * torch.rsqrt(a3.pow(2).mean(-1, keepdim=True) + 1e-5) * nw.double()).view(T, K)
    xin = xin * torch.sigmoid(gate.double())
    for BN in (16, 64):
        cfg = (BN, 128, 4, 2, splitk)
        kw = dict(prologue=dg.GATED_RMSNORM, aux=gate, norm_weight=nw, eps=1e-5, group=128, config=cfg)
        y = dg.decode_linear(a, w, b, n_valid=900, **kw)
        ref = xin @ w.double().t() + b.double()
        assert ((y[:, :900].double() - ref[:, :900]).abs() <= 2 ** -7 * ref[:, :900].abs() + 1e-4).all()
        assert torch.isneginf(y[:, 900:]).all()
        assert torch.equal(dg.decode_linear(a, w, b, n_valid=900, **kw), y)
        # plain product, K not a multiple of the slice width
        x = torch.randn(T, 960, device=DEV).to(torch.bfloat16)
        w2 = (torch.randn(2656, 960, device=DEV) * 960 ** -0.5).to(torch.bfloat16)
        y2 = dg.decode_linear(x, w2, config=(BN, 128, 4, 2, min(splitk, 8)))
        ref2 = x.double() @ w2.double().t()
        assert ((y2.double() - ref2).abs() <= 2 ** -7 * ref2.abs() + 1e-4).all()
    assert int(dg.counters(1, DEV, "decode_gemm").abs().sum()) == 0


def test_decode_linear_masked_columns():
    """the LM-head use: columns >= n_valid come out -inf, the rest as without the mask"""
    from nenya_vllm.ops.decode_gemv import decode_linear

    torch.manual_seed(90)
    w = (torch.randn(1000, 960, device=DEV) * 960 ** -0.5).to(torch.bfloat16)
    x = torch.randn(3, 960, device=DEV).to(torch.bfloat16)
    full = decode_linear(x, w)
    masked = decode_linear(x, w, n_valid=926)
    assert torch.equal(masked[:, :926], full[:, :926])
    assert torch.isneginf(masked[:, 926:]).all()


@pytest.mark.parametrize("T", [1, 7, 300])
def test_embed_norm_streams(T):
    """embedding + fp32 RMSNorm + 4 identical streams in one kernel == F.embedding, rms_norm_fp32w, expand"""
    import torch.nn.functional as F

    from nenya_vllm.ops.small_kernels import embed_norm_streams, rms_norm_fp32w

    torch.manual_seed(91 + T)
    emb = torch.randn(5000, 960, device=DEV).to(torch.bfloat16)
    w = torch.rand(960, device=DEV) + 0.5
    ids = torch.randint(0, 5000, (T,), device=DEV)
    ref = rms_norm_fp32w(F.embedding(ids, emb), w, 1e-6, torch.float32).unsqueeze(1).expand(T, 4, 960)
    assert torch.equal(embed_norm_streams(ids, emb, w, 1e-6, 4), ref)


@pytest.mark.parametrize("T", [1, 5, 128])
def test_mhc_boundary_split_sinkhorn(T, monkeypatch):
    """the Sinkhorn on its own program (default) == the Sinkhorn in the tail, bitwise (same partials, same order)"""
    import nenya_vllm.ops.mhc_boundary as mb

    a, b = _mhc(23), _mhc(24, 40.0)
    torch.manual_seed(25)
    x = (torch.randn(T, 4, 960, device=DEV) * 2).to(torch.bfloat16)
    layer = torch.randn(T, 960, device=DEV).to(torch.bfloat16)
    w = torch.rand(960, device=DEV) + 0.5
    _, _, _, coef_a = mb.mhc_boundary(x, None, None, a, w, 1e-5)
    monkeypatch.setattr(mb, "SPLIT", True)
    split = mb.mhc_boundary(x, layer, coef_a, b, w, 1e-5)
    monkeypatch.setattr(mb, "SPLIT", False)
    tail = mb.mhc_boundary(x, layer, coef_a, b, w, 1e-5)
    for u, v in zip(split, tail):
        assert torch.equal(u, v)
    torch.cuda.synchronize()
    from nenya_vllm.ops.decode_gemv import _COUNTERS

    assert all(int(c.abs().sum()) == 0 for c in _COUNTERS.values())


def _gdn2_seq_ref64(proj, hist0, cw, ssm0, qsl, idx, has_init, A_log, dtb, f, H=12, K=128):
    """fp64 reference of the GDN-2 step over sequences (convolution with history, gates, fla's recurrence), with
    the release's bf16 roundings of the convolution output and the gates."""
    import torch.nn.functional as F

    KD = H * K
    T = proj.shape[0]
    out = torch.zeros(T, KD, dtype=torch.float64, device=DEV)
    states, hists = {}, {}
    for i in range(qsl.shape[0] - 1):
        a, e = int(qsl[i]), int(qsl[i + 1])
        slot, ini = int(idx[i]), bool(has_init[i])
        h = hist0[slot].double() if ini else torch.zeros(3 * KD, 3, dtype=torch.float64, device=DEV)
        S = ssm0[slot].double() if ini else torch.zeros(H, K, K, dtype=torch.float64, device=DEV)
        full = torch.cat([h.t(), proj[a:e, :3 * KD].double()], 0)
        for j in range(e - a):
            acc = (full[j:j + 4] * cw.double().t()).sum(0)
            y = (acc * torch.sigmoid(acc)).to(torch.bfloat16).double()
            q, k, v = y[:KD].view(H, K), y[KD:2 * KD].view(H, K), y[2 * KD:].view(H, K)
            t = a + j
            b = torch.sigmoid(proj[t, 3 * KD:4 * KD].float()).to(torch.bfloat16).double().view(H, K)
            w = torch.sigmoid(proj[t, 4 * KD:5 * KD].float()).to(torch.bfloat16).double().view(H, K)
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * K ** -0.5
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            gk = -torch.exp(A_log.double())[:, None] * F.softplus(f[t].double().view(H, K) + dtb.double().view(H, K))
            S = S * torch.exp(gk)[:, :, None]
            S = S + k[:, :, None] * (w * v - (S * (b * k)[:, :, None]).sum(1))[:, None, :]
            out[t] = (S * q[:, :, None]).sum(1).reshape(-1)
        states[slot], hists[slot] = S, full[-3:].t()
    return out, states, hists


@pytest.mark.parametrize("lens", [[5], [1, 7, 3], [2, 1, 40, 1]])
def test_gdn2_prefill_short(lens):
    """varlen short convolution + sequential recurrence (the short-prefill path) against an fp64 reference, on a
    batch of sequences with and without initial states. (fla's chunked kernels, the long-prefill path, keep WY
    intermediates in bf16: their states measured ~1e3x further from this reference, 6e-3 vs 1e-6.)"""
    from nenya_vllm.ops.gdn2_decode import gdn2_prefill_short

    torch.manual_seed(100 + sum(lens))
    H, K = 12, 128
    KD = H * K
    N, T = len(lens), sum(lens)
    slots = N + 3
    proj = torch.randn(T, 5 * KD + 2 * K, device=DEV).to(torch.bfloat16)
    f = (torch.randn(T, KD, device=DEV) * 0.5).to(torch.bfloat16)
    conv = torch.randn(slots, 3 * KD, 3, device=DEV).to(torch.bfloat16)
    cw = torch.randn(3 * KD, 4, device=DEV) * 0.5
    ssm = torch.randn(slots, H, K, K, device=DEV) * 0.1
    A_log, dtb = torch.rand(H, device=DEV) * 2 - 1, torch.randn(KD, device=DEV) * 0.5
    qsl = torch.zeros(N + 1, dtype=torch.int32, device=DEV)
    qsl[1:] = torch.cumsum(torch.tensor(lens, device=DEV), 0)
    idx = (torch.randperm(slots - 1, device=DEV)[:N] + 1).to(torch.int32)
    has_init = torch.tensor([i % 2 == 0 for i in range(N)], device=DEV)
    out64, st64, hist64 = _gdn2_seq_ref64(proj, conv.clone(), cw, ssm.clone(), qsl, idx, has_init, A_log, dtb, f)
    core = torch.empty(T, KD, dtype=torch.bfloat16, device=DEV)
    gdn2_prefill_short(proj, f, conv, cw, ssm, qsl, idx, has_init, A_log, dtb, N, H, K, core)
    assert ((core.double() - out64).abs() <= 2 ** -8 * out64.abs() + 1e-4).all()
    for slot, S in st64.items():
        assert (ssm[slot].double() - S).abs().max() <= 1e-4 * S.abs().max()
        assert torch.equal(conv[slot].double(), hist64[slot])


@pytest.mark.parametrize("lens", [[5], [1, 7, 3], [2, 1, 40, 1], [300, 2, 129]])
def test_gdn2_chunk_prep(lens):
    """the long-prefill path's prep kernel == vLLM's causal_conv1d_fn (SiLU) + the q/k/v permute + fla's l2norm_fwd
    on q, k (what chunk_gdn2 runs with use_qk_l2norm_in_kernel=True) + the gate kernel; same histories afterwards.
    The conv state in vLLM's layout ([slots, 3, 3 KD] storage viewed dim-first)."""
    from fla.modules.l2norm import l2norm_fwd
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    from nenya_vllm.ops.gdn2_decode import gdn2_chunk_prep
    from nenya_vllm.ops.small_kernels import gdn2_gates

    torch.manual_seed(120 + sum(lens))
    H, K = 12, 128
    KD = H * K
    N, T = len(lens), sum(lens)
    slots = N + 3
    proj = torch.randn(T, 5 * KD + 2 * K, device=DEV).to(torch.bfloat16)
    conv = torch.randn(slots, 3, 3 * KD, device=DEV).to(torch.bfloat16).transpose(-1, -2)
    cw = torch.randn(3 * KD, 4, device=DEV) * 0.5
    qsl = torch.zeros(N + 1, dtype=torch.int32, device=DEV)
    qsl[1:] = torch.cumsum(torch.tensor(lens, device=DEV), 0)
    idx = (torch.randperm(slots - 1, device=DEV)[:N] + 1).to(torch.int32)
    has_init = torch.tensor([i % 2 == 0 for i in range(N)], device=DEV)

    conv_ref = conv.clone()
    x = causal_conv1d_fn(proj[:, :3 * KD].transpose(0, 1), cw, None, conv_states=conv_ref, query_start_loc=qsl,
                         cache_indices=idx, has_initial_state=has_init, activation="silu")
    qkv = x.reshape(3, H, K, T).permute(0, 3, 1, 2).contiguous()
    q_ref, _ = l2norm_fwd(qkv[0].unsqueeze(0))
    k_ref, _ = l2norm_fwd(qkv[1].unsqueeze(0))
    b_ref, w_ref = gdn2_gates(proj, 3 * KD, 4 * KD, KD)

    q, k, v, b, w = gdn2_chunk_prep(proj, conv, cw, qsl, idx, has_init, N, H, K)
    assert torch.equal(v.view(T, H, K), qkv[2])
    assert torch.equal(b, b_ref) and torch.equal(w, w_ref)
    # q, k: the same bf16 values but for rare 1-ulp roundings (the norm's fp32 sum in a different order)
    for got, ref in ((q, q_ref), (k, k_ref)):
        ref = ref.view(T, KD)
        diff = (got.float() - ref.float()).abs()
        assert (diff <= ref.float().abs() * 2 ** -7).all() and (diff > 0).float().mean() < 1e-3
    for s in range(N):
        assert torch.equal(conv[idx[s]], conv_ref[idx[s]])
    assert torch.equal(conv, conv_ref)


@pytest.mark.parametrize("post", [True, False])
def test_engram_prefill_kernels(post):
    """the prefill path's per-token kernel (post + gate + norms -> conv input) and combine kernel == the release's ops
    around the same dilated convolution (vLLM's ple_conv, prefill mode)"""
    import math

    from nenya_vllm.layers.mhc import rms_norm_fp32
    from nenya_vllm.ops.dilated_conv import ple_conv
    from nenya_vllm.ops.engram_decode import engram_combine, engram_gate
    from nenya_vllm.ops.mhc_kernels import mhc_post

    torch.manual_seed(110)
    n, d, KS, dil = 4, 960, 4, 3
    eps = torch.finfo(torch.float32).eps
    lens = [5, 1, 12]
    T, N = sum(lens), len(lens)
    slots = N + 3
    x_old = (torch.randn(T, n, d, device=DEV) * 2).to(torch.bfloat16)
    layer = torch.randn(T, d, device=DEV).to(torch.bfloat16)
    coef = torch.cat([torch.rand(T, n, device=DEV) * 2, torch.softmax(torch.randn(T, n * n, device=DEV), -1)], -1)
    kv = (torch.randn(T, (n + 1) * d, device=DEV) * 0.5).to(torch.bfloat16)
    nh, nk, nv = (torch.rand(d, device=DEV) + 0.5 for _ in range(3))
    cw = torch.randn(n * d, KS, device=DEV) * 0.5
    state = torch.randn(slots, n * d, (KS - 1) * dil, device=DEV)
    idx = (torch.randperm(slots - 1, device=DEV)[:N] + 1).to(torch.int32)
    qsl = torch.zeros(N + 1, dtype=torch.int32, device=DEV)
    qsl[1:] = torch.cumsum(torch.tensor(lens, device=DEV), 0)
    has_init = torch.tensor([True, False, True], device=DEV)

    def conv(c, st):
        y = torch.zeros(T, n * d, device=DEV)
        ple_conv(inputs=c, residual=y, conv_state=st, conv_weights=cw, state_indices=idx, outer_residual=None,
                 mode="prefill", dilation=dil, query_start_loc=qsl, has_initial_states=has_init)
        return y

    xr = mhc_post(x_old, layer, coef) if post else x_old
    key, value = kv[:, :n * d].view(T, n, d), kv[:, n * d:]
    z = (rms_norm_fp32(xr, nh, eps) * rms_norm_fp32(key, nk, eps)).sum(-1, keepdim=True) / math.sqrt(d)
    z = z.abs().clamp_min(1e-6).sqrt() * z.sign()
    vg = torch.sigmoid(z) * value.float().unsqueeze(1)
    c_ref = rms_norm_fp32(vg, nv, eps).reshape(T, n * d).contiguous()
    st_r = state.clone()
    y_ref = conv(c_ref, st_r)
    ref = (xr + (y_ref.view(T, n, d) + vg).to(torch.bfloat16)).to(torch.bfloat16)

    c, gate, xs = engram_gate(x_old, layer if post else None, coef if post else None, kv, nh, nk, nv, eps)
    assert ((c - c_ref).abs() <= 1e-5 * c_ref.abs().amax()).all()
    _bf16_close(xs, xr, frac=1e-3, atol=1e-5)
    st = state.clone()
    y = conv(c, st)
    out = torch.empty(T, n, d, dtype=torch.bfloat16, device=DEV)
    engram_combine(xs, y, gate, kv, out)
    _bf16_close(out, ref, frac=1e-2, atol=1e-4)
    assert ((st - st_r).abs() <= 1e-5 * st_r.abs().amax()).all()
