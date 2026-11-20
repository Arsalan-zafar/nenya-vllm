# SPDX-License-Identifier: Apache-2.0
# Nenya's Engram for decode-sized batches in three kernels instead of ~33 (layers/engram.py routes decode-only
# batches here): the n-gram hash rows (ops/ngram_rows.py), the [W_K x4 | W_V] GEMV with the 16 table rows gathered
# in its input prologue (ops/decode_gemv.py, GATHER), and this kernel -- one program per (token, stream) -- which also
# applies the previous sublayer's mHC post (so the posted streams never go to memory):
#   x_s    = bf16(post_s * layer + sum_j H_sj x_j)                        (ops/mhc_kernels.py's post)
#   z_s    = <RMSNorm_h(x_s), RMSNorm_k(k_s)> / sqrt(D);  gate_s = sigmoid(sign(z) sqrt(max(|z|, 1e-6)))
#   v_s    = gate_s * value;  c_s = RMSNorm_v(v_s)                        (fp32, eps = fp32 machine epsilon)
#   y_s    = SiLU(dilated causal conv of c_s: taps t-9, t-6, t-3, t), history in fp32   (ple_conv, decode mode)
#   out_s  = bf16(x_s + bf16(y_s + v_s))
# with the release's operations in its order (layers/engram.py: torch ops under autocast + vLLM's ple_conv).

import torch

from vllm.triton_utils import tl, triton

from nenya_vllm.ops.decode_gemv import PDL, pdl, repeats


@triton.jit
def _rms(v, w, eps, D: tl.constexpr):
    """torch: v * rsqrt(mean(v^2) + eps) * w (fp32)"""
    return v * tl.math.rsqrt(tl.sum(v * v, axis=0) / D + eps) * w


@triton.jit
def _engram_decode_kernel(
    x_ptr,                      # [T, N, D] bf16 streams before the post (or the posted streams if not DO_POST)
    layer_ptr, coef_ptr,        # [T, D] bf16 sublayer output, [T, N + N*N] fp32 post gates + H   (DO_POST)
    kv_ptr, kv_stride,          # [T, (N + 1) D] bf16: keys of the N streams | value
    nh_ptr, nk_ptr, nv_ptr,     # [D] fp32 norm weights
    cw_ptr,                     # [N D, KS] fp32 conv taps (stream-major channels)
    st_ptr, st_s_slot, st_s_ch, st_s_w,   # conv history [slots, N D, STATE_LEN] fp32 (any strides)
    idx_ptr, has_init_ptr,      # [T] state slots (0 = vLLM's null block), [T] bool
    out_ptr,                    # [T, N, D] bf16
    n_real, eps, sqrt_d,
    N: tl.constexpr, D: tl.constexpr, DP2: tl.constexpr, KS: tl.constexpr, DIL: tl.constexpr,
    STATE_LEN: tl.constexpr, DO_POST: tl.constexpr, HAS_INIT: tl.constexpr, PDL: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)
    t64 = t.to(tl.int64)
    offs = tl.arange(0, DP2)
    m = offs < D
    ch = s * D + offs
    nh = tl.load(nh_ptr + offs, mask=m, other=0.0)
    nk = tl.load(nk_ptr + offs, mask=m, other=0.0)
    nv = tl.load(nv_ptr + offs, mask=m, other=0.0)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    xb = x_ptr + t64 * N * D
    if DO_POST:
        cin = coef_ptr + t64 * (N + N * N)
        xs = tl.load(cin + s) * tl.load(layer_ptr + t64 * D + offs, mask=m, other=0.0).to(tl.float32)
        for j in tl.static_range(N):
            xs += tl.load(cin + N + s * N + j) * tl.load(xb + j * D + offs, mask=m, other=0.0).to(tl.float32)
        xs = xs.to(tl.bfloat16).to(tl.float32)
    else:
        xs = tl.load(xb + s * D + offs, mask=m, other=0.0).to(tl.float32)
    krow = kv_ptr + t64 * kv_stride
    key = tl.load(krow + s * D + offs, mask=m, other=0.0).to(tl.float32)
    value = tl.load(krow + N * D + offs, mask=m, other=0.0).to(tl.float32)
    z = tl.sum(_rms(xs, nh, eps, D) * _rms(key, nk, eps, D), axis=0) / sqrt_d
    sgn = tl.where(z > 0, 1.0, tl.where(z < 0, -1.0, 0.0))
    gate = tl.sigmoid(tl.sqrt(tl.maximum(tl.abs(z), 1e-6)) * sgn)
    vg = gate * value
    c = _rms(vg, nv, eps, D)
    # ple_conv, decode mode: taps h = DIL k for k < KS - 1 from the history, the last tap is the current input
    slot = tl.load(idx_ptr + t, mask=t < n_real, other=0).to(tl.int64)
    ok = slot != 0
    if HAS_INIT:
        init = tl.load(has_init_ptr + t, mask=ok & (t < n_real), other=0) != 0
    else:
        init = ok
    sbase = st_ptr + slot * st_s_slot + ch.to(tl.int64) * st_s_ch
    acc = tl.zeros([DP2], dtype=tl.float32)
    for k in tl.static_range(KS):
        if k < KS - 1:
            tap = tl.load(sbase + (DIL * k) * st_s_w, mask=m & ok & init, other=0.0)
        else:
            tap = c
        acc += tl.load(cw_ptr + ch * KS + k, mask=m, other=0.0) * tap
    y = tl.where(ok, acc * tl.sigmoid(acc), 0.0)
    mem = (y + vg).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + t64 * N * D + s * D + offs, (xs + mem).to(tl.bfloat16), mask=m)
    if ok:
        # shift the history by one position, the current input last; every warp has read it first
        nxt = tl.zeros([DP2], dtype=tl.float32)
        tl.debug_barrier()
        for i in tl.static_range(STATE_LEN):
            if i < STATE_LEN - 1:
                nxt = tl.load(sbase + (i + 1) * st_s_w, mask=m & init, other=0.0)
            else:
                nxt = c
            tl.debug_barrier()
            tl.store(sbase + i * st_s_w, nxt, mask=m)


def engram_decode_combine(x_res: torch.Tensor, layer_out: torch.Tensor | None, coef: torch.Tensor | None,
                          kv: torch.Tensor, norm_h: torch.Tensor, norm_k: torch.Tensor, norm_v: torch.Tensor,
                          conv_weight: torch.Tensor, conv_state: torch.Tensor, state_idx: torch.Tensor,
                          has_init: torch.Tensor | None, n_real: int, eps: float, dilation: int,
                          out: torch.Tensor | None = None) -> torch.Tensor:
    """[T, N, D] bf16: the streams after the post (if layer_out is given) and the Engram; updates conv_state
    ([slots, N D, state_len] canonical view, fp32) for the rows < n_real with a real slot."""
    T, N, D = x_res.shape
    if out is None:
        out = torch.empty((T, N, D), dtype=torch.bfloat16, device=x_res.device)
    assert out.is_contiguous()
    if T:
        KS = conv_weight.shape[1]
        state_len = conv_state.shape[2]
        assert state_len == (KS - 1) * dilation and kv.stride(1) == 1 and n_real <= min(T, state_idx.shape[0])
        do_post = layer_out is not None
        for _ in range(repeats("engram")):
            _engram_decode_kernel[(T, N)](
                x_res.contiguous(), layer_out if do_post else x_res, coef if do_post else x_res, kv, kv.stride(0),
                norm_h, norm_k, norm_v, conv_weight, conv_state, conv_state.stride(0), conv_state.stride(1),
                conv_state.stride(2), state_idx, has_init if has_init is not None else state_idx, out, n_real, eps,
                D ** 0.5, N=N, D=D, DP2=triton.next_power_of_2(D), KS=KS, DIL=dilation, STATE_LEN=state_len,
                DO_POST=do_post, HAS_INIT=has_init is not None, PDL=pdl("engram"), launch_pdl=pdl("engram"), num_warps=4)
    return out


# ---- batches with prefill tokens: the per-token part in one kernel, vLLM's dilated convolution, one combine ---------

@triton.jit
def _engram_gate_kernel(x_ptr, layer_ptr, coef_ptr,     # streams [T, N, D] bf16 (before the post if DO_POST), ...
                        kv_ptr, kv_stride,              # [T, (N + 1) D] bf16: keys | value
                        nh_ptr, nk_ptr, nv_ptr,         # [D] fp32 norm weights
                        c_ptr,                          # [T, N D] fp32 out: RMSNorm_v(gate * value), the conv input
                        gate_ptr,                       # [T, N] fp32 out
                        xs_ptr,                         # [T, N, D] bf16 out: the posted streams (DO_POST)
                        eps, sqrt_d,
                        N: tl.constexpr, D: tl.constexpr, DP2: tl.constexpr, DO_POST: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    offs = tl.arange(0, DP2)
    m = offs < D
    xb = x_ptr + t * N * D
    if DO_POST:
        cin = coef_ptr + t * (N + N * N)
        xs = tl.load(cin + s) * tl.load(layer_ptr + t * D + offs, mask=m, other=0.0).to(tl.float32)
        for j in tl.static_range(N):
            xs += tl.load(cin + N + s * N + j) * tl.load(xb + j * D + offs, mask=m, other=0.0).to(tl.float32)
        xs = xs.to(tl.bfloat16)
        tl.store(xs_ptr + t * N * D + s * D + offs, xs, mask=m)
        xs = xs.to(tl.float32)
    else:
        xs = tl.load(xb + s * D + offs, mask=m, other=0.0).to(tl.float32)
    krow = kv_ptr + t * kv_stride
    key = tl.load(krow + s * D + offs, mask=m, other=0.0).to(tl.float32)
    value = tl.load(krow + N * D + offs, mask=m, other=0.0).to(tl.float32)
    nh = tl.load(nh_ptr + offs, mask=m, other=0.0)
    nk = tl.load(nk_ptr + offs, mask=m, other=0.0)
    nv = tl.load(nv_ptr + offs, mask=m, other=0.0)
    z = tl.sum(_rms(xs, nh, eps, D) * _rms(key, nk, eps, D), axis=0) / sqrt_d
    sgn = tl.where(z > 0, 1.0, tl.where(z < 0, -1.0, 0.0))
    gate = tl.sigmoid(tl.sqrt(tl.maximum(tl.abs(z), 1e-6)) * sgn)
    tl.store(gate_ptr + t * N + s, gate)
    tl.store(c_ptr + t * N * D + s * D + offs, _rms(gate * value, nv, eps, D), mask=m)


@triton.jit
def _engram_combine_kernel(xs_ptr, y_ptr, gate_ptr, kv_ptr, kv_stride, out_ptr,
                           N: tl.constexpr, D: tl.constexpr, DP2: tl.constexpr):
    """out = bf16(x_s + bf16(SiLU(conv) + gate_s * value))"""
    t = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    offs = tl.arange(0, DP2)
    m = offs < D
    value = tl.load(kv_ptr + t * kv_stride + N * D + offs, mask=m, other=0.0).to(tl.float32)
    vg = tl.load(gate_ptr + t * N + s) * value
    y = tl.load(y_ptr + t * N * D + s * D + offs, mask=m, other=0.0)
    mem = (y + vg).to(tl.bfloat16).to(tl.float32)
    xs = tl.load(xs_ptr + t * N * D + s * D + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + t * N * D + s * D + offs, (xs + mem).to(tl.bfloat16), mask=m)


def engram_gate(x_res: torch.Tensor, layer_out: torch.Tensor | None, coef: torch.Tensor | None, kv: torch.Tensor,
                norm_h: torch.Tensor, norm_k: torch.Tensor, norm_v: torch.Tensor, eps: float):
    """-> (c [T, N D] fp32 conv input, gate [T, N] fp32, posted streams [T, N, D] bf16 (x_res itself without a
    post))."""
    T, N, D = x_res.shape
    dev = x_res.device
    do_post = layer_out is not None
    c = torch.empty((T, N * D), dtype=torch.float32, device=dev)
    gate = torch.empty((T, N), dtype=torch.float32, device=dev)
    xs = torch.empty((T, N, D), dtype=torch.bfloat16, device=dev) if do_post else x_res
    if T:
        _engram_gate_kernel[(T, N)](x_res.contiguous(), layer_out if do_post else x_res,
                                    coef if do_post else x_res, kv, kv.stride(0), norm_h, norm_k, norm_v, c, gate, xs,
                                    eps, D ** 0.5, N=N, D=D, DP2=triton.next_power_of_2(D), DO_POST=do_post,
                                    num_warps=4)
    return c, gate, xs


def engram_combine(xs: torch.Tensor, y: torch.Tensor, gate: torch.Tensor, kv: torch.Tensor,
                   out: torch.Tensor) -> None:
    T, N, D = xs.shape
    if T:
        _engram_combine_kernel[(T, N)](xs.contiguous(), y, gate, kv, kv.stride(0), out, N=N, D=D,
                                       DP2=triton.next_power_of_2(D), num_warps=4)
