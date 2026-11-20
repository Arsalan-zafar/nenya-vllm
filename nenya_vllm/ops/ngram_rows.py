# SPDX-License-Identifier: Apache-2.0
# Engram's n-gram hashing on the GPU, over vLLM's flattened, chunked token layout.
#
# Structure follows vLLM v0.33.0's _ple_ngram_ids_kernel (vllm/models/qwen4_exp/common/ops/ple.py, Apache-2.0):
# a binary search over query_start_loc finds each token's request; predecessors inside the current chunk come from
# input_ids, earlier ones from the runner's per-request context (the last CTX token ids before the chunk).
#
# Nenya's semantics (modeling_nenya.NenyaModel.engram_rows), which differ from Qwen4Exp's in two ways:
#   * every token id is first mapped to its compressed id through the shipped `engram_lookup` buffer;
#   * a predecessor at or behind an <|endoftext|> reads as engram_pad_id (= the compressed id of <|endoftext|>),
#     and the context is filled with <|endoftext|> before position 0 -- exactly the release's document masking and
#     "nothing before the start" padding.
# Row of (order n, head h) = offset[n, h] + ((XOR_{s<n} cid[t-s] * mult[s]) mod size[n, h]).
# The multipliers are bounded so the products fit in int64 (checked at load time).

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "binary_search_iters"])
def _nenya_ngram_rows_kernel(
    input_ids_ptr,
    qsl_ptr,
    ctx_ptr,
    token_map_ptr,
    multipliers_ptr,
    sizes_ptr,
    offsets_ptr,
    out_ptr,
    num_tokens,
    num_reqs,
    eot_token_id,
    pad_id,
    binary_search_iters,
    CTX: tl.constexpr,
    HEADS: tl.constexpr,
    NUM_TABLES: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    BLOCK_H: tl.constexpr = triton.next_power_of_2(NUM_TABLES)
    pid = tl.program_id(0)
    tok = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    tok_mask = tok < num_tokens

    lo = tl.full([BLOCK_T], 1, tl.int32)
    hi = tl.full([BLOCK_T], num_reqs + 1, tl.int32)
    for _ in range(binary_search_iters):
        mid = (lo + hi) // 2
        qmid = tl.load(qsl_ptr + mid, mask=tok_mask & (mid <= num_reqs), other=0)
        pred = qmid <= tok
        lo = tl.where(pred, mid + 1, lo)
        hi = tl.where(pred, hi, mid)
    req = tl.minimum(lo - 1, num_reqs - 1).to(tl.int64)
    req_start = tl.load(qsl_ptr + req, mask=tok_mask, other=0)
    chunk_pos = tok - req_start

    cur_raw = tl.load(input_ids_ptr + tok, mask=tok_mask, other=0).to(tl.int64)
    cur = tl.load(token_map_ptr + cur_raw, mask=tok_mask, other=0).to(tl.int64)
    mult0 = tl.load(multipliers_ptr).to(tl.int64)
    g = tl.arange(0, BLOCK_H)
    g_mask = g < NUM_TABLES
    order = g // HEADS + 2
    mixed = (cur * mult0)[:, None] + tl.zeros([BLOCK_T, BLOCK_H], dtype=tl.int64)

    crossed = tl.zeros([BLOCK_T], dtype=tl.int1)
    for shift in tl.static_range(1, CTX + 1):
        in_step = chunk_pos >= shift
        step_raw = tl.load(input_ids_ptr + tok - shift, mask=tok_mask & in_step, other=0)
        ctx_col = CTX - shift + chunk_pos
        ctx_raw = tl.load(ctx_ptr + req * CTX + ctx_col, mask=tok_mask & (~in_step), other=0)
        raw = tl.where(in_step, step_raw, ctx_raw).to(tl.int64)
        cid = tl.load(token_map_ptr + raw, mask=tok_mask, other=0).to(tl.int64)
        # behind a document boundary every predecessor reads as the pad id
        cid = tl.where(crossed, pad_id, cid)
        crossed = crossed | (raw == eot_token_id)
        mult = tl.load(multipliers_ptr + shift).to(tl.int64)
        term = cid * mult
        mixed = mixed ^ tl.where((order > shift)[None, :], term[:, None], 0)

    sizes = tl.load(sizes_ptr + g, mask=g_mask, other=1).to(tl.int64)[None, :]
    offs = tl.load(offsets_ptr + g, mask=g_mask, other=0).to(tl.int64)[None, :]
    rem = mixed % sizes
    rem = tl.where(rem < 0, rem + sizes, rem)  # torch.remainder semantics (never hit: products are positive)
    tl.store(out_ptr + tok[:, None] * NUM_TABLES + g[None, :], rem + offs,
             mask=tok_mask[:, None] & g_mask[None, :])


def nenya_ngram_rows(input_ids: torch.Tensor, query_start_loc: torch.Tensor, ngram_context: torch.Tensor,
                     token_map: torch.Tensor, multipliers: torch.Tensor, table_sizes: torch.Tensor,
                     table_offsets: torch.Tensor, eot_token_id: int, pad_id: int, heads: int,
                     out: torch.Tensor | None = None) -> torch.Tensor:
    """[T, n_orders * heads] int64 table rows (offsets included) for every token of the batch.

    input_ids [T]; query_start_loc [R + 1] (trailing padding requests repeat the last offset); ngram_context
    [R, CTX] raw token ids before each request's chunk (oldest first, <|endoftext|> before position 0);
    token_map [vocab] compressed ids; multipliers [max_order]; table_sizes / table_offsets [n_orders * heads].
    """
    input_ids = input_ids.reshape(-1)
    num_tokens = input_ids.shape[0]
    num_tables = table_sizes.numel()
    if out is None:
        out = torch.empty((num_tokens, num_tables), dtype=torch.int64, device=input_ids.device)
    if num_tokens == 0:
        return out
    num_reqs = query_start_loc.numel() - 1
    block_t = 16
    _nenya_ngram_rows_kernel[(triton.cdiv(num_tokens, block_t),)](
        input_ids, query_start_loc, ngram_context, token_map, multipliers, table_sizes, table_offsets, out,
        num_tokens, num_reqs, eot_token_id, pad_id, binary_search_iters=max(num_reqs, 1).bit_length(),
        CTX=ngram_context.shape[1], HEADS=heads, NUM_TABLES=num_tables, BLOCK_T=block_t, num_warps=4,
    )
    return out


def nenya_ngram_rows_reference(input_ids, query_start_loc, ngram_context, token_map, multipliers, table_sizes,
                               table_offsets, eot_token_id, pad_id, heads):
    """Same result in PyTorch, one request at a time (tests)."""
    input_ids = input_ids.reshape(-1).long()
    qsl = query_start_loc.tolist()
    ctx_len = ngram_context.shape[1]
    num_orders = table_sizes.numel() // heads
    sizes = table_sizes.view(num_orders, heads).long()
    offs = table_offsets.view(num_orders, heads).long()
    mult = multipliers.long()
    out = torch.empty((input_ids.numel(), num_orders * heads), dtype=torch.long, device=input_ids.device)
    for r in range(len(qsl) - 1):
        s, e = qsl[r], qsl[r + 1]
        if e <= s:
            continue
        raw = torch.cat([ngram_context[r].long(), input_ids[s:e]])
        for i in range(e - s):
            t = ctx_len + i
            cids = [int(token_map[raw[t]])]
            crossed = False
            for shift in range(1, ctx_len + 1):
                rw = int(raw[t - shift])
                cids.append(pad_id if crossed else int(token_map[rw]))
                crossed = crossed or rw == eot_token_id
            for oi in range(num_orders):
                order = oi + 2
                mix = cids[0] * int(mult[0])
                for k in range(1, order):
                    mix ^= cids[k] * int(mult[k])
                out[s + i, oi * heads:(oi + 1) * heads] = (mix % sizes[oi]) + offs[oi]
    return out
