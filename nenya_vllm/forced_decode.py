# SPDX-License-Identifier: Apache-2.0
"""A logits processor (vLLM Model Runner V2 interface) that forces each request's tokens to a given sequence, for
accuracy tests of the decode path: with
SamplingParams(extra_args={"forced": [ids...], "prompt_len": n}) a request generates exactly those tokens, while the
returned log-probs are the model's raw ones (vLLM's default logprobs_mode="raw_logprobs"). Every build can then be
scored on identical token sequences instead of its own greedy continuations."""

import torch

from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor


class ForcedTokensLogitsProcessor(LogitsProcessor):
    MAX_LEN = 8192

    def __init__(self, vllm_config, req_states) -> None:
        n = req_states.max_num_reqs
        dev = req_states.device
        self.tok = torch.zeros((n, self.MAX_LEN), dtype=torch.long, device=dev)
        self.on = torch.zeros(n, dtype=torch.bool, device=dev)
        self.plen = torch.zeros(n, dtype=torch.long, device=dev)

    def add_request(self, req_idx: int, sampling_params) -> bool:
        extra = sampling_params.extra_args or {}
        forced = extra.get("forced")
        self.on[req_idx] = forced is not None
        if forced is None:
            return False
        assert len(forced) <= self.MAX_LEN
        self.tok[req_idx, :len(forced)] = torch.tensor(forced, dtype=torch.long)
        self.plen[req_idx] = int(extra["prompt_len"])
        return True

    def apply(self, logits: torch.Tensor, ctx) -> torch.Tensor:
        slots = ctx.expanded_idx_mapping.long()
        on = self.on[slots]
        # the row samples the token at position pos + 1, output index pos + 1 - prompt length
        step = (ctx.pos.long() + 1 - self.plen[slots]).clamp(0, self.MAX_LEN - 1)
        tok = self.tok[slots, step]
        rows = torch.arange(logits.shape[0], device=logits.device)
        keep = logits[rows, tok].clone()
        logits.masked_fill_(on[:, None], float("-inf"))
        logits[rows, tok] = torch.where(on, keep, logits[rows, tok])
        return logits
