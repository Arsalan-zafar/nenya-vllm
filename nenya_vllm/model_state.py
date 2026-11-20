# SPDX-License-Identifier: Apache-2.0
"""Model-runner state for Nenya: the n-gram context Engram hashes across chunk boundaries.

Engram's 3-gram at position t needs the token ids at t-1 and t-2. Inside the current chunk they are in input_ids;
at the start of a chunk (and for every decode step) they are earlier tokens of the request, which the runner keeps
on the GPU (`req_states.all_token_ids`). Every step this state gathers, per request, the CTX ids just before the
chunk (oldest first; <|endoftext|> before position 0, which hashes to Engram's pad id) into a fixed-address buffer
that the model receives with a matching query_start_loc. Adapted from vLLM v0.33.0's Qwen4ExpModelState (its
per-layer n-gram embedding has the same need); rollback-safe because it re-reads the history every step.
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState


class NenyaModelState(MambaHybridModelState):
    def __init__(self, vllm_config: VllmConfig, model: nn.Module, encoder_cache: EncoderCache | None,
                 device: torch.device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.ngram_context_len = max(config.engram_ngram_orders) - 1
        self.fill_token_id = int(config.eot_token_id)
        self.ngram_context = torch.full((self.max_num_reqs, self.ngram_context_len), self.fill_token_id,
                                        dtype=torch.int32, device=self.device)
        self.ngram_offsets = torch.arange(-self.ngram_context_len, 0, dtype=torch.int64, device=self.device)
        self.engram_query_start_loc = torch.zeros(self.max_num_reqs + 1, dtype=torch.int32, device=self.device)

    def _prepare_ngram_context(self, input_batch: InputBatch, req_states: RequestState) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.fill_token_id)
        if num_reqs == 0:
            return context
        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_offsets
        valid = token_indices >= 0
        token_indices.clamp_min_(0)
        tokens = req_states.all_token_ids.gpu[request_indices.unsqueeze(1), token_indices]
        context[:num_reqs].copy_(torch.where(valid, tokens, tokens.new_full((), self.fill_token_id)))
        return context

    def prepare_inputs(self, input_batch: InputBatch, req_states: RequestState) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        num_reqs_padded = input_batch.num_reqs_after_padding
        qsl = self.engram_query_start_loc
        qsl[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # unused capacity = trailing zero-length requests
        qsl[num_reqs_padded + 1:].copy_(input_batch.query_start_loc[-1])
        model_inputs.update(query_start_loc=qsl, ngram_context=self._prepare_ngram_context(input_batch, req_states))
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        qsl = self.engram_query_start_loc
        qsl[0] = 0
        per_req, extra = divmod(num_tokens, num_reqs)
        lens = torch.full((num_reqs,), per_req, dtype=qsl.dtype, device=qsl.device)
        if extra > 0:
            lens[-extra:] += 1
        torch.cumsum(lens, dim=0, out=qsl[1:num_reqs + 1])
        qsl[num_reqs + 1:].fill_(num_tokens)
        self.ngram_context.fill_(self.fill_token_id)
        model_inputs.update(query_start_loc=qsl, ngram_context=self.ngram_context)
        return model_inputs
