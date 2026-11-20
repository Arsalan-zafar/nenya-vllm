# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for Nenya's chat format: `--enable-auto-tool-choice --tool-call-parser nenya`.

The format and all of its parsing live in olmo3_chat.py; this class adapts it to vLLM 0.33:
  * text before <function_calls> is returned (and streamed) as content;
  * the tool name is everything before "(" (dotted, hyphenated and spaced names work; nothing is mapped back);
  * values are JSON, Python literals, or raw <raw_code>...</raw_code> blocks (multi-line strings, e.g. code);
  * special-token strings escaped in data (a zero-width space after the first character) are unescaped;
  * a turn whose call block does not parse is returned as plain text, unchanged, rather than as broken calls;
  * when streaming, each call is sent as soon as its closing parenthesis arrives, with its full arguments;
  * at the end of a stream (a stop id in the last delta, or the request's max_tokens reached) the text held back
    as a possible tag start (a trailing "<" or "|") is sent too. A stop string, or a length cut-off that is not the
    request's own max_tokens (unset, or larger than the room left in the context), is not seen, and the held text
    is lost. If the server does not stop on that stop id, parsing goes on (a warning is logged once).

tool_choice "required" / a named tool (and, in vLLM 0.33, parallel_tool_calls=False, or "auto" with a strict tool):
the structural tag "nenya_olmo3" (structural_tag.py) constrains decoding to well-formed calls in the native format,
with arguments that match the tool's JSON schema when the tool is strict (or the server runs --tool-strict-level
parameter); an unconstrained string argument may also be a <raw_code> block, as training writes multi-line strings.
vLLM's default would force a JSON array the model never saw. A value and the call's closing ")" can be one merged
token (`")`, `")\n`, `"])`, `"})`): structural_tag.py explains the bug that masked them. Where the reply ends with
the call block, the block may close with <|im_end|>, as the format closes a call turn (also in structural_tag.py).

Requests without tools never reach this parser: nenya_vllm.serving's middleware unescapes those replies.

The checkpoint's tokenizer must name ids 100270/100271 <raw_code>/</raw_code>, as non-special tokens (run
`python -m nenya_vllm.olmo3_chat export-hf <checkpoint>` once); the parser warns if they are unnamed or special.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.renderers.chat_utils import make_tool_call_id
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.structural_tag_registry import get_function_parameters, register_vllm_structural_tag

from . import olmo3_chat as oc
from . import structural_tag as st

logger = init_logger(__name__)
CALLS_BEGIN, CALLS_END = oc.S_CALLS_BEGIN, oc.S_CALLS_END
_checked_tokenizers: set[int] = set()


class FastCallStream(oc.CallStream):
    """CallStream with a shortcut for text before the call block, where it spends most of a stream (~15 us per token
    checking 23 tag patterns against the text's end). Before <function_calls>, CallStream releases everything up to
    a possible partial tag or escaped special-token string, all of which start with "<" or "|"; if the unreleased
    text contains neither, it would release all of it unchanged (unescaping is the identity there). Same output as
    CallStream.feed for every input.

    The stream can go on after finish(). The parser calls finish() on a delta that ends with a stop id, but a server
    whose generation config lacks that id does not stop there. When more text arrives, the stream reopens:
      * the errors that finish() recorded for the end (an unclosed block, an unterminated call) are dropped, and the
        call scanner gets back the state it had before finish() scanned the held-back text;
      * stop strings that finish() dropped from the end of the text (a literal <|im_end|>, which the text holds when
        the server decodes with skip_special_tokens=False) are text after all: they are sent with what follows;
      * the shortcut stays off for the rest of the stream: finish() released the held-back text, which may be the
        "<" that starts <function_calls>, and only the full scan finds that tag again;
      * the held-back text finish() released is not sent again, and stays in the content as written, even when the
        text that follows turns it into something the content leaves out or changes: the start of <function_calls>
        (a "<" up to "<function_calls"), of an escaped special-token string (sent with its zero-width space), or of
        a stop string that ends the reply (sent as text).
    The next finish() releases what is held back at the new end. Calls and errors end up as without the early
    finish()."""

    reopened = False
    _before_end = None      # (number of errors, the scanner's state, released) before finish() checked the end

    def feed(self, text):
        if self.finished:
            if len(text) <= len(self.text):
                return "", []
            self._reopen()
        if self.block_start is None and not self.reopened:
            new = text[self.released:]
            if "<" not in new and "|" not in new:
                self.text, self.released = text, len(text)
                return new, []
        return super().feed(text)

    def finish(self, text=None):
        if self.finished:
            if text is None or len(text) <= len(self.text):
                return "", []
            self._reopen()
        out, calls = self.feed(text) if text is not None else ("", [])
        scanner = None if self.scanner is None else copy.deepcopy(vars(self.scanner))
        released = self.released
        if self.block_start is None:
            # CallStream.finish() sends the held-back text without the stop strings that end it, and marks those
            # released too; a reopened stream starts again just after the text it sent
            released += len(_without_stop_strings(self.text[released:]))
        self._before_end = (len(self.errors), scanner, released)
        more, more_calls = super().finish()
        return out + more, calls + more_calls

    def _reopen(self):
        n_errors, scanner, released = self._before_end
        del self.errors[n_errors:]
        if scanner is not None:
            vars(self.scanner).update(scanner)
        if self.block_start is None:
            self.released = released
        self.finished, self.reopened = False, True


def _without_stop_strings(text):
    """`text` without the stop strings that end it (<|im_end|>, <|endoftext|>, in any order and number), as
    CallStream.finish() drops them (olmo3_chat._strip_stops)."""
    while True:
        for stop in (oc.S_IM_END, oc.S_EOT):
            if text.endswith(stop):
                text = text[:-len(stop)]
                break
        else:
            return text


def tokenizer_has_raw_code(tokenizer) -> bool | None:
    """True if the tokenizer writes <raw_code></raw_code> as ids 100270/100271 and keeps them when it decodes with
    skip_special_tokens=True, vLLM's default (the checkpoint went through export-hf); False if not; None if it cannot
    be asked."""
    try:
        ids, text = [oc.RAW_BEGIN, oc.RAW_END], oc.S_RAW_BEGIN + oc.S_RAW_END
        return (tokenizer.encode(text, add_special_tokens=False) == ids
                and tokenizer.decode(ids, skip_special_tokens=True) == text)
    except Exception:  # noqa: BLE001 - stand-in tokenizers in tests
        return None


class NenyaToolParser(ToolParser):
    structural_tag_model = "nenya_olmo3"

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self._stream = FastCallStream()
        self._errors_logged = 0
        self._end_id = None                     # the last token id of the delta the parser took for the last one
        if id(tokenizer) not in _checked_tokenizers:
            _checked_tokenizers.add(id(tokenizer))
            if tokenizer_has_raw_code(tokenizer) is False:
                logger.warning("Nenya tool parser: the tokenizer does not map <raw_code>/</raw_code> to ids "
                               "100270/100271 as non-special tokens, so multi-line arguments are not written or read "
                               "in the trained format. Run `python -m nenya_vllm.olmo3_chat export-hf <checkpoint>` "
                               "once.")

    # ---- complete output ----
    def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
        turn = oc.parse_assistant(model_output)
        if turn.errors:
            logger.warning("Nenya tool parser: returning the output as text (%s)", "; ".join(turn.errors)[:300])
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[],
                                                content=oc.unescape_text(model_output))
        if turn.warnings:
            logger.warning("Nenya tool parser: %s", "; ".join(turn.warnings)[:300])
        if not turn.tool_calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content=turn.content)
        calls = [ToolCall(function=FunctionCall(name=c.name, arguments=json.dumps(c.arguments, ensure_ascii=False)))
                 for c in turn.tool_calls]
        return ExtractedToolCallInformation(tools_called=True, tool_calls=calls, content=turn.content or None)

    # ---- streaming ----
    def extract_tool_calls_streaming(self, previous_text: str, current_text: str, delta_text: str,
                                     previous_token_ids: Sequence[int], current_token_ids: Sequence[int],
                                     delta_token_ids: Sequence[int], request) -> DeltaMessage | None:
        was_reopened, ended_on = self._stream.reopened, self._end_id
        if stream_ended(current_token_ids, delta_token_ids, request):
            content, calls = self._stream.finish(current_text)
            if delta_token_ids:
                self._end_id = int(delta_token_ids[-1])
        else:
            content, calls = self._stream.feed(current_text)
        if self._stream.reopened and not was_reopened:
            _warn_reopened(ended_on)                # the id of the delta taken for the last one, not this delta's
        self._errors_logged = min(self._errors_logged, len(self._stream.errors))   # a reopened stream drops some
        for err in self._stream.errors[self._errors_logged:]:
            logger.warning("Nenya tool parser: skipping an unparsable streamed call (%s)", err)
        self._errors_logged = len(self._stream.errors)
        deltas = []
        for call in calls:
            arguments = json.dumps(call.arguments, ensure_ascii=False)
            index = len(self.prev_tool_call_arr)
            self.prev_tool_call_arr.append({"name": call.name, "arguments": call.arguments})
            self.streamed_args_for_tool.append(arguments)
            self.current_tool_id = index
            deltas.append(DeltaToolCall(index=index, id=make_tool_call_id(), type="function",
                                        function=DeltaFunctionCall(name=call.name, arguments=arguments)))
        if not content and not deltas:
            return None
        delta = DeltaMessage()
        if content:
            delta.content = content
        if deltas:
            delta.tool_calls = deltas
        return delta


_reopen_warned = False


def _warn_reopened(token_id):
    """Once per process: a reply went on after the delta the parser took for its last one."""
    global _reopen_warned
    if _reopen_warned:
        return
    _reopen_warned = True
    logger.warning("Nenya tool parser: a reply went on after a delta that ended with token id %s, which the parser "
                   "took for the reply's end. The server does not stop on that id, so streamed replies may show a "
                   "stray '<' or '|'. List <|im_end|> (100265) and <|endoftext|> (100257) in the generation config's "
                   "eos_token_id: `python -m nenya_vllm.preflight MODEL` checks it, and serve/serve.sh runs that "
                   "check.", token_id)


def stream_ended(current_token_ids, delta_token_ids, request) -> bool:
    """True on a stream's last delta, as far as the parser can tell: vLLM gives it no end-of-stream call.

    The last delta ends with a stop id, or the reply reached the request's max_tokens. vLLM keeps the stop id in the
    token ids even when it drops its text. The stop ids are <|im_end|>, <|endoftext|> and the request's
    stop_token_ids. With ignore_eos, vLLM stops only on the request's stop_token_ids, so only those count then.
    The stop-id rule needs the delta's token ids, the max_tokens rule the reply's (current_token_ids); without
    either, that rule is skipped. Not seen: a stop string, or a length cut-off that is not the request's own
    max_tokens (max_tokens unset, or larger than the room left in the context, where vLLM stops at the context's
    end); there the held-back text ("<" or "|") is lost. If the server does not stop on a stop id after all, the
    stream goes on and FastCallStream reopens."""
    if delta_token_ids:
        stops = set(getattr(request, "stop_token_ids", None) or ())
        if not getattr(request, "ignore_eos", False):
            stops.update(oc.STOP_IDS)
        if int(delta_token_ids[-1]) in stops:
            return True
    limit = getattr(request, "max_completion_tokens", None) or getattr(request, "max_tokens", None)
    return (isinstance(limit, int) and limit > 0 and current_token_ids is not None
            and len(current_token_ids) >= limit)


# ---------------------------------------------------------------------------------------------------------------
# Structural tag (built in structural_tag.py): calls inside <function_calls> ... </function_calls>, one per line
# ---------------------------------------------------------------------------------------------------------------
_raw_allowed = st.raw_allowed
_value_format = st.value_format


def _kwargs_format(function):
    return st.kwargs_format(function, get_function_parameters)


@register_vllm_structural_tag("nenya_olmo3")
def get_nenya_olmo3_structural_tag(tools, builtin_tools, tool_choice, reasoning, token_suffix="", **kwargs):
    """vLLM's hook. Extra keyword arguments from a newer vLLM are accepted; parallel_tool_calls=False, if passed,
    allows one call: one per block, and under "auto" the reply ends after the first block."""
    del builtin_tools, reasoning, token_suffix
    return st.build(tools, tool_choice, get_function_parameters,
                    single_call=kwargs.get("parallel_tool_calls") is False)
