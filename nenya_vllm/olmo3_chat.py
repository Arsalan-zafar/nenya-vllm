#!/usr/bin/env python3
"""Nenya's chat format: the OLMo 3 Instruct template with Nenya's conventions (below).

One module renders and parses the format, so everything that uses it sees the same tokens:

  serving, clients            render(conv, add_generation_prompt=True), parse_assistant(text), CallStream (streaming)
  vLLM chat template          chat_template() / olmo3_chat_template.jinja (generated from the constants below)
  vLLM tool parser            nenya_vllm/tool_parser.py, parser "nenya" (a thin adapter over parse_assistant /
                              CallStream)
  HF checkpoint files         python -m nenya_vllm.olmo3_chat export-hf <dir>   (token names, stop ids, chat template)
  training documents          training_sample(conv)              -> ids + per-token loss mask, or None if it is still
                                                                    over 16K after tool outputs are cut (drop it);
                                                                    what the cut did: .cut_report (CutReport);
                              render_text() writes this format's special tokens as text
  token-in/token-out episodes Episode(conv)                      -> prompt ids; add_assistant(sampled ids) appends them
                                                                    unchanged; add_environment / add_user glue the next
                                                                    turn on without re-rendering earlier turns

The format, for a conversation with tools (special tokens in <...>, each one id):

  <|im_start|>system
  {IDENTITY} {TOOL_SENTENCE} <functions>[{"type": "function", "function": {...}}]</functions><|im_end|>
  <|im_start|>user
  What's the weather in Paris and Madrid?<|im_end|>
  <|im_start|>assistant
  <function_calls>get_weather(city="Paris", unit="celsius")
  get_weather(city="Madrid", unit="celsius")</function_calls><|im_end|>
  <|im_start|>environment
  {"temp": 18}
  {"temp": 24}<|im_end|>
  <|im_start|>assistant
  Paris is 18 °C; Madrid is 24 °C.<|endoftext|>

Conventions:
  * one system layout: IDENTITY, the conversation's own system text if it has one (its own paragraph), then the tool
    sentence and ` <functions>{tools}</functions>`, or the no-tools sentence and ` <functions></functions>`;
  * tool results: every result of a call block in ONE environment turn, one per line (consecutive tool messages merge;
    the environment Message keeps each result in .parts, so a long turn is cut result by result);
  * a turn that makes calls closes with <|im_end|>; only a final text answer closes with <|endoftext|>, which is also
    Nenya's document boundary (a training sample that ends with a call turn gets an extra, untrained <|endoftext|>);
  * tags are special tokens only in text this module writes. Every piece of data (system/user/tool text, assistant
    text, tool schemas, call arguments) is escaped: a special-token string in it gets a zero-width space after its
    first character, so it can never become a control token, in training or in vLLM (whose tokenizer would otherwise
    turn a literal "<|im_end|>" in a tool result into the real token);
  * a string argument with a line break inside it (not only at its ends) is written raw between <raw_code> and
    </raw_code> (ids 100270/100271), with no JSON escaping; every other value is JSON (json.dumps, ensure_ascii off);
  * tool names are written exactly as given: the parser takes everything before the first "(" as the name, so dotted,
    hyphenated or spaced names work and nothing has to be mapped back. Names may not contain "(", ")", quotes or line
    breaks; argument names may not contain "=", ",", brackets, quotes or line breaks (normalize() raises).
  * assistant text is stripped of surrounding whitespace; no reasoning text is rendered (`reasoning_content` is dropped).

Loss mask (training=True): 1 on assistant text, the call block and the turn's closing <|im_end|> / <|endoftext|>;
0 on everything else (system, user, environment, role headers, the newline after <|im_end|>, the separator
<|endoftext|>), and on every token of an assistant turn whose `train` is False (an erroneous step kept as context).

Usage
    import olmo3_chat as oc
    r = oc.training_sample({"tools": tools, "messages": messages})          # r.ids, r.mask, r.text; None = drop
    prompt = oc.render({"tools": tools, "messages": messages}, add_generation_prompt=True).ids
    ep = oc.Episode({"tools": tools, "messages": messages})                  # token in, token out
    ep.add_assistant(sampled_ids); ep.add_environment(['{"temp": 18}'])
    turn = oc.parse_assistant(text)                                          # turn.content, turn.tool_calls

    python olmo3_chat.py write-template [--out FILE]   # regenerate olmo3_chat_template.jinja
    python olmo3_chat.py export-hf DIR                 # patch a checkpoint's tokenizer / generation config in place
    python olmo3_chat.py demo                          # print a rendering with its loss mask
"""

from __future__ import annotations

import argparse
import ast
import bisect
import copy
import functools
import itertools
import json
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field

# =====================================================================================================================
# Tokens and fixed text
# =====================================================================================================================
# OLMo 3 / 3.1 Instruct's added tokens (allenai/Olmo-3-7B-Instruct and Olmo-3.1-32B-Instruct tokenizer.json), with the
# two unused slots 100270/100271 (<|extra_id_1|>/<|extra_id_2|> there) named <raw_code>/</raw_code> for Nenya.
SPECIAL_TOKENS = {
    "<|extra_id_0|>": 100256, "<|endoftext|>": 100257, "<|fim_prefix|>": 100258, "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260, "|||PHONE_NUMBER|||": 100261, "|||EMAIL_ADDRESS|||": 100262,
    "|||IP_ADDRESS|||": 100263, "<|im_start|>": 100264, "<|im_end|>": 100265, "<functions>": 100266,
    "</functions>": 100267, "<function_calls>": 100268, "</function_calls>": 100269, "<raw_code>": 100270,
    "</raw_code>": 100271, "<|extra_id_3|>": 100272, "<|extra_id_4|>": 100273, "<|extra_id_5|>": 100274,
    "<|extra_id_6|>": 100275, "<|endofprompt|>": 100276, "<|pad|>": 100277,
}
N_TEXT_TOKENS = 100256                 # cl100k_base's ranks; every id above is an added token
TOKENIZER_VOCAB = 100278               # ids the tokenizer emits
PADDED_VOCAB = 100352                  # rows of Nenya's embedding / head
PAD_ID = TOKENIZER_VOCAB               # best-fit padding id; the training loss ignores it
EOT, IM_START, IM_END = 100257, 100264, 100265
FUNCTIONS_BEGIN, FUNCTIONS_END, CALLS_BEGIN, CALLS_END, RAW_BEGIN, RAW_END = range(100266, 100272)
STOP_IDS = (IM_END, EOT)               # generation stops on either; list both in generation_config.json
# tokens Nenya's HF tokenizer must treat as special (skipped when decoding); the tool tags and <raw_code> stay
# non-special so they survive skip_special_tokens=True and reach the tool parser, as in OLMo 3 Instruct
HF_SPECIAL = {"<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|im_start|>", "<|im_end|>",
              "<|endofprompt|>", "<|pad|>"}

S_EOT, S_IM_START, S_IM_END = "<|endoftext|>", "<|im_start|>", "<|im_end|>"
S_FUNCTIONS_BEGIN, S_FUNCTIONS_END = "<functions>", "</functions>"
S_CALLS_BEGIN, S_CALLS_END = "<function_calls>", "</function_calls>"
S_RAW_BEGIN, S_RAW_END = "<raw_code>", "</raw_code>"

# The system turn. IDENTITY is Nenya's, naming its maker (after the OLMo 3 Instruct template's "You are a helpful
# function-calling AI assistant."): the renderer and the generated chat template open with it. The two sentences
# below are the tool-use instructions of OLMo 3's instruction data.
IDENTITY = "You are Nenya, a helpful function-calling AI assistant built by arsalanzafar.ai."
TOOL_SENTENCE = ("You are provided with function signatures within <functions></functions> XML tags. You may call "
                 "one or more functions to assist with the user query. Output any function calls within "
                 "<function_calls></function_calls> XML tags. Don't make assumptions about what values to plug into "
                 "functions.")
NO_TOOLS_SENTENCE = "You do not currently have access to any functions."
ELISION_NOTE = "[earlier steps elided]"

# OLMo's pre-tokeniser regex, as in pretraining/dataloader.py and the training script (tiktoken's own cl100k regex differs).
OLMO_PAT = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"""
            r""" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "olmo3_chat_template.jinja")


@functools.lru_cache(maxsize=1)
def encoding():
    """Nenya's tokenizer as a tiktoken Encoding: cl100k_base's merges, OLMo's regex, SPECIAL_TOKENS. Identical ids to
    the HF tokenizer (checked against transformers 5.22.2) once export-hf has renamed 100270/100271."""
    import tiktoken
    base = tiktoken.get_encoding("cl100k_base")
    assert len(base._mergeable_ranks) == N_TEXT_TOKENS
    return tiktoken.Encoding(name="nenya_olmo3", pat_str=OLMO_PAT, mergeable_ranks=base._mergeable_ranks,
                             special_tokens=SPECIAL_TOKENS)


@functools.lru_cache(maxsize=1)
def _token_bytes():
    """Byte length of every id < TOKENIZER_VOCAB, to place tokens on the byte axis of the rendered text."""
    import numpy as np
    enc = encoding()
    return np.array([len(enc.decode_single_token_bytes(i)) for i in range(TOKENIZER_VOCAB)], dtype=np.int64)


def decode(ids):
    """Text of `ids`, special tokens as their strings; padding ids (>= TOKENIZER_VOCAB) are dropped."""
    return encoding().decode([int(i) for i in ids if int(i) < TOKENIZER_VOCAB])


# =====================================================================================================================
# Escaping: data never contains a special-token string
# =====================================================================================================================
ZWSP = "\u200b"
_ESCAPES = [(t, t[0] + ZWSP + t[1:]) for t in SPECIAL_TOKENS]


def escape_text(s):
    """Insert a zero-width space after the first character of every special-token string in `s`."""
    for raw, esc in _ESCAPES:
        if raw in s:
            s = s.replace(raw, esc)
    return s


def unescape_text(s):
    for raw, esc in _ESCAPES:
        if esc in s:
            s = s.replace(esc, raw)
    return s


def _unescape_obj(v):
    if isinstance(v, str):
        return unescape_text(v)
    if isinstance(v, list):
        return [_unescape_obj(x) for x in v]
    if isinstance(v, dict):
        return {unescape_text(k) if isinstance(k, str) else k: _unescape_obj(x) for k, x in v.items()}
    return v


# =====================================================================================================================
# Conversations
# =====================================================================================================================
@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)

    def openai(self, call_id=None):
        """OpenAI `tool_calls` entry (arguments as a JSON string)."""
        out = {"type": "function", "function": {"name": self.name,
                                                "arguments": json.dumps(self.arguments, ensure_ascii=False)}}
        if call_id is not None:
            out["id"] = call_id
        return out


@dataclass
class Message:
    role: str                                   # "user" | "assistant" | "environment"
    content: str = ""
    tool_calls: list = field(default_factory=list)
    train: bool = True                          # assistant turns only: False keeps it as untrained context
    parts: list = None                          # environment turns: the tool results it merges, in call order, as
    #                                             normalize() found them ("\n".join(parts) == content); see results_of


@dataclass
class Conversation:
    system: str = ""                            # the conversation's own system text (a policy, a persona), no layout
    tools: list = field(default_factory=list)   # canonical tool dicts (see canonical_tool)
    messages: list = field(default_factory=list)
    cut_report: object = field(default=None, compare=False, repr=False)   # truncate_tool_outputs: a CutReport


def results_of(m):
    """The tool results an environment Message holds: its .parts while they still join to its content, else the
    content as one result (a Message built by hand, or one whose content was changed after normalize)."""
    parts = m.parts
    if parts and "\n".join(parts) == m.content:
        return list(parts)
    return [m.content]


class FormatError(ValueError):
    """Input that cannot be written in the format (a bad tool or argument name, unparsable arguments, ...)."""


_BAD_NAME = set('()"\'\n\r')
_BAD_KEY = set('=,()[]{}"\'\n\r')


def _check_name(name, what="tool name"):
    if not isinstance(name, str) or not name.strip() or name != name.strip() or _BAD_NAME & set(name):
        raise FormatError(f"{what} {name!r} cannot be written as a call: it must be non-empty, without surrounding "
                          f"whitespace, parentheses, quotes or line breaks")
    return name


def _check_key(key):
    if not isinstance(key, str) or not key.strip() or key != key.strip() or _BAD_KEY & set(key):
        raise FormatError(f"argument name {key!r} cannot be written as a call: it must be a non-empty string without "
                          f"surrounding whitespace, '=', ',', brackets, quotes or line breaks")
    return key


def text_of(content):
    """Message content as text: a string as is, a list of OpenAI content parts as their text joined by newlines (what
    vLLM hands a string-content template), None as ""."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text") or "")
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def canonical_tool(tool):
    """A tool as {"type": "function", "function": {"name", "description"?, "parameters"?}}: the description only if
    non-empty, the parameters only if given; every other key is dropped (vLLM's request objects carry extra ones).
    Also reads MCP / Anthropic / OpenCode tool shapes: the name under "name" (or "id"), the JSON Schema under
    "parameters", "inputSchema" or "input_schema" (OpenCode nests it one level deeper, under "jsonSchema")."""
    if isinstance(tool, str):
        tool = json.loads(tool)
    if not isinstance(tool, dict):
        raise FormatError(f"a tool must be a dict, not {type(tool).__name__}")
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    name = _check_name(fn.get("name") if fn.get("name") is not None else fn.get("id"))
    out = {"name": name}
    if fn.get("description"):
        out["description"] = fn["description"]
    params = next((fn[k] for k in ("parameters", "inputSchema", "input_schema") if fn.get(k) is not None), None)
    if isinstance(params, dict) and set(params) == {"jsonSchema"}:
        params = params["jsonSchema"]
    if params is not None:
        out["parameters"] = params
    return {"type": "function", "function": out}


def _tool_list(tools):
    if tools is None:
        return []
    if isinstance(tools, str):
        tools = json.loads(tools)
    return [canonical_tool(t) for t in tools]


def _arguments(args, where):
    if args is None or args == "":
        return {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError as e:
            raise FormatError(f"{where}: arguments are not valid JSON ({e})") from None
    if not isinstance(args, dict):
        raise FormatError(f"{where}: arguments must be a JSON object, not {type(args).__name__}")
    for k in args:
        _check_key(k)
    return dict(args)


def _tool_calls(m):
    calls = []
    for tc in m.get("tool_calls") or []:
        if isinstance(tc, ToolCall):
            calls.append(ToolCall(_check_name(tc.name), _arguments(tc.arguments, tc.name)))
            continue
        fn = tc.get("function") if isinstance(tc, dict) and isinstance(tc.get("function"), dict) else tc
        if not isinstance(fn, dict):
            raise FormatError(f"unreadable tool call {tc!r}")
        name = _check_name(fn.get("name"))
        calls.append(ToolCall(name, _arguments(fn.get("arguments"), name)))
    fc = m.get("function_calls")                # OLMo 3 data: the calls as OLMo 3 text ("null": none)
    if isinstance(fc, str) and fc.strip() not in ("", "null", "None"):
        parsed, errors = parse_call_block(fc)
        if errors:
            raise FormatError(f"unparsable function_calls {fc[:120]!r}: {errors[0]}")
        calls.extend(ToolCall(_check_name(c.name), _arguments(c.arguments, c.name)) for c in parsed)
    legacy = m.get("function_call")             # OpenAI's older single-call field
    if isinstance(legacy, dict) and legacy.get("name"):
        name = _check_name(legacy["name"])
        calls.append(ToolCall(name, _arguments(legacy.get("arguments"), name)))
    return calls


def normalize(conv, tools=None):
    """Any accepted input -> Conversation.

    Accepts a Conversation; a dict with "messages" and optional "tools" / "system"; or a bare list of messages. Messages
    are OpenAI chat messages (system, developer, user, assistant with tool_calls, tool) or this module's roles
    (environment), plus the fields of OLMo 3's data (a system message's "functions" JSON, an assistant's
    "function_calls" text).
    System and developer messages anywhere are joined (blank line between) into the conversation's system text, as
    vLLM consolidates them; consecutive tool/environment messages are merged into one turn, results one per line, and
    the turn's Message keeps the results apart in .parts (for cutting long results one by one)."""
    if isinstance(conv, Conversation):
        return conv if tools is None else Conversation(conv.system, _tool_list(tools), conv.messages)
    if isinstance(conv, dict):
        messages, system = conv.get("messages") or [], conv.get("system") or ""
        tools = conv.get("tools") if tools is None else tools
    else:
        messages, system = list(conv), ""
    system_parts = [text_of(system)] if text_of(system) else []
    dolci_tools = None
    out = []
    for m in messages:
        if isinstance(m, Message):
            m = {"role": m.role, "content": m.content, "tool_calls": m.tool_calls, "train": m.train, "parts": m.parts}
        role = m.get("role")
        if role in ("system", "developer"):
            c = text_of(m.get("content"))
            if c:
                system_parts.append(c)
            if m.get("functions") and dolci_tools is None:
                dolci_tools = m["functions"]
        elif role == "user":
            out.append(Message("user", text_of(m.get("content"))))
        elif role == "assistant":
            out.append(Message("assistant", text_of(m.get("content")), _tool_calls(m), bool(m.get("train", True))))
        elif role in ("tool", "environment", "function"):          # "function": OpenAI's legacy tool-result role
            c = text_of(m.get("content"))
            parts = m.get("parts")              # a Message's own results (an environment turn normalized before)
            if not (isinstance(parts, list) and parts and all(isinstance(p, str) for p in parts)
                    and "\n".join(parts) == c):
                parts = [c]
            if out and out[-1].role == "environment":
                out[-1].content += "\n" + c
                out[-1].parts += parts
            else:
                out.append(Message("environment", c, parts=list(parts)))
        else:
            raise FormatError(f"unsupported role {role!r}")
    return Conversation("\n\n".join(system_parts).strip(), _tool_list(tools if tools is not None else dolci_tools),
                        out)


# =====================================================================================================================
# Rendering
# =====================================================================================================================
def tools_json(tools):
    return json.dumps([canonical_tool(t) for t in tools], ensure_ascii=False)


def system_text(conv):
    """The system turn's text (between "<|im_start|>system\\n" and "<|im_end|>")."""
    src = conv.system.strip()
    text = IDENTITY + ("\n\n" + escape_text(src) + "\n\n" if src else " ")
    if conv.tools:
        return text + TOOL_SENTENCE + " " + S_FUNCTIONS_BEGIN + escape_text(tools_json(conv.tools)) + S_FUNCTIONS_END
    return text + NO_TOOLS_SENTENCE + " " + S_FUNCTIONS_BEGIN + S_FUNCTIONS_END


def is_raw_value(v):
    """A string value with a line break inside it (leading/trailing line breaks do not count): written raw."""
    return isinstance(v, str) and "\n" in v.strip("\n")


def render_value(v):
    if is_raw_value(v):
        return S_RAW_BEGIN + escape_text(v) + S_RAW_END
    return escape_text(json.dumps(v, ensure_ascii=False))


def render_call(call):
    """`name(key=value, ...)` for one ToolCall (or (name, arguments))."""
    name, args = (call.name, call.arguments) if isinstance(call, ToolCall) else call
    return (escape_text(name) + "(" + ", ".join(escape_text(k) + "=" + render_value(v) for k, v in args.items())
            + ")")


def render_calls(calls):
    return S_CALLS_BEGIN + "\n".join(render_call(c) for c in calls) + S_CALLS_END


def _spans(conv, add_generation_prompt):
    """[(text, trained)] for the whole conversation; the text pieces concatenate to the rendering."""
    spans = [(S_IM_START + "system\n" + system_text(conv) + S_IM_END + "\n", False)]
    msgs = conv.messages
    for i, m in enumerate(msgs):
        last = i == len(msgs) - 1
        if m.role == "user":
            spans.append((S_IM_START + "user\n" + escape_text(m.content) + S_IM_END + "\n", False))
        elif m.role == "environment":
            spans.append((S_IM_START + "environment\n" + escape_text(m.content) + S_IM_END + "\n", False))
        elif m.role == "assistant":
            spans.append((S_IM_START + "assistant\n", False))
            body = escape_text(m.content.strip()) + (render_calls(m.tool_calls) if m.tool_calls else "")
            if body:
                spans.append((body, m.train))
            spans.append((S_IM_END if (m.tool_calls or not last) else S_EOT, m.train))
            if not last:
                spans.append(("\n", False))
        else:
            raise FormatError(f"unsupported role {m.role!r}")
    if add_generation_prompt:
        spans.append((S_IM_START + "assistant\n", False))
    return spans


@dataclass
class Rendered:
    text: str
    ids: list
    mask: list                                  # 1 = trained target, 0 = context only
    straddles: int = 0                          # tokens that cross a trained/untrained boundary (0 by construction)
    cut_report: object = field(default=None, compare=False, repr=False)   # training_sample: the tool-output CutReport

    def trained_text(self):
        """The trained tokens' text, turn pieces separated by ' | ' (for inspection)."""
        out, cur = [], []
        for t, m in zip(self.ids, self.mask):
            if m:
                cur.append(t)
            elif cur:
                out.append(decode(cur))
                cur = []
        if cur:
            out.append(decode(cur))
        return " | ".join(out)


def _mask(ids, spans):
    """Per-token mask from the spans: a token is trained if it overlaps a trained span on the byte axis."""
    import numpy as np
    tb = _token_bytes()[np.asarray(ids, dtype=np.int64)] if ids else np.zeros(0, dtype=np.int64)
    ends = np.cumsum(tb)
    starts = ends - tb
    mask = np.zeros(len(ids), dtype=np.uint8)
    straddles, pos, prev = 0, 0, None
    for text, trained in spans:
        n = len(text.encode("utf-8"))
        b0, b1 = pos, pos + n
        pos = b1
        if not n:
            continue
        i0 = int(np.searchsorted(ends, b0, side="right"))
        i1 = int(np.searchsorted(starts, b1, side="left"))
        if trained:
            mask[i0:i1] = 1
        if prev is not None and prev != trained and i0 < len(ids) and starts[i0] < b0:
            straddles += 1                      # a token spans the switch between trained and untrained text
        prev = trained
    if pos != int(ends[-1] if len(ends) else 0):
        raise AssertionError("rendered text and token bytes disagree")
    return mask.tolist(), straddles


def _training_conv(conv):
    msgs = list(conv.messages)
    while msgs and msgs[-1].role != "assistant":
        msgs.pop()                              # nothing after the last assistant turn is supervised
    if not msgs:
        raise FormatError("a training sample needs at least one assistant turn")
    return Conversation(conv.system, conv.tools, msgs)


def render(conv, *, add_generation_prompt=False, training=False, tools=None):
    """Render a conversation.

    training=False: the conversation as vLLM's chat template renders it (closers as described above; with
    add_generation_prompt=True it ends with "<|im_start|>assistant\\n" for the model to continue).
    training=True: a training document. Messages after the last assistant turn are dropped, and the document always
    ends with exactly one <|endoftext|>: a final text answer's own (trained), or an untrained separator after a final
    call turn. Do not let the data producer append another."""
    conv = normalize(conv, tools)
    if training:
        if add_generation_prompt:
            raise ValueError("training documents have no generation prompt")
        conv = _training_conv(conv)
    spans = _spans(conv, add_generation_prompt)
    if training and spans[-1][0] != S_EOT:
        spans.append((S_EOT, False))
    text = "".join(s for s, _ in spans)
    ids = encoding().encode(text, allowed_special="all")
    mask, straddles = _mask(ids, spans)
    return Rendered(text, ids, mask, straddles)


def render_text(conv, *, training=True, tools=None):
    """The same rendering with every special token written as ordinary text: no id >= 100256 except the document's
    closing <|endoftext|>, which stays a real token (Nenya's document boundary).
    With training=True, a final text answer's closer is that real <|endoftext|> (trained); after a final call turn it
    is an untrained separator. Masks follow the same rules as render()."""
    conv = normalize(conv, tools)
    if training:
        conv = _training_conv(conv)
    spans = _spans(conv, add_generation_prompt=False)
    eot_trained = False
    if training and spans[-1][0] == S_EOT:
        eot_trained = spans.pop()[1]
    text = "".join(s for s, _ in spans)
    ids = encoding().encode_ordinary(text)
    mask, straddles = _mask(ids, spans)
    if training:
        ids.append(EOT)
        mask.append(1 if eot_trained else 0)
    return Rendered(text + (S_EOT if training else ""), ids, mask, straddles)


# =====================================================================================================================
# Episodes: token in, token out
# =====================================================================================================================
class Episode:
    """A conversation that grows turn by turn without ever being re-rendered.

    Episode(conv) renders the conversation so far (it must end with a user or environment turn) and the generation
    prompt. add_assistant(ids) appends the sampled ids exactly as sampled, stop token included (mask 1); re-tokenising
    sampled text can give different ids, and RL needs the ids it sampled. add_environment / add_user append the glue
    a full rendering would put after the sampled <|im_end|> ("\\n<|im_start|>role\\n...<|im_end|>\\n<|im_start|>
    assistant\\n", mask 0). The episode ends when a turn stops on <|endoftext|> or without a stop token (truncated)."""

    def __init__(self, conv, tools=None):
        conv = normalize(conv, tools)
        if conv.messages and conv.messages[-1].role == "assistant":
            raise FormatError("an episode starts from a conversation that ends with a user or environment turn")
        r = render(conv, add_generation_prompt=True)
        self.conversation = conv
        self.ids, self.mask = list(r.ids), [0] * len(r.ids)
        self.turns = []                         # (start, end) index range of each sampled assistant turn
        self.done = False
        self.truncated = False

    def add_assistant(self, sampled_ids):
        if self.done:
            raise RuntimeError("the episode has ended")
        sampled = [int(t) for t in sampled_ids]
        self.turns.append((len(self.ids), len(self.ids) + len(sampled)))
        self.ids += sampled
        self.mask += [1] * len(sampled)
        stop = sampled[-1] if sampled and sampled[-1] in STOP_IDS else None
        if stop != IM_END:
            self.done = True
            self.truncated = stop is None
        return self

    def _glue(self, role, content):
        if self.done:
            raise RuntimeError("the episode has ended")
        if not self.turns:
            raise RuntimeError("add the assistant turn first")
        text = ("\n" + S_IM_START + role + "\n" + escape_text(content) + S_IM_END + "\n" + S_IM_START + "assistant\n")
        glue = encoding().encode(text, allowed_special="all")
        self.ids += glue
        self.mask += [0] * len(glue)
        return self

    def add_environment(self, results):
        """Tool results of the last call block, in call order (a list of strings, or one string)."""
        return self._glue("environment", results if isinstance(results, str) else "\n".join(map(text_of, results)))

    def add_user(self, text):
        return self._glue("user", text_of(text))

    def turn_ids(self, k=-1):
        a, b = self.turns[k]
        return self.ids[a:b]

    def parse_turn(self, k=-1):
        return parse_assistant_ids(self.turn_ids(k))

    def has_raw_code(self):
        """True if any sampled turn opened a <raw_code> block."""
        return any(RAW_BEGIN in self.ids[a:b] for a, b in self.turns)


# =====================================================================================================================
# Parsing model output
# =====================================================================================================================
@dataclass
class ParsedAssistant:
    content: str                                # visible text (before the call block, plus anything after it)
    tool_calls: list                            # [ToolCall]
    errors: list                                # problems with the calls; empty when the call block is well formed
    warnings: list = field(default_factory=list)  # format deviations that do not affect the calls

    @property
    def ok(self):
        return not self.errors


class _Scanner:
    """Finds complete calls in a call block as text arrives: tracks string literals, brackets and <raw_code> blocks
    so that a ')' or a line break inside a value does not end a call. Incremental: feed() resumes where it stopped."""

    def __init__(self):
        self.text = ""
        self.i = 0
        self.depth = 0
        self.quote = None
        self.escape = False
        self.in_raw = False
        self.start = None                       # where the current call (its name) starts
        self.calls = []                         # (start, end) of complete calls
        self.junk = []                          # (start, end) of top-level text that is not a call

    def feed(self, text):
        self.text = text
        n, s = len(text), text
        while self.i < n:
            i = self.i
            if self.in_raw:
                j = s.find(S_RAW_END, i)
                if j < 0:
                    self.i = max(i, n - (len(S_RAW_END) - 1))   # keep a possibly partial end tag
                    return
                self.in_raw, self.i = False, j + len(S_RAW_END)
                continue
            c = s[i]
            if self.quote is not None:
                if self.escape:
                    self.escape = False
                elif c == "\\":
                    self.escape = True
                elif c == self.quote:
                    self.quote = None
                self.i += 1
                continue
            if c == "<" and self.depth > 0:
                if s.startswith(S_RAW_BEGIN, i):
                    self.in_raw, self.i = True, i + len(S_RAW_BEGIN)
                    continue
                if S_RAW_BEGIN.startswith(s[i:]) and n - i < len(S_RAW_BEGIN):
                    return                      # wait: maybe the start of <raw_code>
            if c in "\"'" and self.depth > 0:
                self.quote = c
            elif c in "([{":
                if self.start is None:
                    self.start = i              # a call cannot start with a bracket; recorded as junk later
                self.depth += 1
            elif c in ")]}":
                self.depth -= 1
                if self.depth <= 0:
                    self.depth = 0
                    if c == ")" and self.start is not None:
                        self.calls.append((self.start, i + 1))
                        self.start = None
            elif self.depth == 0:
                if c == "\n":
                    if self.start is not None:
                        self.junk.append((self.start, i))
                        self.start = None
                elif self.start is None and not c.isspace() and c not in ",;":
                    self.start = i              # a call's name starts (stray "," / ";" between calls are skipped)
            self.i += 1

    def finish(self):
        """After the block ended: anything unterminated."""
        problems = []
        if self.in_raw:
            problems.append("unterminated <raw_code> block")
        elif self.quote is not None:
            problems.append("unterminated string")
        if self.start is not None:
            problems.append(f"unterminated call {self.text[self.start:self.start + 80]!r}")
        problems += [f"not a call: {self.text[a:b][:80]!r}" for a, b in self.junk]
        return problems


def _split_top(s, sep):
    """Split `s` at `sep` characters outside strings, brackets and <raw_code> blocks."""
    parts, depth, quote, escape, i, start, n = [], 0, None, False, 0, 0, len(s)
    while i < n:
        if quote is None and s.startswith(S_RAW_BEGIN, i):
            j = s.find(S_RAW_END, i + len(S_RAW_BEGIN))
            i = n if j < 0 else j + len(S_RAW_END)
            continue
        c = s[i]
        if quote is not None:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == sep and depth == 0:
            parts.append(s[start:i])
            start = i + 1
        i += 1
    parts.append(s[start:])
    return parts


_JSON_NAMES = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}


def _literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        return node.value
    if isinstance(node, ast.Name) and node.id in _JSON_NAMES:
        return _JSON_NAMES[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict) and None not in node.keys:
        return {_literal(k): _literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _literal(node.operand)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v
    raise ValueError(f"not a literal: {ast.dump(node)[:80]}")


def parse_value(text):
    """One argument value: a <raw_code> block, JSON, or (leniently) a Python literal (True/None/'single quotes')."""
    t = text.strip()
    if t.startswith(S_RAW_BEGIN):
        if not t.endswith(S_RAW_END):
            raise ValueError("unterminated <raw_code> block")
        return unescape_text(t[len(S_RAW_BEGIN):-len(S_RAW_END)])
    try:
        v = json.loads(t)
    except ValueError:
        try:
            v = _literal(ast.parse(t, mode="eval").body)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"bad value {t[:80]!r}: {e}") from None
    return _unescape_obj(v)


def parse_call(text):
    """`name(key=value, ...)` -> ToolCall. The name is everything before the first '('."""
    t = text.strip()
    i = t.find("(")
    if i <= 0 or not t.endswith(")"):
        raise ValueError(f"not a call: {t[:80]!r}")
    name = unescape_text(t[:i].strip())
    if not name or "\n" in name:
        raise ValueError(f"bad tool name {name!r}")
    inner = t[i + 1:-1]
    args = {}
    if inner.strip():
        for part in _split_top(inner, ","):
            if not part.strip():
                raise ValueError(f"empty argument in {t[:80]!r}")
            kv = _split_top(part, "=")
            if len(kv) < 2:
                raise ValueError(f"positional argument {part.strip()[:60]!r} (arguments must be key=value)")
            key = unescape_text(kv[0].strip())
            if not key:
                raise ValueError(f"missing argument name in {part.strip()[:60]!r}")
            args[key] = parse_value(part[len(kv[0]) + 1:])
    return ToolCall(name, args)


def parse_call_block(block):
    """The text between <function_calls> and </function_calls> -> ([ToolCall], [problem])."""
    sc = _Scanner()
    sc.feed(block)
    calls, errors = [], []
    for a, b in sc.calls:
        try:
            calls.append(parse_call(block[a:b]))
        except ValueError as e:
            errors.append(str(e))
    return calls, errors + sc.finish()


def _strip_stops(text):
    """Drop trailing stop-token strings (<|im_end|>, <|endoftext|>, in any order and number)."""
    while True:
        for stop in (S_IM_END, S_EOT):
            if text.endswith(stop):
                text = text[:-len(stop)]
                break
        else:
            return text


def parse_assistant(text):
    """A complete assistant turn's text -> content and tool calls.

    Accepts text with or without the closing <|im_end|> / <|endoftext|> (vLLM strips them). Text before
    <function_calls> is content; so is any text after </function_calls> (a warning: the format puts nothing there).
    An unterminated block parses what is complete and reports the rest as errors."""
    text = _strip_stops(text)
    k = text.find(S_CALLS_BEGIN)
    if k < 0:
        return ParsedAssistant(unescape_text(text.strip()), [], [])
    content, rest = text[:k], text[k + len(S_CALLS_BEGIN):]
    errors, warnings = [], []
    e = rest.find(S_CALLS_END)
    if e < 0:
        block, tail = rest, ""
        errors.append("missing </function_calls>")
    else:
        block, tail = rest[:e], rest[e + len(S_CALLS_END):]
    if tail.strip():
        warnings.append(f"text after </function_calls>: {tail.strip()[:80]!r}")
    calls, problems = parse_call_block(block)
    errors += problems
    if not calls and not problems:
        errors.append("empty call block")
    visible = (content.strip() + ("\n" + tail.strip() if tail.strip() else "")).strip()
    return ParsedAssistant(unescape_text(visible), calls, errors, warnings)


def parse_assistant_ids(ids):
    """Sampled ids (stop token included or not) -> ParsedAssistant."""
    ids = [int(t) for t in ids]
    while ids and ids[-1] in STOP_IDS:
        ids.pop()
    return parse_assistant(decode(ids))


def _held_back(text, patterns):
    """Length of the longest suffix of `text` that is a proper prefix of one of `patterns`."""
    best = 0
    for p in patterns:
        for k in range(min(len(p) - 1, len(text)), best, -1):
            if text.endswith(p[:k]):
                best = k
                break
    return best


_STOPS = [S_IM_END, S_EOT]
_TAIL_HOLDBACK = [esc for _, esc in _ESCAPES] + _STOPS
_CONTENT_HOLDBACK = [S_CALLS_BEGIN] + _TAIL_HOLDBACK
_BLOCK_HOLDBACK = [S_CALLS_END] + _STOPS


def _hold(text, patterns, escapes=True):
    """Characters at the end of `text` a stream cannot release yet: a possible start of one of `patterns`, before it
    any complete stop strings (parse_assistant drops stop strings only where they end the text), and with `escapes`
    never part of a complete escaped special-token string (unescaping needs it whole, and the escaped
    |||PHONE_NUMBER||| ends with a "|" that could start another). Every pattern starts with "<" or "|", so text
    without either is never held (nenya_vllm's FastCallStream relies on this)."""
    k = _held_back(text, patterns)
    b = len(_strip_stops(text[:len(text) - k]))
    moved = escapes and b < len(text)
    while moved:
        moved = False
        for _, esc in _ESCAPES:
            s = text.find(esc, max(0, b - len(esc) + 1))
            if 0 <= s < b < s + len(esc):
                b, moved = s, True
    return len(text) - b


class CallStream:
    """Streaming parser: feed() the cumulative text, get back (new content, newly completed tool calls); finish() at
    the end of the stream.

    Content is released as soon as it cannot be the start of <function_calls>, of an escaped special-token string
    (which unescaping would shorten) or of a stop string that ends the text (which parse_assistant drops). A call is
    released as soon as its closing parenthesis arrives. Calls that do not parse are skipped and recorded in .errors.
    Text after </function_calls> is released as content. finish() releases what was held back at the end.

    Whitespace rule: the stream sends the content as written (escapes undone, the call block and the stop strings
    that end the text left out). parse_assistant() returns the same text with the whitespace around the part before
    the block and around the part after it stripped, and the two parts joined by a newline. So for a turn without
    text after </function_calls> (what the format trains), streamed + finish() content, stripped, equals
    parse_assistant(text).content exactly; with text after the block, each part matches after stripping."""

    def __init__(self):
        self.released = 0                       # characters of the raw text released as content
        self.n_calls = 0
        self.scanner = None
        self.block_start = None
        self.tail_sent = 0
        self.errors = []
        self.text = ""                          # the last cumulative text fed
        self.finished = False

    def feed(self, text):
        self.text = text
        new_calls, out = [], ""
        if self.block_start is None:
            k = text.find(S_CALLS_BEGIN)
            if k < 0:
                safe = len(text) - _hold(text, _CONTENT_HOLDBACK)
                if safe > self.released:
                    out = unescape_text(text[self.released:safe])
                    self.released = safe
                return out, new_calls
            out = unescape_text(text[self.released:k])
            self.released = k
            self.block_start = k + len(S_CALLS_BEGIN)
            self.scanner = _Scanner()
        body = text[self.block_start:]
        e = body.find(S_CALLS_END)
        # an open block: its end may be the start of </function_calls>, or a stop string that ends the text (which
        # parse_assistant drops); the scanner sees neither until it is known, so what it is fed only ever grows
        self._scan(body[:len(body) - _hold(body, _BLOCK_HOLDBACK, escapes=False)] if e < 0 else body[:e], new_calls)
        if e >= 0:
            tail = body[e + len(S_CALLS_END):]
            safe = len(tail) - _hold(tail, _TAIL_HOLDBACK)
            if safe > self.tail_sent:
                out += unescape_text(tail[self.tail_sent:safe])
                self.tail_sent = safe
        return out, new_calls

    def _scan(self, block, new_calls):
        """Feed the call block's text so far to the scanner; parse the calls it completed."""
        self.scanner.feed(block)
        for a, b in self.scanner.calls[self.n_calls:]:
            try:
                new_calls.append(parse_call(self.scanner.text[a:b]))
            except ValueError as err:
                self.errors.append(str(err))
            self.n_calls += 1

    def finish(self, text=None):
        """End of the stream: (content, newly completed calls), like feed(). Feeds `text` first if given (the final
        cumulative text; None = the text of the last feed). Then releases what feed() held back: a possible start of
        a tag, an escape or a stop string at the end (a trailing "<" or "|", say), with the stop strings that end the
        text removed and escapes undone. A stream that ended inside the call block gets "missing </function_calls>"
        in .errors, plus whatever is unterminated (parse_assistant's errors); a block without a call gets "empty call
        block". A second call returns ("", [])."""
        if self.finished:
            return "", []
        out, calls = self.feed(text) if text is not None else ("", [])
        self.finished = True
        text = self.text
        if self.block_start is None:
            rest = _strip_stops(text[self.released:])      # FastCallStream: a stale text gives "" (nothing was held)
            out += unescape_text(rest)
            self.released = max(self.released, len(text))
            return out, calls
        body = text[self.block_start:]
        e = body.find(S_CALLS_END)
        if e < 0:
            self._scan(_strip_stops(body), calls)          # the held-back end of the block, as parse_assistant reads it
            self.errors.append("missing </function_calls>")
        else:
            tail = _strip_stops(body[e + len(S_CALLS_END):])
            if len(tail) > self.tail_sent:
                out += unescape_text(tail[self.tail_sent:])
                self.tail_sent = len(tail)
        problems = self.scanner.finish()
        self.errors += problems
        if not self.n_calls and not problems:
            self.errors.append("empty call block")
        return out, calls


# =====================================================================================================================
# Long trajectories: tool outputs cut to fit, context windows
# =====================================================================================================================
MAX_TOKENS = 16384          # Nenya's context: a document this long always fits a training row (16,384 + 1 tokens) whole
TOOL_OUTPUT_TOKENS = 1536   # an environment turn longer than this is cut, each of its results on its own
MIN_RESULT_TOKENS = 48      # the least a result keeps when a turn holds so many results that a fair share is smaller
MIN_STRING_TOKENS = 16      # the shortest a cut JSON string gets (head, note, tail): what the JSON skeleton keeps
SHORT_LINE = 24             # a line this long or shorter is priced by rendering its cut at every budget from its length
#                             down to the least a cut showing the quote can take
PRICE_SPARE = 4             # a line priced up to this many tokens over the room is still rendered (a cut can take
#                             fewer tokens than its budget)
QUOTED_STRING_TOKENS = 64   # the shortest a cut JSON string holding a quote gets in an item kept for its quotes
LINES_NOTE = "[… {n} lines omitted …]"         # a run of omitted lines
ITEMS_NOTE = "[… {n} more items …]"            # a JSON list's elided items (a string item in their place)
CHARS_NOTE = "[… {n} characters omitted …]"    # a run of omitted characters (in one long line, or a JSON string)
_ITEMS_MARK = re.compile(r"\[… (\d+) more items? …\]")
# (a note as written, or as a JSON dump with ASCII escapes writes it: "[\u2026 284 more items \u2026]")
_NOTES = re.compile(r"\[(?:…|\\u2026) \d+ (?:more items?|lines? omitted|characters? omitted) (?:…|\\u2026)\]")


def _note(template, n):
    s = template.format(n=n)
    if n == 1:
        s = s.replace(" lines ", " line ").replace(" items ", " item ").replace(" characters ", " character ")
    return s


def _ntok(text):
    return len(encoding().encode_ordinary(text))


_NOTE_TOKENS = {}


def _note_tokens(template, n):
    """Tokens of the note for n omitted lines on a line of its own, its newline included (after LINES_NOTE's closing
    bracket the newline joins the bracket's token, so it adds none). Kept per number of digits: digits are split into
    runs of up to three, one token each, so the count depends only on how many there are."""
    key = (template, len(str(n)), n == 1)
    got = _NOTE_TOKENS.get(key)
    if got is None:
        got = _NOTE_TOKENS[key] = _ntok(_note(template, n) + "\n")
    return got


def _chars_note_tokens(n):
    """Tokens of the note for n omitted characters (CHARS_NOTE), kept per number of digits like _note_tokens (no
    newline: the note stands inside a line)."""
    key = (CHARS_NOTE, len(str(n)), n == 1, False)
    got = _NOTE_TOKENS.get(key)
    if got is None:
        got = _NOTE_TOKENS[key] = _ntok(_note(CHARS_NOTE, n))
    return got


def _strip_notes(text):
    """`text` without the cut's own notes, whose counts are not quotes: as written, and as a JSON result re-dumped
    with ASCII escapes writes them ("[\\u2026 284 more items \\u2026]")."""
    return _NOTES.sub(" ", text) if "[… " in text or "[\\u2026 " in text else text


# ------------------------------------------------------------------------------------------- what later turns quote
_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)")
_WORD = re.compile(r"\w(?:[\w.:/@#-]*\w)?")
_TOKEN_CHAR = re.compile(r"[\w.:/@#-]")
_QUOTED = re.compile(r'"([^"\n]*)"|“([^”\n]*)”|`([^`\n]*)`|(?<!\w)\'([^\'\n]*)\'(?!\w)')
_GUTTER = re.compile(r"\s*(?:L?\d+(?:\s*[:|]\s?|\t| {1,2})|[+\-](?=\s))")   # `cat -n` / grep -n numbers, diff marks
_WORD_SEPS = frozenset("_./@:#-")
_NUMBER_NL = re.compile(r"\n|" + _NUMBER.pattern)        # (findall over a whole text: its numbers and line breaks)
_WORD_NL = re.compile(r"\n|" + _WORD.pattern)
_LINES_NOTE_LINE = re.compile(r"\[… \d+ lines? omitted …\]")


def _number_key(s):
    """A number (a _NUMBER match) as an anchor: thousands commas and trailing fractional zeros dropped, so 1,234.50 and
    1234.5 match, and so do 25.00 and 25; None under 2 digits."""
    k = s.replace(",", "")
    if "." in k:
        k = k.rstrip("0").rstrip(".")
    return k if len(k) - ("." in k) >= 2 else None


def _numbers(text):
    """The number anchors (_number_key) a text writes."""
    return {k for k in map(_number_key, set(_NUMBER.findall(text))) if k}


def _is_word_anchor(tok):
    """An identifier-like token: letters with digits (ORD-1234, a3f9c2e, user42), letters with an inner separator
    (src/app.py, user_name, a@b.io), or digits with two separators (10.0.0.1, 2024-11-18, 12:30:45)."""
    if len(tok) < 4:
        return False
    digit, alpha = any(c.isdigit() for c in tok), any(c.isalpha() for c in tok)
    if digit and alpha:
        return True
    seps = sum(c in _WORD_SEPS for c in tok)
    return len(tok) >= 5 and ((alpha and seps >= 1) or (digit and seps >= 2))


_GLUE = re.compile(r"[:#]+")
PATH_TAILS = 16             # path tails a token gives (src/app.py also gives app.py): of its last 16 parts at most
_VARIANTS_CACHED = 1024     # tokens up to this many characters keep their variants in a cache; longer ones never do
_VARIANTS_CACHE_CHARS = 1 << 22     # the cache is emptied when its variants would pass this many characters
_variants_cache = {}
_variants_cache_chars = [0]


def _glued(tok):
    """True if a word-like token holds other tokens (see _variants)."""
    return "/" in tok or ":" in tok or "#" in tok


def _variants(tok):
    """A word-like token and the tokens glued into it: its runs of parts between ':' or '#' (a grep -n / flake8 /
    compiler location src/app.py:212:5 also gives src/app.py and src/app.py:212, a pytest id tests/t.py::test_x gives
    tests/t.py and test_x, a link's file.py#L12 gives file.py), and the trailing parts of every path among them
    (src/app.py also gives app.py; at most the tails of the last PATH_TAILS parts). Each starts and ends with a word
    character, like a token. Linear in the token's length, so a classpath of 300 jars or a 20,000-part token is cheap.
    Only tokens up to _VARIANTS_CACHED characters are cached, and the cache is emptied when it would hold more than
    _VARIANTS_CACHE_CHARS characters, so it stays small in a long-lived worker."""
    got = _variants_cache.get(tok)
    if got is None:
        got = _token_variants(tok)
        if len(tok) <= _VARIANTS_CACHED:
            size = sum(map(len, got))
            if _variants_cache_chars[0] + size > _VARIANTS_CACHE_CHARS:
                _variants_cache.clear()
                _variants_cache_chars[0] = 0
            _variants_cache[tok] = got
            _variants_cache_chars[0] += size
    return got


def _token_variants(tok):
    out = {tok}
    if ":" in tok or "#" in tok:
        bounds = [0] + [x for m in _GLUE.finditer(tok) for x in m.span()] + [len(tok)]
        pieces = list(zip(bounds[::2], bounds[1::2]))
        runs = ([(a, b) for k, (a, _) in enumerate(pieces) for _, b in pieces[k:]] if len(pieces) <= 6
                else pieces)
        out.update(tok[a:b] for a, b in runs)
    if "/" in tok:
        for t in [t for t in out if "/" in t]:
            bits = t.split("/")
            out.update("/".join(bits[k:]) for k in range(max(1, len(bits) - PATH_TAILS), len(bits)))
    return frozenset(w for w in (t.strip(".:/@#-") for t in out) if w)


def _tokens(text):
    """Every word-like token of a text and the tokens glued into it (_variants: path tails, the parts of
    path:line:col locations and of pytest ids)."""
    out = set(_WORD.findall(text))
    for tok in [t for t in out if _glued(t)]:
        out |= _variants(tok)
    return out


def _words(text):
    """The identifier-like tokens of a text (_is_word_anchor), path tails included: word anchors."""
    return {t for t in _tokens(text) if _is_word_anchor(t)}


@functools.lru_cache(maxsize=1 << 16)
def _inner_token(p):
    """The longest word-like token inside phrase `p` with a non-token character on both sides (within p): a text that
    holds p holds it as a whole token, a quick test before searching the text for p. None if there is none."""
    best = None
    for m in _WORD.finditer(p):
        a, b = m.span()
        if (0 < a and b < len(p) and not _TOKEN_CHAR.match(p[a - 1]) and not _TOKEN_CHAR.match(p[b])
                and (best is None or b - a > len(best))):
            best = m.group()
    return best


_RUN = re.compile(r"\w+")
_RUN_NL = re.compile(r"\n|\w+")
_PHRASE_SCAN = 1 << 26      # phrases x text characters above which phrases are looked up by their inner runs


@functools.lru_cache(maxsize=1 << 16)
def _inner_run(p):
    """The longest run of word characters inside phrase `p` with a non-word character on both sides (within p): a text
    that holds p holds it as a whole run (as _RUN finds it). None if there is none."""
    best = None
    for m in _RUN.finditer(p):
        a, b = m.span()
        if 0 < a and b < len(p) and (best is None or b - a > len(best)):
            best = m.group()
    return best


def _phrase_lines(text, phrases, toks=None, limit=64, lines=None, starts=None):
    """{phrase: the first `limit` lines (indices into text.split("\\n")) holding it} for every phrase of `phrases` that
    `text` holds (a phrase has no line break). A phrase is searched in the whole text, after a quick test of its inner
    token against `toks` (the text's _tokens, if given); with many phrases over a long text, one pass over the text's
    word runs first lists the lines holding each phrase's inner run (_inner_run) and only those lines are searched.
    The same lines either way (every line holding a phrase holds its inner run)."""
    out = {}
    if not phrases:
        return out
    if lines is None:
        lines = text.split("\n")
    by_run = {}
    if len(phrases) * len(text) > _PHRASE_SCAN:
        for p in phrases:
            r = _inner_run(p)
            if r is not None:
                by_run.setdefault(r, []).append(p)
    if by_run:
        run_lines, i = {}, 0
        for t in _RUN_NL.findall(text):
            if t == "\n":
                i += 1
            elif t in by_run:
                got = run_lines.setdefault(t, [])
                if not got or got[-1] != i:
                    got.append(i)
        for r, ps in by_run.items():
            cand = run_lines.get(r, ())
            for p in ps:
                hit = []
                for i in cand:
                    if p in lines[i]:
                        hit.append(i)
                        if len(hit) >= limit:
                            break
                if hit:
                    out[p] = hit
    for p in phrases:
        if by_run and _inner_run(p) is not None:
            continue                                        # (looked up above)
        t = _inner_token(p) if toks is not None else None
        if t is not None and t not in toks:
            continue
        pos = text.find(p)
        if pos < 0:
            continue
        if starts is None:
            starts = list(itertools.accumulate((len(x) + 1 for x in lines), initial=0))
        hit = []
        while pos >= 0 and len(hit) < limit:
            i = bisect.bisect_right(starts, pos) - 1
            hit.append(i)
            pos = text.find(p, starts[i + 1]) if i + 1 < len(lines) else -1
        out[p] = hit
    return out


def _line_keys(line):
    """How a line is matched as a whole: stripped, and stripped of a leading line number or diff mark (so a file shown
    by `cat -n` still matches the lines an edit quotes). Lines under 4 characters or without a letter or digit are
    never anchors."""
    k = line.strip()
    if len(k) < 4 or not any(c.isalnum() for c in k):
        return ()
    m = _GUTTER.match(line)
    if m:
        k2 = line[m.end():].strip()
        if k2 != k and len(k2) >= 4 and any(c.isalnum() for c in k2):
            return k, k2
    return (k,)


class Anchors:
    """What later assistant turns quote from earlier tool results (made by quote_anchors): numbers, identifier-like
    words, phrases (quoted strings, one-line string arguments) and whole lines (of assistant text and of multi-line
    arguments: the file lines an edit quotes). Matching is per line of a tool result: numbers (_number_key: 1,234.50
    = 1234.5) and words as whole tokens (or the tokens glued into one, _variants: src/app.py:212: holds src/app.py
    and app.py), phrases as substrings, lines as the whole (stripped, line-number-free) line."""

    __slots__ = ("numbers", "words", "phrases", "lines")

    def __init__(self, numbers=(), words=(), phrases=(), lines=()):
        self.numbers, self.words = frozenset(numbers), frozenset(words)
        self.phrases, self.lines = frozenset(phrases), frozenset(lines)

    def __bool__(self):
        return bool(self.numbers or self.words or self.phrases or self.lines)

    def __or__(self, other):
        return Anchors(self.numbers | other.numbers, self.words | other.words, self.phrases | other.phrases,
                       self.lines | other.lines)

    def __repr__(self):
        return (f"Anchors({len(self.numbers)} numbers, {len(self.words)} words, {len(self.phrases)} phrases, "
                f"{len(self.lines)} lines)")

    def all(self):
        return set(self.numbers | self.words | self.phrases | self.lines)

    def subset(self, keep):
        keep = frozenset(keep)
        return Anchors(self.numbers & keep, self.words & keep, self.phrases & keep, self.lines & keep)

    def weight(self, a):
        """How surely a line holding anchor `a` is a quote: whole lines first, then long words and phrases, numbers by
        their digits (a 2-digit number is weak evidence)."""
        if a in self.lines:
            return 10 + min(len(a), 60)
        if a in self.phrases or a in self.words:
            return min(len(a), 40)
        if a in self.numbers:
            return min(len(a) - ("." in a), 20)
        return 0

    def restrict(self, text):
        """The anchors that occur in `text`, as an Anchors: one pass over the whole text (its numbers, tokens and line
        keys), plus a substring search for each phrase whose inner token (_inner_token) the text holds (many phrases
        over a long text: only in the lines holding its inner run, _phrase_lines)."""
        if not self:
            return self
        nums = _numbers(text) & self.numbers if self.numbers else ()
        toks = _tokens(text) if self.words or self.phrases else frozenset()
        words = toks & self.words if self.words else ()
        if len(self.phrases) * len(text) <= _PHRASE_SCAN:
            phrases = [p for p in self.phrases if ((t := _inner_token(p)) is None or t in toks) and p in text]
        else:
            phrases = _phrase_lines(text, self.phrases, toks, 1)
        lines = ({k for ln in text.split("\n") for k in _line_keys(ln)} & self.lines) if self.lines else ()
        return Anchors(nums, words, phrases, lines)

    def found(self, text):
        """Every anchor that occurs in `text`."""
        return self.restrict(text).all()

    def in_line(self, line):
        """The anchors in one line of a tool result."""
        out = set()
        if self.numbers:
            out |= _numbers(line) & self.numbers
        if self.words:
            out |= _tokens(line) & self.words
        if self.phrases:
            out.update(p for p in self.phrases if p in line)
        if self.lines:
            out.update(k for k in _line_keys(line) if k in self.lines)
        return out


def quote_anchors(turns):
    """The anchors of assistant turns: what they can quote from an earlier tool result. `turns`: Messages or message
    dicts (assistant only; their text and every call argument count) or plain strings (text).

    From the text and the one-line arguments: numbers with 2+ digits (thousands commas and trailing fractional zeros
    ignored: 25.00 = 25), identifier-like words (ORD-1234, user_42, src/app.py, 10.0.0.1, 2024-11-18, and the names
    glued into a location: src/app.py:212 also gives src/app.py) and phrases (quoted strings in the text, each opening
    quote paired with the next closing one, and one-line string arguments; 4-200 characters with a letter: digit
    strings count as numbers, which match only whole). From the text and the multi-line arguments: their lines (file
    lines: an edit's old text, a quoted log line); a multi-line argument (a file written, an edit's text) gives only
    its lines, since the numbers and names in code would match unrelated output. Anchors.found(text) / in_line(line)
    say which of them a tool result holds. The one definition of a quote: the tool-output cut keeps what holds them
    (cut_tool_outputs), its report lists the ones it removed that the answer still cites (CutReport.anchors_lost), and
    the data producer's re-check of the cut sample reads that report."""
    numbers, words, phrases, lines = set(), set(), set(), set()

    def add_text(t, quoted):
        if not t:
            return
        numbers.update(_numbers(t))
        words.update(_words(t))
        if quoted:
            for m in _QUOTED.finditer(t):
                p = next(g for g in m.groups() if g is not None).strip()
                if 4 <= len(p) <= 200 and any(c.isalpha() for c in p):
                    phrases.add(p)
        for ln in t.split("\n"):
            lines.update(k for k in _line_keys(ln) if len(k) <= 400)

    def add_value(v):
        if isinstance(v, str):
            s = v.strip()
            if "\n" in s:                      # a file's or an edit's text: what it quotes is whole lines, and its
                for ln in s.split("\n"):       # numbers and names are code, which matches unrelated output
                    lines.update(k for k in _line_keys(ln) if len(k) <= 400)
                return
            add_text(v, False)
            if 4 <= len(s) <= 200 and any(c.isalpha() for c in s):
                phrases.add(s)
        elif isinstance(v, bool) or v is None:
            return
        elif isinstance(v, (int, float)):
            k = str(v).lstrip("-")
            if _NUMBER.fullmatch(k):
                k = _number_key(k)
                if k:
                    numbers.add(k)
        elif isinstance(v, dict):
            for x in v.values():
                add_value(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                add_value(x)

    for t in turns:
        if isinstance(t, str):
            add_text(t, True)
            continue
        if isinstance(t, Message):
            if t.role != "assistant":
                continue
            add_text(t.content, True)
            for c in t.tool_calls:
                add_value(c.arguments)
            continue
        if not isinstance(t, dict) or t.get("role", "assistant") != "assistant":
            continue
        add_text(text_of(t.get("content")), True)
        for c in t.get("tool_calls") or []:
            if isinstance(c, ToolCall):
                add_value(c.arguments)
                continue
            fn = c.get("function") if isinstance(c, dict) and isinstance(c.get("function"), dict) else c
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            add_value(args)
    return Anchors(numbers, words, phrases, lines)


# ------------------------------------------------------------------------------------------------------ the report
@dataclass
class ResultCut:
    """How one tool result over its share of its turn's budget was cut (CutReport.cuts). In cut_tool_outputs' report,
    tokens are counted as the rendering shows the result (special-token strings escaped); cut_result alone counts the
    text as given."""
    how: str = ""                   # "lines" (head, quoted and tail lines), "line" (one long line), "json" (structure)
    share: int = 0                  # tokens the result was allowed
    tokens_before: int = 0
    tokens_after: int = 0
    turn: int = -1                  # index of its environment turn in Conversation.messages
    index: int = 0                  # its position among that turn's results
    quoted_kept: int = 0            # quoted anchors (quote_anchors of the later turns) it held and still shows
    quoted_lost: int = 0            # ... that it no longer shows (= len(anchors_lost))
    quoted_lines: int = 0           # lines (JSON: list items; one long line: runs of spans) kept for a quote and
    #                                 showing it, beyond the first and last lines (JSON: the head and tail items)
    items_elided: int = 0           # JSON list items replaced by a "[… N more items …]" marker
    strings_cut: int = 0            # JSON string values cut
    over_budget: bool = False       # JSON left over its share as its skeleton (no longer happens: such JSON is cut as
    #                                 text, json_text_fallback; kept so the report keeps its fields)
    anchors_lost: tuple = ()        # those quoted anchors, sorted
    json_text_fallback: bool = False    # JSON over its share even as its skeleton (every key and scalar), so cut as
    #                                     text (how "lines" or "line"): the one case where keys can go


@dataclass
class CutReport:
    """What the tool-output cut did to one conversation (Conversation.cut_report; training_sample's
    Rendered.cut_report). counts() gives the numbers to add up per source."""
    results: int = 0                                # tool results looked at
    cuts: list = field(default_factory=list)        # a ResultCut for every result that was cut, in order
    anchors_lost: tuple = ()        # anchors an assistant turn after the cut output cites that the cut removed and
    #                                 that nothing the turn sees shows any more (no tool result, JSON read as its
    #                                 decoded strings; no system text, tools, user or earlier assistant turn): words,
    #                                 phrases, lines, numbers of 3+ digits; sorted
    weak_lost: tuple = ()           # ... numbers of 2 digits (too common to tell a citation from a coincidence)

    @property
    def results_cut(self):
        return len(self.cuts)

    @property
    def quoted_kept(self):
        return sum(c.quoted_kept for c in self.cuts)

    @property
    def quoted_lost(self):
        return sum(c.quoted_lost for c in self.cuts)

    @property
    def quoted_lines(self):
        return sum(c.quoted_lines for c in self.cuts)

    @property
    def json_items_elided(self):
        return sum(c.items_elided for c in self.cuts)

    @property
    def tokens_removed(self):
        return sum(c.tokens_before - c.tokens_after for c in self.cuts)

    @property
    def json_text_fallback(self):
        """JSON results cut as text because even their skeleton was over their share."""
        return sum(c.json_text_fallback for c in self.cuts)

    def counts(self):
        return {"results": self.results, "results_cut": self.results_cut,
                "json_cut": sum(c.how == "json" for c in self.cuts),
                "quoted_kept": self.quoted_kept, "quoted_lost": self.quoted_lost, "quoted_lines": self.quoted_lines,
                "json_items_elided": self.json_items_elided,
                "json_over_budget": sum(c.over_budget for c in self.cuts),
                "json_text_fallback": self.json_text_fallback,
                "tokens_removed": self.tokens_removed, "anchors_lost": len(self.anchors_lost),
                "weak_lost": len(self.weak_lost)}


# ------------------------------------------------------------------------------------------------- cutting one text
def _joins_previous(c):
    """A character that continues the one before it: a combining mark, a joiner, a variation selector, an emoji
    modifier or tag. A cut just before it would split what reads as one character."""
    o = ord(c)
    return (unicodedata.category(c) in ("Mn", "Mc", "Me") or o in (0x200C, 0x200D) or 0xFE00 <= o <= 0xFE0F
            or 0x1F3FB <= o <= 0x1F3FF or 0xE0020 <= o <= 0xE007F or 0xE0100 <= o <= 0xE01EF)


def cut_span(text, head_tokens, tail_tokens, ids=None):
    """(head, tail): a prefix of `text` of about head_tokens tokens and a suffix of about tail_tokens tokens, for a text
    cut in its middle. The cuts are made on bytes, at the tokens' boundaries moved to UTF-8 character boundaries (so
    nothing decodes to U+FFFD) and off character sequences (a combining mark, a zero-width joiner, a variation
    selector or an emoji modifier stays with what it modifies, in the piece or out of it). head is a prefix and tail a
    suffix of `text`, and they never overlap. A raw lone surrogate (invalid text) counts as one 3-byte character: it
    goes with the part it falls in. `ids`: the text's encode_ordinary ids, if known."""
    enc = encoding()
    ids = enc.encode_ordinary(text) if ids is None else ids
    data = text.encode("utf-8", "surrogatepass")
    hb = len(enc.decode_bytes(ids[:head_tokens])) if head_tokens > 0 else 0
    tb = len(enc.decode_bytes(ids[max(0, len(ids) - tail_tokens):])) if tail_tokens > 0 else 0
    start = max(len(data) - tb, hb)
    while 0 < hb < len(data) and (data[hb] & 0xC0) == 0x80:          # inside a multi-byte character: back off
        hb -= 1
    while hb < start < len(data) and (data[start] & 0xC0) == 0x80:
        start += 1
    h = len(data[:hb].decode("utf-8", "surrogatepass"))
    t = len(text) - len(data[start:].decode("utf-8", "surrogatepass"))
    while h > 0 and ((h < len(text) and _joins_previous(text[h])) or text[h - 1] in "‌‍"
                     or _splits_flag(text, h)):
        h -= 1
    while t < len(text) and (_joins_previous(text[t]) or (t > 0 and text[t - 1] in "‌‍")
                             or _splits_flag(text, t)):
        t += 1
    return text[:h], text[max(t, h):]


def _splits_flag(text, i):
    """True if position i falls between the two regional indicators of a flag (an odd run of them before it)."""
    def ri(k):
        return 0 <= k < len(text) and 0x1F1E6 <= ord(text[k]) <= 0x1F1FF
    if not (ri(i - 1) and ri(i)):
        return False
    k = i - 1
    while ri(k - 1):
        k -= 1
    return (i - k) % 2 == 1


def _splits_sequence(text, i):
    """True if a cut at position i would split a character sequence (see cut_span)."""
    return 0 < i < len(text) and (_joins_previous(text[i]) or text[i - 1] in "‌‍" or _splits_flag(text, i))


def _word_places(text, words):
    """{word anchor: (start, end) of its first place in `text`}: each word-like token of the text that is one of
    `words`, or that holds one glued into it (_variants: src/app.py in src/app.py:212:), as Anchors.in_line matches
    words; never inside a longer token (user_42 is not in user_429). One pass over the text's tokens, so linear in its
    length however many anchors there are (a few are first looked for as substrings, to skip a text holding none)."""
    if len(words) <= 8:
        words = {w for w in words if w in text}
    out = {}
    if not words:
        return out
    for m in _WORD.finditer(text):
        tok = m.group()
        if tok in words and tok not in out:
            out[tok] = m.span()
        if _glued(tok):
            for w in _variants(tok) & words:        # (each a substring of the token: its first place in it)
                if w not in out:
                    p = text.find(w, m.start(), m.end())
                    out[w] = (p, p + len(w))
        if len(out) == len(words):
            break
    return out


def _anchor_spans(text, anchors, json_text=False):
    """[(start, end, weight, anchor)]: where each anchor first stands in a text, strongest first (ties by place, then
    by the anchor, never by set order: the cut must not depend on PYTHONHASHSEED), found as Anchors.in_line finds
    them: numbers and words as whole tokens (a word also as a token glued into one), phrases as substrings. A
    whole-line anchor has no place (a line cut inline never shows all of itself), except in JSON cut as text
    (json_text), where it can be a whole string literal and is placed like a phrase (one that is a scalar, - 25.50, at
    that scalar); there anchors are looked for first in what the text's escapes decode to (a key written Malm\\u00f6
    holds Malmö), each place mapped back, and a number no other place shows is placed in a scalar (412 in the compact
    list [412,512], which _NUMBER reads as one number)."""
    spans = {}
    views = [(text, None)]
    if json_text and "\\" in text:
        views.insert(0, _json_decoded(text))
    phrases = anchors.phrases | anchors.lines if json_text else anchors.phrases
    scalars = _json_scalars(text, False) if json_text and (anchors.lines or anchors.numbers) else ()
    if anchors.lines:
        for s, p in scalars:
            if s in anchors.lines and s not in spans:
                spans[s] = (p, p + len(s))
    for t, back in views:
        if anchors.numbers:
            for m in _NUMBER.finditer(t):
                k = _number_key(m.group())
                if k in anchors.numbers and k not in spans:
                    spans[k] = back(*m.span()) if back else m.span()
        if anchors.words:
            for a, (s, e) in _word_places(t, anchors.words - spans.keys()).items():
                spans[a] = back(s, e) if back else (s, e)
        for a in phrases:
            if a not in spans:
                p = t.find(a)
                if p >= 0:
                    spans[a] = back(p, p + len(a)) if back else (p, p + len(a))
    if anchors.numbers:
        for s, p in scalars:
            for m in _NUMBER.finditer(s):
                k = _number_key(m.group())
                if k in anchors.numbers and k not in spans:
                    spans[k] = (p + m.start(), p + m.end())
    return _Spans(sorted(((s, e, anchors.weight(a), a) for a, (s, e) in spans.items()),
                         key=lambda x: (-x[2], x[0], x[1], x[3])))


class _Spans(list):
    """_anchor_spans' list, with the windows found around its spans kept for every later cut of the same text
    (.geo: (context, start, end) -> (window start, window end, its tokens)); take() keeps some spans, sharing it."""
    __slots__ = ("geo", "index")

    def __init__(self, items=(), geo=None, index=None):
        super().__init__(items)
        self.geo = {} if geo is None else geo
        self.index = index                          # (anchor -> its span in this list, made on a small take)

    def take(self, keep):
        """The spans whose anchor is in `keep`, in the same order."""
        if len(keep) * 8 >= len(self):
            return _Spans((x for x in self if x[3] in keep), self.geo)
        if self.index is None:
            self.index = {x[3]: x for x in self}
        got = sorted((self.index[a] for a in keep if a in self.index), key=lambda x: (-x[2], x[0], x[1], x[3]))
        return _Spans(got, self.geo)


def _inline_ctx(text, n_ids, max_tokens):
    """Characters of context on each side of a quote window: about 3 tokens' worth under a budget of 128, else 6."""
    return max(1, round((3 if max_tokens < 128 else 6) * len(text) / n_ids))


def _inline_room(max_tokens, nn):
    """Tokens _inline's quote windows may take (with a note before each run of them): all of the budget under 128
    tokens (a small budget shows only its quotes if it must), else all but an eighth (kept for the head and the tail),
    after the note that follows the last run and 2 tokens to spare."""
    return max_tokens - nn - 2 - (0 if max_tokens < 128 else max(8, max_tokens // 8))


def _inline_gap(text, n_ids, nn):
    """The longest gap (in characters: twice a note's worth) that _windows checks for costing fewer tokens than the
    note that would stand in its place."""
    return int(2 * nn * len(text) / n_ids) + 1


def _windows(text, spans, nn, ctx, room, gap=0):
    """_inline's quote windows: `ctx` characters on each side of each span (moved off character sequences), strongest
    first, each kept while the windows' tokens and a note before each run of them stay within `room` (a window inside
    or next to kept ones costs only what it adds) -> (windows [(start, end)], the most they took at any point). Notes
    are priced as rendered: a run at the text's start or end joins the head or the tail, so it needs no note on that
    side (the note after the last run is in _inline_room), and a window reaches a kept run or the text's start or end
    when the gap between them (`gap` characters at most) costs fewer tokens than the note in its place."""
    n = len(text)
    geo = spans.geo if isinstance(spans, _Spans) else {}
    runs, wins, used, peak = _Runs(), [], 0.0, 0.0
    runs.add(0, 0)                                    # (the text's two ends, as empty runs a window can join)
    runs.add(n, n)

    def short(a, b):
        """True if the gap [a, b) costs fewer tokens than a note (kept with the windows' places, per text)."""
        if not 0 < b - a <= gap:
            return False
        got = geo.get(("gap", a, b))
        if got is None:
            got = geo["gap", a, b] = _ntok(text[a:b]) < nn
        return got
    for x in spans:
        got = geo.get((ctx, x[0], x[1]))
        if got is None:
            a, b = max(0, x[0] - ctx), min(n, x[1] + ctx)
            while _splits_sequence(text, a):
                a -= 1
            while _splits_sequence(text, b):
                b += 1
            got = geo[ctx, x[0], x[1]] = (a, b, _ntok(text[a:b]))
        a, b, tok = got
        k = bisect.bisect_right(runs.ends, a) - 1     # (the run ending before it, and the one starting after it)
        a2 = runs.ends[k] if k >= 0 and short(runs.ends[k], a) else a
        k = bisect.bisect_left(runs.starts, b)
        b2 = runs.starts[k] if k < len(runs.starts) and short(b, runs.starts[k]) else b
        chars, notes = runs.added(a2, b2)
        cost = nn * notes + (tok * chars / (b - a) if chars > 0 else 0)
        if used + cost <= room:
            wins.append((a2, b2))
            runs.add(a2, b2)
            used += cost
            peak = max(peak, used)
    return wins, peak


def _inline_need(text, n_ids, spans):
    """The least budget at which _inline keeps a window for every one of `spans` (as _windows counts them)."""
    nn = _chars_note_tokens(len(text))
    gap = _inline_gap(text, n_ids, nn)
    m = math.ceil(_windows(text, spans, nn, _inline_ctx(text, n_ids, 0), math.inf, gap)[1]) + nn + 2
    if m < 128:
        return m
    w = _windows(text, spans, nn, _inline_ctx(text, n_ids, 128), math.inf, gap)[1]
    m = max(128, int((w + nn + 2) * 8 / 7))
    while _inline_room(m, nn) < w:
        m += 1
    return m


def _inline(text, max_tokens, ids=None, anchors=None, json_text=False, spans=None, context=None):
    """A text (one long line, or one with no lines worth keeping) cut to at most max_tokens tokens -> (text, runs of
    quote windows kept between the head and the tail): the places it shows `anchors` (spans: _anchor_spans of them,
    if known) with a few tokens around each, strongest first, while they fit (_inline_room: all of a budget under
    128 tokens, else all but an eighth), then its head and its tail from what is left, each omitted run replaced by
    CHARS_NOTE. Cut on UTF-8 character boundaries and off character sequences (cut_span): the head is a prefix and the
    tail a suffix of `text`. json_text: JSON cut as text (anchors also found in what its escapes decode to).
    `context`: the head and the tail only fill the cut up to this many tokens (a JSON string cut around its quotes:
    max_tokens is what its quotes may take); the text stays whole only if the cut would be no shorter. A budget too
    small for anything else gives one note."""
    ids = encoding().encode_ordinary(text) if ids is None else ids
    if len(ids) <= (max_tokens if context is None else min(max_tokens, context)):
        return text, 0
    n = len(text)
    nn = _chars_note_tokens(n)                        # one note's tokens (no count is longer)
    if spans is None:
        spans = _anchor_spans(text, anchors, json_text) if anchors else ()
    wins = (_windows(text, spans, nn, _inline_ctx(text, len(ids), max_tokens), _inline_room(max_tokens, nn),
                     _inline_gap(text, len(ids), nn))[0] if spans else [])     # quote windows, strongest first

    def build(head_tok, tail_tok):
        head, tail = cut_span(text, head_tok, tail_tok, ids)
        h, t = len(head), n - len(tail)
        spans_ = sorted([(0, h)] + [w for w in wins if not (w[1] <= h or w[0] >= t)] + [(t, n)])
        out, pos, shown = [], 0, 0
        for a, b in spans_:
            if b <= pos:
                continue
            if a > pos:
                out.append(_note(CHARS_NOTE, a - pos))
                shown += h <= a < b <= t                  # a run of quote windows between head and tail (an empty
                #                                           tail, t == n, is none)
            out.append(text[max(a, pos):b])
            pos = b
        if pos < n:
            out.append(_note(CHARS_NOTE, n - pos))
        return "".join(out), shown

    room = None
    for _ in range(64):
        if room is None:                              # head and tail: what the windows (with their notes) leave
            w = _ntok(build(0, 0)[0]) if wins else nn
            room = max(0, (max_tokens if context is None else min(max_tokens, max(context, w))) - w)
        out, shown = build(room - room // 2, room // 2)
        over = _ntok(out) - max_tokens
        if over <= 0:
            return (text, 0) if context is not None and len(ids) <= max_tokens + over else (out, shown)
        if room > 0:
            room = max(0, room - over - 1)
        elif wins:
            wins.pop()                                # the weakest window goes
            room = None
        else:
            break
    return _note(CHARS_NOTE, n), 0


class _Runs:
    """Disjoint character runs [start, end) of a text, sorted; a run added over or next to others merges with them."""

    def __init__(self):
        self.starts, self.ends, self.chars = [], [], 0

    def __len__(self):
        return len(self.starts)

    def _touching(self, a, b):
        """Indices k0..k1-1 of the runs that overlap or touch [a, b)."""
        return bisect.bisect_left(self.ends, a), bisect.bisect_right(self.starts, b)

    def added(self, a, b):
        """(characters [a, b) adds, runs it adds: 1 alone, 0 joining one run, -k joining k + 1)."""
        k0, k1 = self._touching(a, b)
        covered = sum(max(0, min(self.ends[k], b) - max(self.starts[k], a)) for k in range(k0, k1))
        return (b - a) - covered, 1 - (k1 - k0)

    def add(self, a, b):
        k0, k1 = self._touching(a, b)
        if k1 > k0:
            a, b = min(a, self.starts[k0]), max(b, self.ends[k1 - 1])
            self.chars -= sum(self.ends[k] - self.starts[k] for k in range(k0, k1))
        self.starts[k0:k1], self.ends[k0:k1] = [a], [b]
        self.chars += b - a


def _line_costs(text, lines, ids=None):
    """Tokens of each line, its newline included, from one encoding of the whole text (a token counts for the line it
    starts in). `ids`: the text's encode_ordinary ids, if known."""
    import numpy as np
    ids = encoding().encode_ordinary(text) if ids is None else ids
    if not ids:
        return [0] * len(lines)
    tb = _token_bytes()[np.asarray(ids, dtype=np.int64)]
    starts = np.cumsum(tb) - tb
    # (surrogatepass: a raw lone surrogate, from JSON that escaped half an emoji, is 3 bytes, as the U+FFFD tiktoken
    # encodes in its place)
    lb = np.fromiter((len(x.encode("utf-8", "surrogatepass")) + 1 for x in lines), dtype=np.int64, count=len(lines))
    owner = np.searchsorted(np.cumsum(lb) - lb, starts, side="right") - 1
    return np.bincount(owner, minlength=len(lines)).tolist()


def _line_hits(text, lines, anchors):
    """{line index: the anchors it holds} for a multi-line text, the same matches as Anchors.in_line line by line at
    a fraction of the cost: one findall pass over the whole text for numbers and one for words (line breaks counted),
    a search per word when the text holds few of them (each place confirmed against its line's tokens), the first 64
    lines of each phrase the text holds (_phrase_lines), whole-line anchors line by line."""
    hits = {}
    starts = list(itertools.accumulate((len(x) + 1 for x in lines), initial=0))

    def lines_of(needle, holds):
        out, pos = [], text.find(needle)
        while pos >= 0 and len(out) < 64:
            i = bisect.bisect_right(starts, pos) - 1
            if holds(i):
                out.append(i)
            pos = text.find(needle, starts[i + 1]) if i + 1 < len(lines) else -1
        return out

    if anchors.numbers:
        nums, i = anchors.numbers, 0
        for t in _NUMBER_NL.findall(text):
            if t == "\n":
                i += 1
            elif len(t) > 1:
                k = t if "," not in t and "." not in t else _number_key(t)
                if k in nums:
                    hits.setdefault(i, set()).add(k)
    toks = set()
    if anchors.words or anchors.phrases:
        found = _WORD_NL.findall(text)
        toks = set(found)
        toks.discard("\n")
        glued = {t: _variants(t) for t in toks if _glued(t)}     # path tails, the parts of path:line:col (_variants)
        for v in glued.values():
            toks |= v
        words = toks & anchors.words
        if 0 < len(words) <= 32:
            line_toks = {}                                  # (a line's tokens, for every word it holds)

            def holds(w, i):
                got = line_toks.get(i)
                if got is None:
                    got = line_toks[i] = _tokens(lines[i])
                return w in got
            for w in sorted(words):
                for i in lines_of(w, lambda i: holds(w, i)):
                    hits.setdefault(i, set()).add(w)
        elif words:
            i = 0
            for t in found:
                if t == "\n":
                    i += 1
                    continue
                if t in words:
                    hits.setdefault(i, set()).add(t)
                if t in glued:
                    got = glued[t] & words
                    if got:
                        hits.setdefault(i, set()).update(got)
    for p, at in _phrase_lines(text, anchors.phrases, toks, 64, lines, starts).items():
        for i in at:
            hits.setdefault(i, set()).add(p)
    if anchors.lines:
        whole = anchors.lines
        for i, ln in enumerate(lines):
            k = ln.strip()
            if len(k) < 4:
                continue
            if k in whole:
                hits.setdefault(i, set()).add(k)
            m = _GUTTER.match(ln)
            if m:
                k2 = ln[m.end():].strip()
                if k2 != k and k2 in whole:
                    hits.setdefault(i, set()).add(k2)
    for i in [i for i in hits if _LINES_NOTE_LINE.fullmatch(lines[i])]:      # an earlier cut's note is no quote
        del hits[i]
    return hits


class _TextCut:
    """A multi-line text prepared for _cut_lines at any budget: its lines and their token costs, the anchors each line
    holds (_line_hits; JSON cut as text, json_text: also in its decoded strings, _json_line_hits), the lines showing
    each anchor, the anchors strongest first, each line's anchors' total weight, and .anchors: those the text holds
    (anchors restricted to it). Kept across budgets: each line's ids (line_ids), the places it shows its anchors
    (spans), its cuts (render) and the least it takes to show some of them (need). .dense: at least 16 lines, and
    three in four, hold a quote (a file read and then written back whole); .by_place: the anchors in the order of their
    first line."""

    def __init__(self, text, anchors=None, ids=None, json_text=False):
        self.text, self.ids, self.json_text = text, ids, json_text
        self.lines = text.split("\n")
        self.costs = _line_costs(text, self.lines, ids)
        self.quoted, self.where, self.order, self.weights, self.totals, self.anchors = {}, {}, [], {}, {}, None
        self._line_ids, self._held, self._spans, self._need, self._by_place = {}, {}, {}, {}, None
        self._cuts = {}
        self.dense = False
        if anchors:
            self.quoted = (_json_line_hits(self.lines, anchors) if json_text and ('"' in text or "\\" in text)
                           else _line_hits(text, self.lines, anchors))
            for i in sorted(self.quoted):
                for a in self.quoted[i]:
                    self.where.setdefault(a, []).append(i)
            self.anchors = anchors.subset(self.where) if self.where else None
            self.weights = {a: anchors.weight(a) for a in self.where}
            self.totals = {i: sum(self.weights[a] for a in f) for i, f in self.quoted.items()}
            # (ties broken by the anchor itself, never by set order: the cut must not depend on PYTHONHASHSEED)
            self.order = sorted(self.where, key=lambda a: (-self.weights[a], self.where[a][0], a))
            self.dense = len(self.quoted) >= 16 and 4 * len(self.quoted) >= 3 * len(self.lines)

    @property
    def by_place(self):
        if self._by_place is None:
            self._by_place = sorted(self.where, key=lambda a: (self.where[a][0], -self.weights[a], a))
        return self._by_place

    def line_ids(self, i):
        got = self._line_ids.get(i)
        if got is None:
            got = self._line_ids[i] = encoding().encode_ordinary(self.lines[i])
        return got

    def held(self, i):
        """The anchors line i holds, as an Anchors."""
        got = self._held.get(i)
        if got is None:
            got = self._held[i] = self.anchors.subset(self.quoted[i])
        return got

    def spans(self, i, want=None):
        """The places line i shows those of its anchors in `want` (all if None): _anchor_spans, strongest first."""
        if want is not None and not want:
            return _Spans()
        got = self._spans.get(i)
        if got is None:
            got = self._spans[i] = (_anchor_spans(self.lines[i], self.held(i), self.json_text) if i in self.quoted
                                    else _Spans())
        return got if want is None else got.take(want)

    def need(self, i, want, room=None):
        """The price of showing the anchors `want` (some line i holds) on line i, its newline included: the whole line
        if one of them has no place (a whole-line quote); else the tokens of the cheapest cut of the line that shows
        them, as _inline renders it, found by rendering (exact(); budget() is the budget that renders it, which can be
        more than the cut's tokens), or the whole line if no rendered cut is shorter (whole()). A line of SHORT_LINE
        tokens or less is rendered at every budget from its length down to the least a cut showing them can take (a
        note less 2 tokens, for joins, and the tokens of their longest place). A longer line is rendered around its
        head or tail price (_ends_need: the quotes lie at its start or end) when that is below the places' estimate:
        up to 3 budgets down while its cut shows them, and 1 and 2 budgets up. Where none of those shows them, or the
        head and tail cannot (a quote in the line's middle), the price is the places' estimate (_inline_need, 2 tokens
        to spare): a budget, at which the cut can take fewer tokens. `room`: what the caller can give it (None: no
        limit). A short line is rendered only if its least cut fits the room and its head or tail price, or else its
        places' estimate, is at most PRICE_SPARE tokens over it; a longer one only if its head or tail price is. Else
        the whole line or the estimate is returned, and a later call with more room renders it."""
        key = (i, frozenset(want))
        got = self._need.get(key)
        if got is None:
            sp = self.spans(i, want)
            est = least = self.costs[i]
            todo, kind = None, "whole"
            if len(sp) == len(key[1]):
                kind = "estimate"
                if est > SHORT_LINE:
                    est = min(est, _inline_need(self.lines[i], len(self.line_ids(i)), sp) + 1)
                    least = min(est, self._ends_need(i, sp, est))
                    todo = "ends" if least < est else None
                else:
                    todo = "short"
                    least = (_chars_note_tokens(len(self.lines[i])) - 2
                             + max(_ntok(self.lines[i][s:e]) for s, e, *_ in sp))
            # (price, the estimate the room is checked against: a long line's head or tail price, a short line's least
            # cut, then its head or tail price, then its places'; rendering to do, kind of price, the budget of the cut
            # priced, a short line's estimates so far (0-2), its least cut)
            got = self._need[key] = [est, least, todo, kind, None, 0, least]
        todo = got[2]
        if todo is None:
            return got[0]
        if room is not None:
            if todo == "short" and got[5] == 0:
                if got[6] > room:                   # (no cut that shows them fits)
                    return got[0]
                got[1], got[5] = min(got[0], self._ends_need(i, self.spans(i, key[1]), got[0])), 1
            if todo == "short" and got[5] == 1 and got[1] > room + PRICE_SPARE:
                got[1], got[5] = min(got[1], _inline_need(self.lines[i], len(self.line_ids(i)),
                                                          self.spans(i, key[1])) + 1), 2
            if got[1] > room + PRICE_SPARE:
                return got[0]
        best = self._rendered_need(i, key[1], got[6] if todo == "short" else got[1], todo == "short")
        if best is not None and best[0] < self.costs[i]:
            got[0], got[3], got[4] = best[0], "cut", best[1]
        elif todo == "short":
            got[3] = "whole"
        got[2] = None
        return got[0]

    def exact(self, i, want):
        """True if need(i, want) is the tokens of a rendered cut."""
        got = self._need.get((i, frozenset(want)))
        return got is not None and got[3] == "cut"

    def whole(self, i, want):
        """True if need(i, want) is the whole line and no cut is cheaper: a whole-line quote among `want`, or a short
        line whose cuts that show them all cost as much."""
        got = self._need.get((i, frozenset(want)))
        return got is not None and got[3] == "whole"

    def budget(self, i, want):
        """The budget at which line i's cut shows `want` in need(i, want) tokens (exact prices only)."""
        return self._need[(i, frozenset(want))][4]

    def render(self, i, want, m):
        """Line i cut to budget m (its newline included: _inline at m - 1), the places of the anchors `want` first ->
        (text, tokens, anchors it shows); kept for later tries. The tokens can be fewer than m (a CRLF row's newline
        joins its '\\r'), and a larger budget can give a shorter cut."""
        key = (i, frozenset(want), m)
        got = self._cuts.get(key)
        if got is None:
            t = _inline(self.lines[i], m - 1, self.line_ids(i), spans=self.spans(i, want), json_text=self.json_text)[0]
            got = self._cuts[key] = (t, _ntok(t + "\n" if i < len(self.lines) - 1 else t), _shows(self, i, t))
        return got

    def _ends_need(self, i, sp, need):
        """The budget at which _inline's head or tail would show every place `sp` of line i without a window (a quote
        at the line's start or end): the head and the tail share what a note leaves, about half each (the head gets
        the odd token), so it is a note and twice the tokens up to the last place in the head, or from the first place
        in the tail. `need` if a place is in neither at `need` (a quote in the middle of the line)."""
        line, ids = self.lines[i], self.line_ids(i)
        nn = _chars_note_tokens(len(line))
        r = need - 1 - nn                                  # (head and tail at `need`, the newline aside)
        if r < 2:
            return need
        head, tail = cut_span(line, r - r // 2, r // 2, ids)
        h, t = len(head), len(line) - len(tail)
        if not all(e <= h or s >= t for s, e, *_ in sp):
            return need
        e = max((e for _, e, *_ in sp if e <= h), default=0)
        s = min((s for s, e, *_ in sp if e > h), default=len(line))
        kh, kt = _ntok(line[:e]) if e else 0, _ntok(line[s:]) if s < len(line) else 0
        return nn + 1 + max(2 * kh - 1, 2 * kt)

    def _rendered_need(self, i, want, least, short):
        """(tokens, budget) of the cheapest cut of line i that shows every one of `want`, among those rendered, or None
        if none of them shows them. `short`: the cut at every budget from the line's own tokens down to `least` (no
        cut showing them takes fewer tokens: a note less 2, for joins, and the tokens of their longest place). Else
        around `least` (_ends_need: the head or the tail showing them): down from it while the cut shows them (at
        most 3 budgets), and 1 and 2 budgets over it (token joins move the cut by a token or two, and a larger budget
        can give a shorter cut)."""
        best = None

        def shows(m):
            nonlocal best
            _, k, s = self.render(i, want, m)
            if not want <= s:
                return False
            if best is None or k < best[0] or (k == best[0] and m < best[1]):
                best = (k, m)
            return True
        if short:
            for m in range(len(self.line_ids(i)), max(1, least - 1), -1):
                shows(m)
            return best
        if shows(least):
            m = least
            for _ in range(3):
                if m - 1 < 2 or not shows(m - 1):
                    break
                m -= 1
        for m in (least + 1, least + 2):
            shows(m)
        return best


def _shows(prep, i, t):
    """The anchors line i of a prepared text (_TextCut) shows as `t`, a cut of it."""
    if i not in prep.quoted:
        return set()
    t = _strip_notes(t)
    return prep.held(i).found(_json_view(t)) if prep.json_text else prep.held(i).in_line(t)


def _select_lines(prep, budget, note, shown=(), context=None, by_place=False):
    """One try of _cut_lines at `budget`: (text, lines kept for a quote, the weight of the quotes it shows, True if a
    quote is left out), or None if not even one line fits. `shown`: anchors shown elsewhere already (no line is kept
    for them). `context`: the text is cut around its quotes (a JSON string in an item kept for its quotes): no first
    and last lines while quotes wait, the quotes and their lines within `budget`, then gaps, head and tail only up to
    `context` tokens in all. by_place: quotes are placed in the order of their first line, not strongest first."""
    orig, quoted, where, full = prep.lines, prep.quoted, prep.where, prep.costs
    around = context is not None and any(a not in shown for a in prep.order)
    if context is not None and not around:  # (no quote left to show: a plain cut to `context`)
        budget = min(budget, context)
    n = len(orig)
    lines, c, part = list(orig), list(full), {}
    cap = max(8, budget // 4)              # a head or tail line longer than this shows its ends and its quotes
    half = max(cap, budget // 2)           # a quoted line is kept whole up to this
    ends = min(cap, max(16, budget // 16)) if where else cap    # what a long first or last line starts with
    nc = _note_tokens(note, n)             # the longest omitted run's note and its newline
    kept, keep, used = [], set(), [nc]     # used: kept lines + a note per omitted run (now: all lines, one run)
    seen, credit = set(), {}               # anchors shown (elsewhere, or by a kept line); kept line -> anchors it shows
    weight, unseen = prep.weights, dict(prep.totals)    # each line's weight of anchors not shown yet

    def see(s):
        for x in s:
            if x not in seen:
                seen.add(x)
                for j in where.get(x, ()):
                    unseen[j] -= weight[x]

    see(shown)

    def notes(i):
        """The change in the notes' tokens if line i is added: its run of omitted lines becomes the runs before and
        after it, a note each (if not empty)."""
        k = bisect.bisect_left(kept, i)
        lo = kept[k - 1] + 1 if k else 0
        hi = kept[k] - 1 if k < len(kept) else n - 1
        return ((_note_tokens(note, i - lo) if i > lo else 0) + (_note_tokens(note, hi - i) if i < hi else 0)
                - _note_tokens(note, hi - lo + 1))

    def unshown(i):
        return {a for a in quoted.get(i, ()) if a not in seen}

    def form(i, to, want=(), cut=False):
        """Line i cut to `to` tokens, its newline included (whole if it fits), showing the places of the anchors
        `want` first -> (text, tokens, anchors it shows), or None if it no longer shows those it is kept for. `cut`:
        the cut rendered at that budget even if the line fits, when it is shorter than the line (a CRLF row: the
        line's newline joins its '\\r', while the cut keeps a token for it)."""
        got = None
        if to < full[i] or (cut and to - 1 < len(prep.line_ids(i))):
            got = prep.render(i, want, to)                  # (a newline can join its last token)
            if to >= full[i] and got[1] >= full[i]:
                got = None
        if got is None:
            got = orig[i], full[i], set(quoted.get(i, ()))
        return got if credit.get(i, set()) <= got[2] else None

    def put(i, f):
        """Line i kept as `f` (a form), if that fits the budget."""
        if f is None:
            return False
        t, k, s = f
        d = k - c[i] if i in keep else k + notes(i)
        if used[0] + d > budget:
            return False
        if i not in keep:
            bisect.insort(kept, i)
            keep.add(i)
        lines[i], c[i] = t, k
        if t is orig[i]:
            part.pop(i, None)
        else:
            part[i] = t
        used[0] += d
        credit[i] = s
        see(s)
        return True

    def grow(i, to):
        """A kept line cut shorter grows to `to` tokens (if that gives more), still showing what it showed."""
        f = form(i, to, credit[i] | unshown(i)) if to > c[i] else None
        return f is not None and f[1] > c[i] and put(i, f)

    def add_context(i):
        """Line i for context (head, tail): whole up to a quarter of the budget, else cut to that (its ends and the
        places of the quotes no kept line shows yet); False if it does not fit."""
        return i in keep or put(i, form(i, full[i] if full[i] <= cap else cap, unshown(i)))

    # 1. the first and last lines: one longer than a quarter of the budget shows its ends (and its quotes), and while
    #    there are quotes to place one longer than a sixteenth starts with that much (it grows back in step 5). Cut
    #    around its quotes (context), the text keeps them only as head and tail (step 6) while quotes wait
    for i in () if around else sorted({0, n - 1}):
        put(i, form(i, ends, unshown(i)))
    # 2. every quote at its least, before any other line: for each anchor no kept line shows yet (strongest first),
    #    the lines holding it are tried in turn, the one showing the most weight not shown yet first (then one kept
    #    already, then one next to a kept line, then the first), until one takes it: the line is kept as the places of
    #    that anchor, or the first time of all its quotes of that weight (cut to the room left if they do not all fit:
    #    equal quotes, so as many as fit, in their order), with a few tokens around each (whole if that is no shorter,
    #    or if the places of all its quotes would cost as much: a line of quoted words; a kept line cut shorter adds
    #    the places to what it shows). A whole-line quote needs the whole line, up to half the budget. So a quote
    #    whose place does not fit on a long line holding many quotes is shown by a short line holding it, if that fits.
    #    A quote's price is what its line's cut takes (prep.need: rendered, or an estimate in a long line's middle);
    #    a rendered price is kept if it fits, an estimate is rendered up to PRICE_SPARE tokens over the room.
    #    (The room only shrinks here: a line that missed 4 quotes in a row is passed over, so a line holding thousands
    #    of quotes costs no more than one holding a few)
    picks, tried, missed = [], set(), {}
    nts = [_note_tokens(note, k) for k in [1, 2] + [10 ** d for d in range(1, len(str(n)))]]
    apart = 2 * min(nts) - max(nts)        # the least a line with omitted lines on both sides adds to the notes

    def place(i, a):
        """Line i kept (or grown) to show anchor a at its least, if that fits -> True."""
        new = i not in keep
        if full[i] > half and not prep.spans(i, {a}):
            return False                                    # a whole-line quote in a line over half the budget
        if (new and 0 < i < n - 1 and i - 1 not in keep and i + 1 not in keep
                and budget - used[0] - apart < min(full[i], nc)):
            return False                    # (the check below, without counting the notes: it adds at least `apart`)
        now = 0 if new else c[i]
        room = budget - used[0] + (c[i] if not new else -notes(i))     # what line i can take
        if room - now < (min(full[i] - now, nc) if new else 2):
            return False                    # (not even one more place fits: about a note on a new line; a kept line
            #                                 can show a quote next to its head or tail for a few tokens)
        has, tries = credit.get(i, set()), []
        if full[i] <= half and full[i] - now <= room:      # the whole line, where the places of all its quotes would
            rest = has | unshown(i)                         # cost as much (a line of quoted words)
            if len(rest) > len(has) + 1 and prep.need(i, rest, room) >= full[i]:
                tries.append((rest, {a}))
        if (i, weight[a]) not in tried:
            tried.add((i, weight[a]))
            same = {x for x in quoted[i] if x not in seen and weight[x] == weight[a]}
            if full[i] > half:
                same = {x[3] for x in prep.spans(i, same)}
            if len(same) > 1:
                tries.append((has | same, same))
        tries.append((has | {a}, {a}))
        for want, show in tries:
            need = prep.need(i, want, room)
            exact = prep.exact(i, want)
            if exact or prep.whole(i, want) or full[i] <= SHORT_LINE:  # the tokens of a rendered cut or of the whole
                if need > room and len(show) == 1:      # line (a short line not rendered has no cut that fits): kept if
                    continue                            # it fits; equal quotes that do not all fit are cut to the room
                tos = [room if need > room else prep.budget(i, want) if exact else need]    # (as many as fit)
            else:                                       # an estimate (a budget: the cut at it can take fewer tokens),
                if need > room + PRICE_SPARE and len(show) == 1:    # rendered at the room, at its price and up to
                    continue                                        # PRICE_SPARE over the room
                tos = [min(max(need, now), room)]
                if need > room:
                    tos += range(max(need, room + 1), room + PRICE_SPARE + 1)
            cut = need < full[i]
            for to in tos:
                f = form(i, max(to, now), want, cut)
                if f is not None and f[2] & show and put(i, f):
                    if new:
                        picks.append(i)
                    missed[i] = 0
                    return True
        missed[i] = missed.get(i, 0) + 1
        return False

    for a in (prep.by_place if by_place else prep.order) if where else ():
        if a in seen:
            continue
        cands = sorted((-unseen[i], -(3 if i in keep else (i - 1 in keep) + (i + 1 in keep)), i)
                       for i in where[a][:32] if (i not in keep or i in part) and missed.get(i, 0) < 4)
        for *_, i in cands:                                 # (a line kept whole shows all it holds)
            if place(i, a) and a in seen:
                break
    # 3. the lines kept for quotes grow back, strongest quote first: whole up to half the budget while they fit, then
    #    the first one that did not fit grows into what is left (up to half the budget)
    for i in picks:
        if i in part and full[i] <= half:
            grow(i, full[i])
    for i in picks:
        if i in part:
            grow(i, min(full[i], half, c[i] + budget - used[0]))
    if context is not None:                 # (cut around its quotes: the rest fills it up to `context` tokens at most)
        budget = min(budget, max(context, used[0]))
        cap = max(8, budget // 4)
    # 4. gaps of one or two lines between kept lines are filled (about the cost of the note they would need)
    for a, b in list(zip(kept, kept[1:])):
        if 1 < b - a <= 3:
            for i in range(a + 1, b):
                if full[i] <= cap:
                    put(i, form(i, full[i]))
    # 5. a long first or last line grows back to a quarter of the budget, or into what is left
    if ends < cap:
        for i in sorted({0, n - 1}):
            if i in part:
                grow(i, min(full[i], cap, c[i] + budget - used[0]))
    # 6. head and tail from what is left: up to an eighth of the budget each, then more of both, alternately, while
    #    they fit; a line longer than a quarter of the budget shows its ends and its quotes, never the whole line
    lo, hi, quota = (0, n - 1, budget // 8) if around else (1, n - 2, budget // 8)
    got = 0 if around else c[0]
    while lo <= hi and got < quota and add_context(lo):
        got += c[lo]
        lo += 1
    got = 0 if around else c[n - 1]
    while lo <= hi and got < quota and add_context(hi):
        got += c[hi]
        hi -= 1
    head_ok = tail_ok = True
    side = 0
    while head_ok or tail_ok:
        while lo <= hi and lo in keep:
            lo += 1
        while lo <= hi and hi in keep:
            hi -= 1
        if lo > hi:
            break
        if (side == 0 and head_ok) or not tail_ok:
            if add_context(lo):
                lo += 1
            else:
                head_ok = False
        elif add_context(hi):
            hi -= 1
        else:
            tail_ok = False
        side ^= 1
    if not kept:
        return None
    # 7. budget left (a text of a few huge lines, or a huge line that stopped the head and the tail): the kept lines
    #    cut shorter grow into it, then the first line not kept after the head is added, cut to fit
    for _ in range(3):
        left = budget - used[0]
        short = [i for i in kept if i in part and c[i] < full[i]]
        if not short or left // len(short) < 8:
            break
        extra = left // len(short)
        for i in short:
            grow(i, c[i] + extra)
    while lo <= hi and lo in keep:
        lo += 1
    if lo <= hi and budget - used[0] >= 24:
        f = form(lo, budget - used[0] - max(0, notes(lo)), unshown(lo))
        if f[1] < full[lo]:
            put(lo, f)
    out, prev = [], -1
    for i in kept:
        if i > prev + 1:
            out.append(_note(note, i - prev - 1))
        out.append(lines[i])
        prev = i
    if prev < n - 1:
        out.append(_note(note, n - 1 - prev))
    shown = set(shown)
    return ("\n".join(out), len(picks), sum(weight[a] for a in seen if a in weight and a not in shown),
            any(a not in seen for a in prep.order))


def _unshown(anchors, shown):
    """`anchors` without those in `shown`."""
    return anchors.subset(anchors.all() - set(shown)) if anchors and shown else anchors


def _fit_lines(prep, budget, note, shown, context, by_place=False):
    """_select_lines at `budget`, its budget lowered until the rendered text fits (token joins) -> its result, or
    None."""
    target = budget
    for _ in range(6):
        got = _select_lines(prep, target, note, shown, context, by_place)
        if got is None:
            return None
        k = _ntok(got[0])
        if k <= budget:
            return got
        target -= k - budget + 2
    return None


def _cut_lines(prep, budget, note=LINES_NOTE, shown=(), context=None, dense=True):
    """A multi-line text (prepared: _TextCut) cut to at most `budget` tokens. Quotes come first: its first and last
    lines, then every anchor not in `shown` (what later turns quote, strongest first) at its least while they fit (the
    place a line shows it with a few tokens around each, or the whole line if that is no longer; each line holding it
    is tried in turn until one takes it), then those lines grow back (whole up to half the budget), then gaps of one
    or two lines between kept lines are filled; head and tail come from what is left: up to an eighth of the budget
    each, then more of both while they fit. Each run of omitted lines becomes one note (_note_tokens: priced as
    rendered). A head or tail line longer than a quarter of the budget shows only its ends and its quotes (cut
    inline), and while there are quotes to place a long first or last line starts at a sixteenth of the budget and
    grows back after them; what the lines leave goes back to the lines cut shorter and to the first long line after
    the head. A text whose lines are mostly quoted (prep.dense: a file read and then written back) that cannot show
    every quote is also cut with its quotes placed in the order of their lines, and that cut is kept if it shows more
    quoted weight: a note between lines kept apart costs about as much as a line, so a run of quoted lines shows more
    of them (dense=False: not tried here; the JSON cut tries it once, on the level it keeps). If no lines fit, the
    whole text is cut inline. `context`: cut around its quotes (_select_lines): its quotes and their lines within
    `budget`, the rest up to `context` tokens. -> (text, lines kept for a quote beyond the first and last lines, each
    showing its quote)."""
    got = _fit_lines(prep, budget, note, shown, context)
    if got is not None and got[3] and prep.dense and dense:
        alt = _fit_lines(prep, budget, note, shown, context, by_place=True)
        if alt is not None and alt[2] > got[2]:
            got = alt
    if got is None:
        return _inline(prep.text, budget, prep.ids, _unshown(prep.anchors, shown), prep.json_text, context=context)
    return got[:2]


def _cut_text(text, budget, anchors=None, note=LINES_NOTE, ids=None, prep=None, shown=(), json_text=False,
              spans=None, context=None, dense=True):
    """Plain text (or JSON cut as text: json_text) cut to at most `budget` tokens, by lines or (one long line) inline;
    no line or span is kept for the anchors in `shown`. `prep`: its _TextCut, `spans`: a one-line text's
    _anchor_spans, if known. `context`: cut around its quotes, which may take `budget`; the rest (head, tail) fills
    it up to `context` tokens. `dense`: see _cut_lines. -> (text, how: "lines" or "line", lines or runs of spans kept
    for a quote beyond the first and last lines or the head and tail)."""
    if "\n" not in text:
        if spans is None:
            spans = _anchor_spans(text, anchors, json_text) if anchors else ()
        if shown and spans:
            spans = spans.take(set(x[3] for x in spans) - set(shown))
        out, k = _inline(text, budget, ids, json_text=json_text, spans=spans, context=context)
        return out, "line", k
    out, k = _cut_lines(prep or _TextCut(text, anchors, ids, json_text), budget, note, shown, context, dense)
    return out, "lines", k


def truncate_text(text, max_tokens, note=LINES_NOTE, *, anchors=None):
    """A text cut to at most max_tokens tokens (unchanged if it fits): its first and last lines, then every one of
    `anchors` it holds (quote_anchors: what later turns quote) at its least while they fit (the place a line shows it,
    or the whole line; on any line holding it), those lines grown back, then head and tail lines from what is left,
    each run of omitted lines replaced by `note` (_cut_lines). A text of one long line keeps the places it shows
    anchors first, then its head and its tail (cut_span: UTF-8 safe), with CHARS_NOTE between. Measured as given
    (cut_tool_outputs measures the escaped text a rendering shows). Plain text only; cut_result also cuts JSON by its
    structure."""
    if _ntok(text) <= max_tokens:
        return text
    return _cut_text(text, max_tokens, anchors.restrict(text) if anchors else None, note)[0]


# ------------------------------------------------------------------------------------------------ cutting JSON
class _Num(str):
    """A JSON float as written (_json_value's parse_float), so a re-dump (_dump) writes it back unchanged: 25.00 stays
    25.00 and a float with more digits than a double keeps them. Held as NUL + text + NUL; never cut."""
    __slots__ = ()

    @property
    def text(self):
        return self[1:-1]


def _num(s):
    return _Num("\x00" + s + "\x00")


_NUM_DUMPED = re.compile(r'"\\u0000([^"\\]*)\\u0000"')


def _dump(value, style):
    """json.dumps(value, **style), with the floats _json_value kept written as they came."""
    out = json.dumps(value, **style)
    return _NUM_DUMPED.sub(lambda m: m.group(1), out) if "\\u0000" in out else out


def _json_value(text):
    """The JSON object, array or string a tool result holds, floats kept as written (_Num), or None (also for duplicate
    keys, which a re-dump would lose)."""
    t = text.strip()
    if len(t) < 2 or t[0] not in "{[\"":
        return None

    def pairs(kv):
        d = {}
        for k, v in kv:
            if k in d:
                raise ValueError("duplicate key")
            d[k] = v
        return d
    floats = {} if "\\u0000" in t or "\x00" in t else {"parse_float": _num}      # a NUL in the data: floats re-dumped
    try:
        return json.loads(t, object_pairs_hook=pairs, **floats)
    except (ValueError, RecursionError):
        return None


def _json_style(text, value):
    """json.dumps arguments that write `value` as `text` was written: indent, separators, ASCII escapes."""
    t = text.strip()
    indent = None
    if "\n" in t:
        for ln in t.split("\n")[1:]:
            ws = ln[:len(ln) - len(ln.lstrip(" \t"))]
            if ws:
                indent = "\t" if ws[0] == "\t" else len(ws)
                break
    ascii_only = t.isascii() and "\\u" in t
    styles = [{"indent": indent, "separators": (item, key), "ensure_ascii": ascii_only}
              for item in ((",", ", ") if indent is not None else (", ", ",")) for key in (": ", ":")]
    for style in styles:
        if _dump(value, style) == t:
            return style
    item = "," if indent is not None or ", " not in t else ", "
    return {"indent": indent, "separators": (item, ": " if '": ' in t else ":"), "ensure_ascii": ascii_only}


_JSON_ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|(["\\/bfnrt]))')
_JSON_ESCAPED = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


def _json_unescape(text):
    """`text` (JSON cut as text, no longer JSON) with JSON's string escapes decoded wherever they stand (\\n, \\",
    \\u00fc and surrogate pairs): what its strings say, so anchors found in a JSON value's decoded strings can be
    looked for in it."""
    if "\\" not in text:
        return text
    out = _JSON_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)) if m.group(1) else _JSON_ESCAPED[m.group(2)], text)
    return out.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")     # pairs -> characters


def _json_decoded(text):
    """(`text` with JSON's string escapes decoded wherever they stand, as _json_unescape, a function mapping a span
    (start, end) of the decoded text back to the span of `text` it came from)."""
    out, dpos, rpos, rend, pos, d = [], [], [], [], 0, 0
    ms = list(_JSON_ESCAPE.finditer(text))
    k = 0
    while k < len(ms):
        m = ms[k]
        s, e = m.span()
        out.append(text[pos:s])
        d += s - pos
        if m.group(1):
            u = int(m.group(1), 16)
            if 0xD800 <= u < 0xDC00 and k + 1 < len(ms) and ms[k + 1].start() == e and ms[k + 1].group(1):
                low = int(ms[k + 1].group(1), 16)
                if 0xDC00 <= low < 0xE000:                  # a surrogate pair: one character
                    u, e = 0x10000 + ((u - 0xD800) << 10) + (low - 0xDC00), ms[k + 1].end()
                    k += 1
            out.append(chr(u))
        else:
            out.append(_JSON_ESCAPED[m.group(2)])
        dpos.append(d)
        rpos.append(s)
        rend.append(e)
        d += 1
        pos = e
        k += 1
    out.append(text[pos:])

    def raw(x):
        j = bisect.bisect_right(dpos, x) - 1
        if j < 0:
            return x
        return rpos[j] if dpos[j] == x else rend[j] + x - dpos[j] - 1

    return "".join(out), lambda s, e: (raw(s), raw(e))


_JSON_STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
_JSON_SCALAR = re.compile(r"(?<![\w.-])(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|null)(?![\w.-])")
_JSON_TOKEN = re.compile(r'"(?:[^"\\\n]|\\.)*"|' + _JSON_SCALAR.pattern)     # (a string literal, or a scalar)
_COMPACT_NUMBERS = re.compile(r"\d,-?\d")     # (numbers written 412,512, which _NUMBER reads as one: 412512)


def _json_scalars(text, in_strings=True):
    """[(scalar, its start)]: the numbers (as written), true, false and null of JSON text. in_strings=False skips
    those inside string literals (whole JSON, whose quotes pair up); True reads them wherever they stand (a cut can
    leave the quotes unpaired)."""
    if in_strings:
        return [(m.group(), m.start()) for m in _JSON_SCALAR.finditer(text)]
    return [(m.group(), m.start()) for m in _JSON_TOKEN.finditer(text) if m.group()[0] != '"']


def _json_view(text, in_strings=True):
    """JSON cut as text, as the cut matches anchors in it: the text, then the text with JSON's escapes decoded, then
    each string literal in it (a key, a string value) decoded on its own lines and each scalar (a number as written,
    true, false, null) on its own line, as _leaf_text gives the keys, strings and scalars of whole JSON: a key written
    Malm\\u00f6 holds Malmö, a string value or a scalar that is a whole line an answer quotes (- 25.50) matches it, and
    the compact list [412,512] holds 412 and 512. What a cut shows is read with scalars wherever they stand (a cut can
    leave the quotes unpaired); what whole JSON holds, with in_strings=False (_json_scalars)."""
    out = [text]
    if "\\" in text:
        out.append(_json_unescape(text))
    if '"' in text:
        out += [_json_unescape(m.group(1)) for m in _JSON_STRING.finditer(text)]
    out += [x for x, _ in _json_scalars(text, in_strings)]
    return "\n".join(out) if len(out) > 1 else text


def _json_line_hits(lines, anchors):
    """_line_hits for JSON cut as text: each line matched as its _json_view of whole JSON (itself, its escapes decoded,
    its string literals and the scalars outside them on their own lines; a literal or a scalar only where it is a
    whole-line anchor, and the numbers of a line that writes them 412,512 one by one: its other numbers, words and
    phrases are in the decoded line already)."""
    view, owner, whole, nums = [], [], anchors.lines, anchors.numbers
    for i, ln in enumerate(lines):
        compact = bool(nums) and _COMPACT_NUMBERS.search(ln) is not None
        if "\\" not in ln and not compact and not (whole and ('"' in ln or _JSON_SCALAR.search(ln))):
            view.append(ln)
            owner.append(i)
            continue
        t = _strip_notes(ln)
        vs = [t] + (_json_unescape(t).split("\n") if "\\" in t else [])
        if whole:
            for m in _JSON_STRING.finditer(t):
                vs += [x for x in _json_unescape(m.group(1)).split("\n") if any(k in whole for k in _line_keys(x))]
        if compact or whole:
            vs += [x for x, _ in _json_scalars(t, False) if compact or any(k in whole for k in _line_keys(x))]
        view += vs
        owner += [i] * len(vs)
    hits = {}
    for j, f in _line_hits("\n".join(view), view, anchors).items():
        hits.setdefault(owner[j], set()).update(f)
    return hits


def _leaf_text(v):
    """A JSON value's keys, strings and scalars, one per line (each key before its value): what anchors are looked for
    in. Keys count: a key inside an elided list item (an id keying its record) goes with the item."""
    if type(v) is _Num:
        return v.text
    if isinstance(v, str):
        return v
    if isinstance(v, dict):         # (str(k): call arguments built in Python may have non-string keys)
        return "\n".join(str(k) + "\n" + _leaf_text(x) for k, x in v.items())
    if isinstance(v, list):
        return "\n".join(_leaf_text(x) for x in v)
    return json.dumps(v)


class _JsonCut:
    """One try at cutting a JSON value: each list to its `head` first and `tail` last items plus up to q items that
    show quotes nothing kept so far shows (strongest anchor first; _JSON_LEVELS keeps q unlimited until no head or
    tail item is left), each string to s tokens, except that in an item kept for its quotes a string holding a quote
    not shown yet is cut around its quotes: its quoted lines or places (with notes) may take up to qs tokens (the
    share, then less before the weakest quoted items go), and its head and tail fill it up to s tokens at most; one of
    QUOTED_STRING_TOKENS or less stays whole. Whole-line quotes keep room whatever qs (_whole_lines): a string keeps
    up to `whole` tokens for all of them and up to `one` tokens for one of them (half the share each; _JSON_LEVELS
    shrinks `whole` to none before the weakest quoted items go, and `one` only before the cut would fall to its
    skeleton). So a one-line string a later turn quotes as a whole line stays whole if it is no longer than `one`, and
    a multi-line string keeps such lines whole, its weakest quotes left out when they do not all fit. Walked in order,
    it remembers the anchors the output shows (.shown), so a quote is kept once, and an item counts as showing an
    anchor only if its cut still does (keys count: _leaf_text). An item whose cut shows no quote not shown yet is not
    kept for its quotes: the next item holding the quote is tried (at most 4 per quote; `tries`: at most that many in
    all per list, None for no limit), and quoted_lines counts only the items kept. `cache` (per list: its items,
    elided counts and quoted items; per string: its ids, anchors and prepared line cut or places) lasts across the
    tries of one value. `dense`: a string whose lines are mostly quoted also tries its quotes in line order
    (_cut_lines); without it, .dense_seen says whether such a string was cut."""

    def __init__(self, anchors, head, tail, s, q, qs, whole, one, cache, dense=False, tries=None):
        self.anchors, self.head, self.tail, self.s, self.q, self.qs = anchors or None, head, tail, s, q, qs
        self.whole, self.one = whole, one   # (the most a string keeps for all its whole-line quotes, and for one of
        #                                     them, at this level, whatever qs)
        self.cache, self.dense, self.dense_seen, self.tries = cache, dense, False, tries
        self.items_elided = self.strings_cut = self.quoted_lines = 0
        self.shown = set()

    def value(self, v, quoted=False):
        if isinstance(v, dict):
            return {key: self.value(x, quoted) for key, x in v.items()}
        if isinstance(v, list):
            return self.list(v, quoted)
        if type(v) is str and len(v) * 4 > self.s:          # (fewer bytes than s cannot make more than s tokens)
            return self.string(v, quoted)
        return v

    def string(self, v, quoted):
        info = self.cache["str"].get(id(v))
        if info is None:
            info = self.cache["str"][id(v)] = [encoding().encode_ordinary(v), False, None, None, (), {}]
        ids = info[0]                   # (ids, anchors, prepared lines or places of the anchors, the anchors as a set,
        #                                  the room of its whole-line quotes by caps while none of them is shown)
        if len(ids) <= self.s or _ITEMS_MARK.fullmatch(v):
            return v
        if info[1] is False:
            info[1] = (self.anchors.restrict(v) or None) if self.anchors else None
            info[4] = info[1].all() if info[1] else ()
        held = info[1]
        around = quoted and held is not None and not self.shown.issuperset(info[4])     # (a quote to show here)
        if around and len(ids) <= max(self.s, min(self.qs, QUOTED_STRING_TOKENS)):
            self.shown |= info[4]
            return v
        if "\n" in v:
            if info[2] is None:
                info[2] = _TextCut(v, held, ids)            # (held: its anchors, so the hits pass is short)
            self.dense_seen |= info[2].dense
        elif info[3] is None:
            info[3] = _anchor_spans(v, held) if held else ()
        if around:                      # around its quotes: they take up to qs, head and tail fill it up to s
            whole = self._whole_lines(held, ids, info[2], info[5])
            if whole and info[2] is None:                   # (one line a later turn quotes whole)
                self.shown |= info[4]
                return v
            out, _, k = _cut_text(v, max(self.s, self.qs, whole), held, ids=ids, prep=info[2], shown=self.shown,
                                  spans=info[3], context=self.s, dense=self.dense)
        else:
            out, _, k = _cut_text(v, self.s, held, ids=ids, prep=info[2], shown=self.shown, spans=info[3],
                                  dense=self.dense)
        if held:
            self.shown |= held.found(_strip_notes(out))
        self.strings_cut += 1
        self.quoted_lines += k
        return out

    def _whole_lines(self, held, ids, prep, memo):
        """The room the whole-line quotes of a string (held.lines that nothing shows yet) keep at this level. A
        one-line string: all its tokens, if that is at most self.one. A multi-line string: room for one of them, its
        cheapest line with a note on each side (at least twice the line: a quoted line is kept whole up to half the
        budget), at most self.one; and room for all of them, their lines (the cheapest line for each) with a note
        around each (at least twice the longest), at most self.whole. Only lines up to half the cap count (no longer
        one is kept whole in that room); 0 if there is none. A whole-line quote cannot be cut shorter, so a smaller qs
        would only lose it, and the item with it; when they do not all fit, the cut keeps the strongest (_cut_lines).
        `memo`: the string's rooms by caps while none of its whole-line quotes is shown (kept across tries)."""
        one, every = self.one, self.whole
        if not held.lines or one <= 0:
            return 0
        if prep is None:
            return len(ids) if len(ids) <= one and not self.shown.issuperset(held.lines) else 0
        fresh = self.shown.isdisjoint(held.lines)
        if fresh and (one, every) in memo:
            return memo[(one, every)]
        lines = set()
        for a in held.lines:
            if a not in self.shown:
                at = prep.where.get(a)
                if at:
                    lines.add(min(at, key=lambda j: (prep.costs[j], j)))
        got = 0
        if lines:
            nc = _note_tokens(LINES_NOTE, len(prep.lines))
            least = min(prep.costs[i] for i in lines)
            if least <= one // 2:
                got = min(one, max(2 * least, least + 2 * nc))
            cost = [prep.costs[i] for i in lines if prep.costs[i] <= every // 2]
            if cost:
                got = max(got, min(every, max(2 * max(cost), sum(cost) + (len(cost) + 1) * nc)))
        if fresh:
            memo[(one, every)] = got
        return got

    def _items(self, v):
        """(indices of the real items, elided count before each index, real items before each index): an earlier
        cut's marker counts as the items it stands for."""
        info = self.cache["list"].get(id(v))
        if info is None:
            real, before, reals, p, r = [], [0], [0], 0, 0
            for i, x in enumerate(v):
                m = _ITEMS_MARK.fullmatch(x) if type(x) is str else None
                if m:
                    p += int(m.group(1))
                else:
                    real.append(i)
                    p += 1
                    r += 1
                before.append(p)
                reals.append(r)
            info = self.cache["list"][id(v)] = (real, before, reals)
        return info

    def _leaf(self, v):
        """_leaf_text(v), kept for the containers of this value (a list's text holds its items')."""
        if not isinstance(v, (list, dict)):
            return _leaf_text(v)
        got = self.cache["leaf"].get(id(v))
        if got is None:
            got = ("\n".join(self._leaf(x) for x in v) if isinstance(v, list) else
                   "\n".join(str(k) + "\n" + self._leaf(x) for k, x in v.items()))
            self.cache["leaf"][id(v)] = got
        return got

    def _quoted(self, v, real):
        """({item: anchors it shows}, {anchor: items showing it}, anchors strongest first, {anchor: weight},
        {item: its anchors' total weight}) for a list, or (): one _line_hits pass over the items' joined texts, each
        line traced back to its item."""
        got = self.cache["q"].get(id(v))
        if got is None:
            quoted, where = {}, {}
            texts = [_strip_notes(self._leaf(v[i])) for i in real]
            joined = "\n".join(texts)
            if texts and self.anchors.found(joined):
                first = list(itertools.accumulate((t.count("\n") + 1 for t in texts), initial=0))
                for ln, f in _line_hits(joined, joined.split("\n"), self.anchors).items():
                    quoted.setdefault(real[bisect.bisect_right(first, ln) - 1], set()).update(f)
                for i in sorted(quoted):
                    for a in quoted[i]:
                        where.setdefault(a, []).append(i)
            wt = {a: self.anchors.weight(a) for a in where}
            got = ((quoted, where, sorted(where, key=lambda a: (-wt[a], where[a][0], a)), wt,
                    {i: sum(wt[a] for a in f) for i, f in quoted.items()}) if quoted else ())
            self.cache["q"][id(v)] = got
        return got

    def _still_shows(self, f, cut_item):
        """Which of the anchors `f` an item shows after its cut."""
        return self.anchors.subset(f).found(_strip_notes(_leaf_text(cut_item)))

    def list(self, v, quoted):
        real, before, reals = self._items(v)
        h, t = self.head, self.tail
        whole = len(real) <= h + t
        got = self._quoted(v, real) if self.q and self.anchors else ()
        items = got[0] if got else {}
        kept = {}
        for i in (real if whole else real[:h] + real[len(real) - t:]):    # first and last items (or all of them)
            f = items.get(i)                    # one showing a quote nothing shows yet is cut as kept for it
            kept[i] = self.value(v[i], quoted or bool(f and not f <= self.shown))
            if f:
                self.shown |= self._still_shows(f, kept[i])
        if got and not whole:                   # items for the quotes nothing kept so far shows, strongest first
            items, where, order, wt, total = got
            picks, shown, left = 0, self.shown, self.tries
            for a in order:
                if picks >= self.q or left == 0:
                    break
                if a in shown:
                    continue
                tried = set()                   # the items holding it in turn (at most 4) until one's cut shows it:
                for _ in range(4):              # the one showing the most weight not shown yet first (then the first)
                    if left == 0:
                        break
                    best = bs = None
                    for i in where[a][:64]:
                        if i not in kept and i not in tried and (bs is None or (total[i], -i) > bs):
                            sc = (sum(wt[x] for x in items[i] if x not in shown), -i)     # (its total bounds it)
                            if bs is None or sc > bs:
                                best, bs = i, sc
                    if best is None:
                        break
                    tried.add(best)
                    left = None if left is None else left - 1
                    n0, counts = len(shown), (self.items_elided, self.strings_cut, self.quoted_lines)
                    cut = self.value(v[best], True)
                    shown |= self._still_shows(items[best], cut)
                    if len(shown) == n0:        # its cut shows no quote not shown yet: not kept
                        self.items_elided, self.strings_cut, self.quoted_lines = counts
                        continue
                    kept[best] = cut
                    picks += 1
                    if a in shown or picks >= self.q:
                        break
            self.quoted_lines += picks
        out, prev = [], -1
        for i in sorted(kept):
            if before[i] > before[prev + 1]:
                out.append(_note(ITEMS_NOTE, before[i] - before[prev + 1]))
            self.items_elided += reals[i] - reals[prev + 1]
            out.append(kept[i])
            prev = i
        if before[len(v)] > before[prev + 1]:
            out.append(_note(ITEMS_NOTE, before[len(v)] - before[prev + 1]))
        self.items_elided += reals[len(v)] - reals[prev + 1]
        return out


# (first items and last items kept per list, tokens per string, more items per list for quotes, tokens a quoted string's
# quotes may take, then the room a string keeps whatever that is: for all its whole-line quotes, and for one of them;
# each the share divided by this, 0 for none), least cut first. Quoted items come first: every list keeps an item for
# each quote (q unlimited, _ALL_QUOTED), and the quoted strings in them may show their quotes in up to the share
# (_SHARE: the cut's budget), while head and tail items and string lengths shrink; only once no head or tail item is
# left do the quoted strings shrink (the bisection between the last level keeping every quoted item and the next: q
# stays at 1024 or more while qs comes down to QUOTED_STRING_TOKENS). Whole-line quotes keep up to half the share
# meanwhile. Then the room for all of a string's whole-line quotes shrinks from half the share to none, with bisection
# steps between levels, while every quoted item stays (up to 1024 per list): a string drops its weakest whole-line
# quotes before its item goes, and keeps room for one (its cheapest line, up to half the share; a one-line string
# whole). Then the weakest quotes go (q halving from 1024, with bisection steps between levels), and the last quoted
# item's strings shrink from QUOTED_STRING_TOKENS to half that. If the cut would still fall to the skeleton, the room
# for one whole-line quote shrinks from half the share to none first (_ROOM_LEVELS, searched only then), so a reserved
# room never takes the cut to its skeleton; _cut_json then also tries the last quoted item's strings down to
# MIN_STRING_TOKENS. The last level is the skeleton: every key and scalar, each list one count marker, each string
# MIN_STRING_TOKENS; JSON over its share even there is cut as text (cut_result).
_S, _Q = MIN_STRING_TOKENS, QUOTED_STRING_TOKENS
_ALL_QUOTED = 1 << 20
_SHARE = None
_JSON_LEVELS = ([(h, h, s, _ALL_QUOTED, _SHARE, 2, 2) for h, s in ((4096, 4096), (2048, 2048), (1024, 1024),
                                                                  (512, 512), (256, 256), (128, 128), (64, 64),
                                                                  (32, 32), (16, _S), (8, _S), (4, _S), (2, _S),
                                                                  (1, _S))]
                + [(1, 0, _S, _ALL_QUOTED, _SHARE, 2, 2), (0, 0, _S, _ALL_QUOTED, _SHARE, 2, 2)]
                + [(0, 0, _S, 1024, _Q, d, 2) for d in (2, 4, 8, 16, 0)]
                + [(0, 0, _S, q, _Q, 0, 2) for q in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1)]
                + [(0, 0, _S, 1, _Q // 2, 0, 2), (0, 0, _S, 0, _Q // 2, 0, 0)])
_ROOM_LEVELS = [(0, 0, _S, 1, _Q // 2, 0, d) for d in (4, 8, 16, 0)]      # (between the last two levels)


def _between(a, b, t):
    """A cut parameter a fraction t of the way from level value a to b (geometric while both are positive)."""
    if a > 0 and b > 0:
        return max(1, round(a * (b / a) ** t))
    return round(a + (b - a) * t)


def _cut_json(value, budget, anchors, style, cache=None):
    """JSON cut by its structure to at most `budget` tokens: middle list items elided with a count marker, long strings
    cut (head, quoted lines or spans, tail; in an item kept for its quotes, around its quotes), every key of every
    object kept, never cut inside an object. The least cut level of _JSON_LEVELS that fits wins (a binary search, then
    bisection steps between it and the level before; the room kept for all of a string's whole-line quotes shrinks
    before the weakest quoted items go; when the search ends at the skeleton, the _ROOM_LEVELS before it are searched
    the same way); when the level before keeps more quoted items, those items are tried with their quoted strings
    shorter (down to half of QUOTED_STRING_TOKENS; the last quoted item, before the skeleton, down to
    MIN_STRING_TOKENS), and that cut is kept if it fits and shows more quoted weight (a string cut too short no longer
    shows its quote). A string whose lines are mostly quoted (_TextCut.dense) also tries its quotes in line order,
    once, on the level kept; that cut is kept if it fits and shows more quoted weight. `cache`: _JsonCut's, kept
    between cuts of the same value with the same anchors. -> (text, the _JsonCut used, the cut value, over budget: True
    when even the skeleton does not fit, which is then what is returned)."""
    cache = {"str": {}, "list": {}, "q": {}, "leaf": {}} if cache is None else cache
    share = max(budget, _Q)

    def resolve(lv):
        return lv[:4] + (share if lv[4] is _SHARE else lv[4],) + tuple(share // d if d else 0 for d in lv[5:])
    levels = [resolve(lv) for lv in _JSON_LEVELS]
    tried = {}

    def attempt(params, dense=False, tries=None):
        cut = _JsonCut(anchors, *params, cache, dense=dense, tries=tries)
        cv = cut.value(value)
        out = _dump(cv, style)
        return out, _ntok(out) <= budget, cut, cv, params

    def shows(got):
        """The total weight of the anchors a try's cut value shows (its keys and decoded strings)."""
        return sum(map(anchors.weight, anchors.found(_strip_notes(_leaf_text(got[3])))))

    def fits(i):
        if levels[i] not in tried:
            tried[levels[i]] = attempt(levels[i])
        return tried[levels[i]][1]

    def search(lo, hi):
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        return lo, hi

    lo, hi = -1, len(levels) - 1    # level lo does not fit (-1: the result as it came), level hi does
    if not fits(hi):
        out, _, cut, cv, _ = tried[levels[hi]]
        return out, cut, cv, True
    lo, hi = search(lo, hi)
    if lo == hi - 1 == len(levels) - 2:     # the last quoted item would go: the room for one whole-line quote
        levels[hi:hi] = [resolve(lv) for lv in _ROOM_LEVELS]      # shrinks first (the main search keeps its levels)
        lo, hi = search(lo, hi + len(_ROOM_LEVELS))
    best, kept = tried[levels[hi]], levels[hi]
    if lo >= 0:                           # between the two levels: 4 bisection steps, 5 when only the rooms differ (a
        a, b, t0, t1 = levels[lo], levels[hi], 0.0, 1.0     # room moves every quoted string's cut at once)
        for _ in range(5 if a[:5] == b[:5] else 4):
            mid = (t0 + t1) / 2
            params = tuple(_between(x, y, mid) for x, y in zip(a, b))
            got = attempt(params)
            if got[1]:
                t1, best, kept = mid, got, params
            else:
                t0 = mid
        # quoted items lost for their strings' length: keep more of them, their quoted strings shorter (down to half;
        # the last quoted item, before the cut falls to its skeleton, down to MIN_STRING_TOKENS: a string's cut is
        # measured in its own tokens, and the dump's escapes can take it a token or two over; there each list tries 4
        # items at most, since strings that short may show no quote), if that fits and shows more quoted weight
        floor, tries = (_Q // 2, None) if a[4] > _Q // 2 else (_S, 4)
        if anchors and a[3] > kept[3] and a[4] > floor:
            lo_qs, hi_qs, top = floor, a[4], None
            for k in range(5):
                qs = lo_qs if k == 0 else (lo_qs + hi_qs) // 2
                got = attempt(a[:4] + (qs,) + a[5:], tries=tries)
                if got[1]:
                    lo_qs, w = qs, shows(got)
                    if top is None or w >= top[0]:            # (equal weight: the longer quoted strings)
                        top = w, got
                elif k == 0:
                    break
                else:
                    hi_qs = qs
                if hi_qs - lo_qs <= 1:
                    break
            if top is not None and top[0] > shows(best):
                best = top[1]
    if anchors and best[2].dense_seen:      # a string whose lines are mostly quoted (a file read, then written back):
        got = attempt(best[4], True, best[2].tries)     # its quotes also tried in line order, once, on the level kept
        if got[1] and shows(got) > shows(best):
            best = got
    out, _, cut, cv, _ = best
    return out, cut, cv, False


def cut_result(text, max_tokens, anchors=None, *, ids=None, _memo=None):
    """One tool result cut to at most max_tokens tokens -> (text, ResultCut); unchanged (how "") if it fits. Tokens
    are counted on `text` as given (cut_tool_outputs counts the escaped text a rendering shows).

    JSON (an object, an array or a string) is cut by its structure: middle list items become "[… N more items …]",
    the items that show what the later assistant turns quote come first (every one while they fit, then the first and
    last items), long strings keep their quoted lines or places first, then their head and tail (in an item kept for
    its quotes, a string is cut around them; lines quoted whole keep their room, up to half the share, and that room
    shrinks to one such line before the weakest quoted items go), every key is kept, nothing is cut mid-object, and
    the result is re-dumped in its own style (floats as written) only when cut. Anchors are looked for in its keys and
    decoded strings, so the lines of a JSON-wrapped log or file are matched as lines and a list item keyed by a quoted
    id is kept for it. JSON still over max_tokens as its skeleton (every key and scalar, each list one count marker,
    each string MIN_STRING_TOKENS) is cut as text (json_text_fallback): every line or place a later turn quotes
    first, then head and tail, the rest replaced by count markers; its anchors are matched in its decoded strings and
    its scalars too (_json_view: a key written with \\u escapes still counts, and so does a scalar an answer quotes as
    a line), and quoted keys it removes are reported lost like any other anchor. JSON that cannot be walked or
    written back (nested too deep, a lone surrogate) is cut as text the same way. Other
    text keeps its first and last lines, then every anchor (quote_anchors of the later assistant turns) at its least
    while they fit (the place a line shows it, or the whole line if that is no longer; each line holding it is tried
    in turn), then those lines whole up to half the budget, then head and tail lines from what is left (_cut_lines);
    one long line keeps the places it shows anchors first, then its head and its tail (_inline; cut_span: UTF-8 safe).
    The ResultCut counts the quoted anchors kept and lost (the lost ones in anchors_lost), the lines or items kept for
    them, items elided and strings cut. `ids`: the text's encode_ordinary ids, if known. `_memo`: a dict kept between
    cuts of the same text with the same anchors (at another budget), so its preparation is done once."""
    ids = encoding().encode_ordinary(text) if ids is None else ids
    before = len(ids)
    rc = ResultCut(share=max_tokens, tokens_before=before, tokens_after=before)
    if before <= max_tokens:
        return text, rc
    memo = {} if _memo is None else _memo
    out = shown = local = json_local = None
    if "value" not in memo:
        memo["value"] = _json_value(text)
    value = memo["value"]
    if value is not None:
        try:
            if "json" not in memo:
                memo["json"] = (anchors.restrict(_strip_notes(_leaf_text(value))) if anchors else None,
                                _json_style(text, value), {"str": {}, "list": {}, "q": {}, "leaf": {}})
            json_local, style, cache = memo["json"]
            if memo.get("skeleton", 0) > max_tokens:    # (its skeleton was over a larger budget already)
                over = True
            else:
                out, cut, cv, over = _cut_json(value, max_tokens, json_local, style, cache)
                if over:
                    memo["skeleton"] = _ntok(out)
            if over:                                    # even its skeleton is over: cut as text below
                out, rc.json_text_fallback = None, True
            else:
                out.encode("utf-8")                     # a lone surrogate written raw: not text a tokenizer can take
                shown, local = _leaf_text(cv), json_local
                rc.how, rc.items_elided = "json", cut.items_elided
                rc.strings_cut, rc.quoted_lines = cut.strings_cut, cut.quoted_lines
        except (RecursionError, UnicodeError):          # cut as text
            out = None
    if out is None:
        if "text" not in memo:                          # JSON cut as text is matched in its decoded strings too
            jt = value is not None                      # (_json_view: an escaped key or string still counts)
            prep = _TextCut(text, anchors, ids, jt) if "\n" in text else None
            held = (prep.anchors if prep else None if not anchors else
                    anchors.restrict(_json_view(_strip_notes(text), False) if jt else _strip_notes(text)))
            memo["text"] = [prep, held, jt, None if prep is None and held else ()]
        prep, local, jt, spans = memo["text"]
        if spans is None:                               # (one line: the places of its anchors, found once)
            spans = memo["text"][3] = _anchor_spans(text, local, jt)
        out, rc.how, rc.quoted_lines = _cut_text(text, max_tokens, local, ids=ids, prep=prep, json_text=jt,
                                                 spans=spans)
        shown = _json_view(_strip_notes(out)) if jt else out
        if rc.json_text_fallback and json_local:        # the JSON's own anchors count too (its keys, decoded strings)
            local = (local | json_local) if local else json_local
    rc.tokens_after = _ntok(out)
    if local:
        held = local.all()
        lost = held - local.found(_strip_notes(shown))
        rc.anchors_lost = tuple(sorted(lost))
        rc.quoted_kept, rc.quoted_lost = len(held) - len(lost), len(lost)
    return out, rc


# ------------------------------------------------------------------------------------- cutting a conversation's turns
def _fair_shares(sizes, budget, floor):
    """Max-min fair split of `budget` over results of these token sizes: a result under the level keeps its size, the
    others get the level, never less than `floor` (so many results together can exceed the budget)."""
    if sum(sizes) <= budget:
        return list(sizes)
    left, k, level = budget, len(sizes), 0
    for s in sorted(sizes):
        if s * k <= left:
            left -= s
            k -= 1
        else:
            level = left // k
            break
    level = max(level, floor)
    return [min(s, level) for s in sizes]


def _rendered_tokens(text, n=None):
    """Tokens of a tool result as a rendering shows it: with its special-token strings escaped (escape_text), as
    render() writes every piece of data. `n`: the text's own token count, if known (the same when nothing is
    escaped)."""
    esc = escape_text(text)
    if esc is text:
        return _ntok(text) if n is None else n
    return _ntok(esc)


def _cut_rendered(text, share, anchors, ids, size, memo):
    """cut_result to at most `share` tokens as rendered (`size`: the text's rendered tokens, `ids`: its own ids): a
    result whose escaped special-token strings take more tokens than its text is cut again with a smaller budget
    until it fits. The ResultCut counts tokens as rendered."""
    budget = share
    for _ in range(8):
        out, rc = cut_result(text, budget, anchors, ids=ids, _memo=memo)
        n = _rendered_tokens(out, rc.tokens_after)
        if n <= share or budget <= 1:
            break
        budget = max(1, min(budget - 1, rc.tokens_after * share // n))     # its own tokens, scaled to fit rendered
    rc.share, rc.tokens_before, rc.tokens_after = share, size, n
    return out, rc


def _cut_turn(parts, ids, max_tokens, anchors, report, turn, json_text=None):
    """The results of one environment turn over max_tokens (`ids`: each one's tokens), each cut to its share; slack
    the cuts leave goes to the results that lost quoted anchors. Sizes, shares and cuts count tokens as rendered.
    `json_text`: a set that gets the index of each result that was JSON and was cut as text."""
    sizes = [_rendered_tokens(p, len(x)) for p, x in zip(parts, ids)]
    budget = max(1, max_tokens - (len(parts) - 1))          # the newlines between results
    shares = _fair_shares(sizes, budget, MIN_RESULT_TOKENS)
    new, cuts, memo = list(parts), {}, {}
    for j, part in enumerate(parts):
        if sizes[j] > shares[j]:
            new[j], cuts[j] = _cut_rendered(part, shares[j], anchors, ids[j], sizes[j], memo.setdefault(j, {}))
    slack = budget - sum(cuts[j].tokens_after if j in cuts else sizes[j] for j in range(len(parts)))
    needy = [j for j, rc in cuts.items() if rc.quoted_lost and not rc.over_budget]
    if needy and slack // len(needy) >= 8:
        extra = slack // len(needy)
        for j in needy:
            text, rc = _cut_rendered(parts[j], cuts[j].tokens_after + extra, anchors, ids[j], sizes[j], memo[j])
            if rc.quoted_lost < cuts[j].quoted_lost and rc.tokens_after <= cuts[j].tokens_after + extra:
                new[j], cuts[j] = text, rc
    for j in sorted(cuts):
        if cuts[j].how:                                     # (a result the extra let through whole is not a cut)
            cuts[j].turn, cuts[j].index = turn, j
            report.cuts.append(cuts[j])
            if json_text is not None and cuts[j].how != "json" and memo[j].get("value") is not None:
                json_text.add(j)
    return new


def _turn_text(m):
    """An assistant turn's text and its call arguments' values, one per line: what later turns see of it."""
    return "\n".join([m.content] + [_decoded(c.arguments) for c in m.tool_calls])


def _decoded(v):
    """_leaf_text(v), or "" for a value nested too deep to walk."""
    try:
        return _leaf_text(v)
    except RecursionError:
        return ""


def _shown_text(m, json_text=()):
    """What an environment turn shows, as the cut matches anchors: each result as written and, for a JSON result, its
    keys and decoded strings (so the lines of a JSON-wrapped log or file count as lines); a result that was JSON and
    was cut as text (its index in `json_text`) also as _json_view of it without the cut's notes, as cut_result read
    it, so a note's count never reads as a shown number. Other text is read as written (a plain log that opens with
    "[" is not read as JSON). The cut's own notes are left out."""
    out = []
    for j, p in enumerate(results_of(m)):
        out.append(p)
        if j in json_text:                              # (JSON cut as text: its decoded strings, as the cut read it)
            out.append(_json_view(_strip_notes(p)))
            continue
        v = _json_value(p)
        if v is not None:
            out.append(_decoded(v))
    return _strip_notes("\n".join(out))


def _cited_lost(conv, msgs, lost_at, mine, union, json_text=None):
    """The anchors the cut removed (lost_at: anchor -> the first environment turn whose cut lost it) that an assistant
    turn after that turn cites although nothing it sees shows them: not the system text or the tools, not an earlier
    user, assistant or (cut) environment turn, JSON results read as their decoded strings (`json_text`: environment
    turn index -> the results that were JSON and were cut as text, read as the cut read them). A turn before the cut
    output cannot quote it: a path an earlier call names is shown to every later turn by that call. -> (anchors_lost,
    weak_lost): numbers of 2 digits go in the second."""
    json_text = json_text or {}
    gone = union.subset(lost_at)
    seen = gone.found(conv.system + "\n" + (tools_json(conv.tools) + "\n" + _decoded(conv.tools) if conv.tools
                                            else ""))
    cited = set()
    for k, m in enumerate(msgs):
        if m.role == "assistant":
            cited |= {a for a in mine[k].all() if lost_at.get(a, k) < k} - seen
            seen |= gone.found(_turn_text(m))
        elif m.role == "environment":
            seen |= gone.found(_shown_text(m, json_text.get(k, ())))
        else:
            seen |= gone.found(_strip_notes(m.content))
    weak = {a for a in cited if a in gone.numbers and len(a) - ("." in a) < 3}
    return tuple(sorted(cited - weak)), tuple(sorted(weak))


def cut_tool_outputs(conv, max_tokens=TOOL_OUTPUT_TOKENS):
    """(Conversation, CutReport): every environment turn longer than max_tokens tokens cut to fit, result by result;
    shorter turns are left byte for byte. A turn is measured as rendered: its results joined one per line, with the
    special-token strings in them escaped (escape_text, as render() writes data), so the budget holds in the rendered
    conversation; its results' shares and cuts are counted the same way.

    The turn's results share its budget max-min fairly: one under the fair share stays whole, the others get the
    share (at least MIN_RESULT_TOKENS each). Each result over its share is cut on its own by cut_result, keeping what
    the later assistant turns quote (quote_anchors) first, then its head and its tail; slack the cuts leave goes to
    results that lost quoted anchors. The environment Messages keep the cut results in .parts. The cut depends only on
    its input (equal anchors are served in a fixed order, never in set order, so every process cuts alike whatever
    PYTHONHASHSEED). Cutting the output again changes nothing: every cut result fits its share, so the turn fits, as
    long as the results stay apart (a Conversation, one tool message per result, or an environment message's "parts";
    a turn written out as one joined message is one result to a second cut). The report says, per result, what was
    cut, kept and elided, and which quoted anchors a turn after the cut output cites that nothing it sees shows any
    more (CutReport.anchors_lost; 2-digit numbers in .weak_lost)."""
    conv = normalize(conv)
    msgs, report, enc = conv.messages, CutReport(), encoding()
    envs = {}                                   # environment turn index -> (results, the turn's ids if over max_tokens)
    for i, m in enumerate(msgs):
        if m.role == "environment":
            parts = results_of(m)
            joined = "\n".join(parts)
            ids = enc.encode_ordinary(joined)
            envs[i] = (parts, ids if _rendered_tokens(joined, len(ids)) > max_tokens else None)
    after, mine, union = {}, None, Anchors()
    if any(ids is not None for _, ids in envs.values()):      # anchors of the assistant turns after each turn to cut
        mine = [quote_anchors([m]) if m.role == "assistant" else None for m in msgs]
        acc = [set(), set(), set(), set()]
        for j in range(len(msgs) - 1, -1, -1):
            if j in envs and envs[j][1] is not None:
                after[j] = Anchors(*acc)
            if mine[j] is not None:
                for s, part in zip(acc, (mine[j].numbers, mine[j].words, mine[j].phrases, mine[j].lines)):
                    s |= part
        union = Anchors(*acc)
    out, lost_at, json_text = [], {}, {}        # anchor -> the first environment turn whose cut lost it; turn -> its
    for i, m in enumerate(msgs):                # results that were JSON and were cut as text
        if m.role != "environment":
            out.append(Message(m.role, m.content, m.tool_calls, m.train))
            continue
        parts, ids = envs[i]
        report.results += len(parts)
        if ids is not None:
            n0 = len(report.cuts)
            ids = [ids] if len(parts) == 1 else [enc.encode_ordinary(p) for p in parts]
            parts = _cut_turn(parts, ids, max_tokens, after[i], report, i, json_text.setdefault(i, set()))
            for rc in report.cuts[n0:]:
                for a in rc.anchors_lost:
                    lost_at.setdefault(a, i)
        out.append(Message("environment", "\n".join(parts), [], m.train, parts))
    if lost_at:
        report.anchors_lost, report.weak_lost = _cited_lost(conv, out, lost_at, mine, union, json_text)
    return Conversation(conv.system, conv.tools, out, cut_report=report), report


def truncate_tool_outputs(conv, max_tokens=TOOL_OUTPUT_TOKENS):
    """Environment turns longer than max_tokens cut to fit, each result on its own (cut_tool_outputs); they are
    untrained context. Returns the Conversation, whose .cut_report says what was cut. Idempotent (while the results
    stay apart: see cut_tool_outputs)."""
    return cut_tool_outputs(conv, max_tokens)[0]


def training_sample(conv, *, max_tokens=MAX_TOKENS, tool_output_tokens=TOOL_OUTPUT_TOKENS, plain_text=False,
                    tools=None):
    """The training document for a conversation (render(training=True), or render_text with plain_text=True), or None
    when it does not fit: drop the sample.

    Tool outputs are first cut (truncate_tool_outputs: an environment turn over tool_output_tokens keeps each result's
    quoted lines first, then its head and tail; they are untrained context); the document's .cut_report says what was
    cut. A conversation still longer than max_tokens is dropped, never split (a later window would hold steps that
    quote text the window no longer shows)."""
    conv = _training_conv(normalize(conv, tools))          # what follows the last assistant turn is not rendered
    report = None
    if tool_output_tokens:
        conv, report = cut_tool_outputs(conv, tool_output_tokens)
    r = render_text(conv) if plain_text else render(conv, training=True)
    r.cut_report = report
    return r if len(r.ids) <= max_tokens else None


def split_into_windows(conv, max_tokens=MAX_TOKENS, note=ELISION_NOTE):
    """Not used by default: over-16K samples are dropped instead (training_sample). Kept for experiments; later
    windows would need the observations their steps depend on pinned before they are valid training data.

    Cut a conversation too long for one training row into windows of at most max_tokens tokens (training=True
    rendering), at assistant-turn boundaries.

    A step is an assistant turn together with the tool results (or user turns) it reacts to, so every window starts
    with the context its first step needs and ends with an assistant turn. Every window repeats the system turn, the
    tools and the opening messages (the task); later windows add `note` to the task. Each assistant turn is trained in
    exactly one window. Returns [] if one step does not fit (drop the trajectory, or truncate tool outputs first)."""
    conv = _training_conv(normalize(conv))
    if len(render(conv, training=True).ids) <= max_tokens:
        return [conv]
    msgs = conv.messages
    first = next(i for i, m in enumerate(msgs) if m.role == "assistant")
    if first == 0:
        return []
    header, steps, pending = msgs[:first], [], []
    for m in msgs[first:]:
        pending.append(m)
        if m.role == "assistant":
            steps.append(pending)
            pending = []

    def window(body, later):
        head = [copy.copy(m) for m in header]
        if later:
            head[-1].content = head[-1].content.rstrip() + "\n\n" + note
            head[-1].parts = None               # an environment turn's results no longer join to its content
        return Conversation(conv.system, conv.tools, head + body)

    def fits(body, later):
        return len(render(window(body, later), training=True).ids) <= max_tokens

    out, body = [], []
    for step in steps:
        if fits(body + step, bool(out)):
            body = body + step
            continue
        if body:
            out.append(window(body, bool(out)))
        if not fits(step, True):
            return []
        body = list(step)
    if body:
        out.append(window(body, bool(out)))
    return out


# =====================================================================================================================
# The Jinja chat template (what vLLM / transformers render), generated from the same constants
# =====================================================================================================================
def _jinja_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def chat_template():
    """The chat template as Jinja source. Byte-identical to render(..., training=False) for OpenAI-style input as vLLM
    0.33 passes it (tool_calls arguments already parsed into dicts, developer turns converted to system, content
    parts joined). Keep it generated: edit this module and run `python -m nenya_vllm.olmo3_chat write-template`."""
    esc_chain = "".join(f" | replace({_jinja_str(raw)}, {_jinja_str(esc).replace(ZWSP, chr(92) + 'u200b')})"
                        for raw, esc in _ESCAPES)
    lines = [
        "{#- Nenya chat template: the OLMo 3 Instruct format with Nenya's conventions. GENERATED by",
        "    nenya_vllm/olmo3_chat.py -- edit the Python and run",
        "    `python -m nenya_vllm.olmo3_chat write-template`. -#}",
        # escape every special-token string in data (zero-width space after its first character)
        "{%- macro esc(s) -%}{{- s | string" + esc_chain + " -}}{%- endmacro -%}",
        # message content: a string, a list of text parts (joined by newlines), or nothing
        "{%- macro text_of(c) -%}",
        "{%- if c is string -%}{{- c -}}",
        "{%- elif c is iterable and c is not mapping -%}",
        "{%- set parts = namespace(items=[]) -%}",
        "{%- for p in c -%}",
        "{%- if p is mapping and p.get('type') == 'text' -%}{%- set parts.items = parts.items + [p.get('text') or ''] -%}",
        "{%- elif p is string -%}{%- set parts.items = parts.items + [p] -%}{%- endif -%}",
        "{%- endfor -%}",
        "{{- parts.items | join('\\n') -}}",
        "{%- endif -%}",
        "{%- endmacro -%}",
        # an argument value: raw between <raw_code> tags if it has a line break inside it, else JSON
        "{%- macro value(v) -%}",
        "{%- if v is string and '\\n' in v.strip('\\n') -%}{{- " + _jinja_str(S_RAW_BEGIN) + " + esc(v) + "
        + _jinja_str(S_RAW_END) + " -}}",
        "{%- else -%}{{- esc(v | tojson) -}}{%- endif -%}",
        "{%- endmacro -%}",
        # one tool, canonical: type, function{name, description if non-empty, parameters if given}
        "{%- macro tool_json(t) -%}",
        "{%- set f = t['function'] if (t is mapping and t.get('function') is mapping) else t -%}",
        "{{- '{\"type\": \"function\", \"function\": {\"name\": ' + (f['name'] | tojson) -}}",
        "{%- if f.get('description') -%}{{- ', \"description\": ' + (f['description'] | tojson) -}}{%- endif -%}",
        "{%- if f.get('parameters') is not none -%}{{- ', \"parameters\": ' + (f['parameters'] | tojson) -}}"
        "{%- endif -%}",
        "{{- '}}' -}}",
        "{%- endmacro -%}",
        # system and developer text anywhere -> the system turn; everything else in order
        "{%- set ns = namespace(sys='', msgs=[]) -%}",
        "{%- for m in messages -%}",
        "{%- if m['role'] == 'system' -%}",
        "{%- set c = text_of(m.get('content')) -%}",
        "{%- if c -%}{%- set ns.sys = (ns.sys + '\\n\\n' + c) if ns.sys else c -%}{%- endif -%}",
        "{%- else -%}{%- set ns.msgs = ns.msgs + [m] -%}{%- endif -%}",
        "{%- endfor -%}",
        "{%- set src = ns.sys | trim -%}",
        "{{- '<|im_start|>system\\n' + " + _jinja_str(IDENTITY) + " -}}",
        "{%- if src -%}{{- '\\n\\n' + esc(src) + '\\n\\n' -}}{%- else -%}{{- ' ' -}}{%- endif -%}",
        "{%- if tools -%}",
        "{%- set tj -%}{{- '[' -}}{%- for t in tools -%}{{- tool_json(t) -}}{%- if not loop.last -%}{{- ', ' -}}"
        "{%- endif -%}{%- endfor -%}{{- ']' -}}{%- endset -%}",
        "{{- " + _jinja_str(TOOL_SENTENCE + " " + S_FUNCTIONS_BEGIN) + " + esc(tj) + "
        + _jinja_str(S_FUNCTIONS_END) + " -}}",
        "{%- else -%}",
        "{{- " + _jinja_str(NO_TOOLS_SENTENCE + " " + S_FUNCTIONS_BEGIN + S_FUNCTIONS_END) + " -}}",
        "{%- endif -%}",
        "{{- '<|im_end|>\\n' -}}",
        "{%- for m in ns.msgs -%}",
        "{%- set role = m['role'] -%}",
        "{%- if role == 'user' -%}",
        "{{- '<|im_start|>user\\n' + esc(text_of(m.get('content'))) + '<|im_end|>\\n' -}}",
        "{%- elif role == 'assistant' -%}",
        "{{- '<|im_start|>assistant\\n' + esc(text_of(m.get('content')) | trim) -}}",
        "{%- if m.get('tool_calls') -%}",
        "{{- " + _jinja_str(S_CALLS_BEGIN) + " -}}",
        "{%- for tc in m['tool_calls'] -%}",
        "{%- set f = tc['function'] if (tc is mapping and tc.get('function') is mapping) else tc -%}",
        "{%- set args = f.get('arguments') -%}",
        "{{- esc(f['name']) + '(' -}}",
        "{%- if args is mapping -%}",
        "{%- for k, v in args.items() -%}{{- esc(k) + '=' + value(v) -}}{%- if not loop.last -%}{{- ', ' -}}"
        "{%- endif -%}{%- endfor -%}",
        "{%- elif args -%}{{- raise_exception('tool call arguments must be a JSON object') -}}{%- endif -%}",
        "{{- ')' -}}",
        "{%- if not loop.last -%}{{- '\\n' -}}{%- endif -%}",
        "{%- endfor -%}",
        "{{- " + _jinja_str(S_CALLS_END) + " + '<|im_end|>' -}}",
        "{%- elif loop.last -%}{{- '<|endoftext|>' -}}",
        "{%- else -%}{{- '<|im_end|>' -}}{%- endif -%}",
        "{%- if not loop.last -%}{{- '\\n' -}}{%- endif -%}",
        "{%- elif role == 'tool' or role == 'environment' -%}",
        "{%- if not (loop.previtem is defined and loop.previtem['role'] in ['tool', 'environment']) -%}"
        "{{- '<|im_start|>environment\\n' -}}{%- else -%}{{- '\\n' -}}{%- endif -%}",
        "{{- esc(text_of(m.get('content'))) -}}",
        "{%- if not (loop.nextitem is defined and loop.nextitem['role'] in ['tool', 'environment']) -%}"
        "{{- '<|im_end|>\\n' -}}{%- endif -%}",
        "{%- else -%}{{- raise_exception('unsupported role: ' + role) -}}",
        "{%- endif -%}",
        "{%- endfor -%}",
        "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}",
    ]
    return "\n".join(lines) + "\n"


def chat_template_path():
    """Path of the generated template file, for `vllm serve --chat-template`."""
    return TEMPLATE_PATH


def write_chat_template(path=TEMPLATE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        f.write(chat_template())
    return path


# =====================================================================================================================
# HF checkpoint files
# =====================================================================================================================
def export_hf(model_dir, *, write_template=True):
    """Make a checkpoint directory serve this format: name ids 100270/100271 <raw_code>/</raw_code> (non-special, so
    they survive skip_special_tokens) in tokenizer.json and tokenizer_config.json, set generation_config.json's
    eos_token_id to [<|im_end|>, <|endoftext|>], and write chat_template.jinja. Edits files in place."""
    changed = []
    tj = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tj):
        with open(tj, encoding="utf-8") as f:
            data = json.load(f)
        names = {v: k for k, v in SPECIAL_TOKENS.items()}
        # the BPE vocabulary lists the added tokens too, and `tokenizers` takes an added token's id from it (by
        # content), not from the added_tokens entry: rename there as well, or "<raw_code>" gets a new id 100278
        vocab = data.get("model", {}).get("vocab")
        if isinstance(vocab, dict):
            for i in (RAW_BEGIN, RAW_END):
                if vocab.get(names[i], i) != i:
                    raise ValueError(f"{names[i]!r} already has id {vocab[names[i]]} in {tj}")
            data["model"]["vocab"] = {(names[v] if v in (RAW_BEGIN, RAW_END) else k): v for k, v in vocab.items()}
        seen = set()
        for tok in data.get("added_tokens", []):
            if tok.get("id") in (RAW_BEGIN, RAW_END):
                tok["content"], tok["special"] = names[tok["id"]], False
                seen.add(tok["id"])
        missing = {RAW_BEGIN, RAW_END} - seen
        for i in sorted(missing):
            data.setdefault("added_tokens", []).append({"id": i, "content": names[i], "single_word": False,
                                                        "lstrip": False, "rstrip": False, "normalized": False,
                                                        "special": False})
        with open(tj, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        changed.append(tj)
    tc = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tc):
        with open(tc, encoding="utf-8") as f:
            cfg = json.load(f)
        dec = cfg.get("added_tokens_decoder")
        if isinstance(dec, dict):
            for i, name in ((RAW_BEGIN, S_RAW_BEGIN), (RAW_END, S_RAW_END)):
                entry = dec.setdefault(str(i), {"content": name, "lstrip": False, "normalized": False,
                                                "rstrip": False, "single_word": False, "special": False})
                entry["content"], entry["special"] = name, False
        cfg.pop("chat_template", None)          # chat_template.jinja wins; avoid two copies drifting apart
        with open(tc, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        changed.append(tc)
    gc = os.path.join(model_dir, "generation_config.json")
    gen = {}
    if os.path.exists(gc):
        with open(gc, encoding="utf-8") as f:
            gen = json.load(f)
    gen["eos_token_id"] = list(STOP_IDS)
    with open(gc, "w", encoding="utf-8") as f:
        json.dump(gen, f, indent=2)
    changed.append(gc)
    if write_template:
        changed.append(write_chat_template(os.path.join(model_dir, "chat_template.jinja")))
    return changed


# =====================================================================================================================
# CLI
# =====================================================================================================================
DEMO = {
    "tools": [{"type": "function", "function": {
        "name": "write_file", "description": "Write a text file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}}],
    "messages": [
        {"role": "user", "content": "Create greet.py with a greet(name) function."},
        {"role": "assistant", "content": "I'll write the file.", "tool_calls": [{"type": "function", "function": {
            "name": "write_file",
            "arguments": {"path": "greet.py", "content": "def greet(name):\n    print(\"Hi,\\n\" + name)\n"}}}]},
        {"role": "tool", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done: greet.py defines greet(name)."},
    ],
}


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-template", help="regenerate the Jinja chat template file")
    w.add_argument("--out", default=TEMPLATE_PATH)
    e = sub.add_parser("export-hf", help="patch a checkpoint directory (tokenizer names, stop ids, chat template)")
    e.add_argument("model_dir")
    e.add_argument("--no-template", action="store_true")
    sub.add_parser("demo", help="print a rendering and its trained tokens")
    a = ap.parse_args(argv)
    if a.cmd == "write-template":
        print(write_chat_template(a.out))
    elif a.cmd == "export-hf":
        for p in export_hf(a.model_dir, write_template=not a.no_template):
            print("updated", p)
    elif a.cmd == "demo":
        r = render(DEMO, training=True)
        print(r.text)
        print(f"\n{len(r.ids)} tokens, {sum(r.mask)} trained; trained text:\n{r.trained_text()}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
