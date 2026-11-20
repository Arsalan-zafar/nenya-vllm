# SPDX-License-Identifier: Apache-2.0
"""The "nenya_olmo3" structural tag: xgrammar formats that constrain decoding to Nenya's call format.

vLLM 0.33 applies the tag for tool_choice "required", a named tool, parallel_tool_calls=False, and under "auto" when a
tool is strict (or the server runs --tool-strict-level parameter). tool_parser.py registers it; this module builds it
and imports neither vLLM nor xgrammar until a tag is built, so tests and scripts can use it without vLLM.

A call block is `<function_calls>name(key=value, ...)\\n...</function_calls>`. Each call is a TagFormat
`name(` + keyword arguments + `)`. A value is JSON, or a `<raw_code>...</raw_code>` block where any string is
allowed. With a strict schema the value must match it; otherwise any JSON value is allowed.

The bug this module fixes: the values used to be xgrammar
JSONSchemaFormat elements. xgrammar's JSON-schema grammar ends its string rule with a lookahead assertion,
`basic_string_sub ::= (...) (= [ \\n\\t]* [,}\\]:])`: after a closing quote only white space and one of `, } ] :` may
follow. xgrammar's token-mask cache rejects every token that crosses the end of a rule and breaks its lookahead,
without looking at what follows the rule. In this format a string value is followed by the call's `)`, so the merged
tokens `")` (909) and `")\\n` (1158) were masked, while `"` and `",` passed. Models then wrote `"x”)`, whose curly
quote the grammar reads as part of the string, and derailed.

The fix: each value is a GrammarFormat holding xgrammar's own JSON-schema grammar for that schema (same converter,
same strict checks) with its lookahead assertions removed. Without them, a token that crosses the end of a value is
checked against the whole grammar at run time, so `")`, `")\\n`, `"])` and `"})` are accepted where the format
allows them. Free-form keys (a tool that is not strict) also take every key the format allows, e.g. `max-results`.

The end of a call turn: Nenya's format closes a call turn with <|im_end|>. vLLM builds xgrammar's TokenizerInfo
with TokenizerInfo.from_huggingface(tokenizer, vocab_size=...) and no stop ids, so the TokenizerInfo's only stop token
is the tokenizer's eos, <|endoftext|>. But vLLM 0.33 gives each request's GrammarMatcher
override_stop_tokens=sampling_params.all_stop_token_ids (the request's stop ids, the eos and generation_config.json's
eos list), which after export-hf holds <|im_end|> too: there it is a stop token, allowed where the grammar can end and
masked inside values. Where the tag ends with the call block (required, a named tool, and auto with
one call), a TokenizerInfo-only stop set would let only <|endoftext|> follow </function_calls>, so a forced call turn
could not end the way the model was trained. The tag therefore also allows an optional "<|im_end|>" after the block;
vLLM ends the request on that token (generation_config.json lists it, see export-hf). Under auto with several calls
allowed, free text follows the block anyway.

Safety nets keep the server working if this xgrammar does not take the new tag, each with one warning: a value whose
grammar xgrammar cannot build or re-read falls back to JSONSchemaFormat; a tag that xgrammar cannot build with the
optional <|im_end|> is served without it; and a tag that xgrammar cannot build at all is replaced by the
legacy tag (build(..., legacy=True)), with the old masking.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging

from . import olmo3_chat as oc

logger = logging.getLogger(__name__)

CALLS_BEGIN, CALLS_END = oc.S_CALLS_BEGIN, oc.S_CALLS_END
# A keyword argument's name in a free-form call (no schema): olmo3_chat's key rule (_check_key, which render()
# applies to every call it writes), i.e. no "=", ",", brackets, quotes or line breaks, and no white space at either end.
KWARG_NAME = r"""[^ \t\n\r=,()\[\]{}"']+([ \t]+[^ \t\n\r=,()\[\]{}"']+)*"""
LEGACY_KWARG_NAME = r"[A-Za-z_][A-Za-z0-9_]*"            # the legacy pattern
_STRING_LIMITS = ("enum", "const", "pattern", "format", "minLength", "maxLength")
_SCHEMA_METADATA = {"description", "title", "examples", "default", "$comment"}
# xgrammar's JSON grammar allows any white space between JSON tokens (its default, any_whitespace=True): the same
# grammar the JSONSchemaFormat values had, so only the lookahead changes.
JSON_ANY_WHITESPACE = True
TURN_END = oc.S_IM_END                # what closes a call turn in Nenya's format
_warned: set[str] = set()
_built: dict[str, str] = {}           # sha256 of a tag's JSON -> "" if xgrammar builds it, else the error


def raw_allowed(schema) -> bool:
    """A <raw_code> block is free text: allowed where any string is (an unconstrained string, or any value)."""
    if schema is True:
        return True
    if not isinstance(schema, dict) or any(k in schema for k in _STRING_LIMITS):
        return False
    t = schema.get("type")
    if t is None:
        return set(schema) <= _SCHEMA_METADATA
    return t == "string" or (isinstance(t, list) and "string" in t)


# ---------------------------------------------------------------------------------------------------------------
# Lookahead assertions out of an EBNF grammar
# ---------------------------------------------------------------------------------------------------------------
def _skip_literal(s: str, i: int) -> int:
    """Index just past the string literal ("...") or character class ([...]) that starts at s[i]."""
    close = '"' if s[i] == '"' else "]"
    j = i + 1
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == close:
            return j + 1
        j += 1
    raise ValueError(f"unterminated {'string' if close == chr(34) else 'character class'} at {i}")


def _group_end(s: str, i: int) -> int:
    """Index just past the parenthesised group that opens at s[i]."""
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c in '"[':
            j = _skip_literal(s, j)
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise ValueError(f"unbalanced parenthesis at {i}")


def strip_lookaheads(ebnf: str) -> str:
    """The grammar without its lookahead assertions: every group `(= ...)` outside string literals, character
    classes and comments is removed. A lookahead only narrows what may follow a rule, so the grammar accepts the same
    strings; what changes is that xgrammar's mask cache no longer rejects tokens that cross the end of a rule."""
    out, i, n = [], 0, len(ebnf)
    while i < n:
        c = ebnf[i]
        if c in '"[':
            j = _skip_literal(ebnf, i)
            out.append(ebnf[i:j])
            i = j
            continue
        if c == "#":                                    # a comment runs to the end of the line
            j = ebnf.find("\n", i)
            j = n if j < 0 else j
            out.append(ebnf[i:j])
            i = j
            continue
        if c == "(":
            k = i + 1
            while k < n and ebnf[k] in " \t":
                k += 1
            if k < n and ebnf[k] == "=":
                while out and out[-1] in (" ", "\t"):
                    out.pop()                           # the space before the assertion
                i = _group_end(ebnf, i)
                continue
        out.append(c)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------
# Values, keyword arguments, calls
# ---------------------------------------------------------------------------------------------------------------
def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


@functools.lru_cache(maxsize=4096)
def value_ebnf(schema_json: str) -> str:
    """EBNF of one JSON value for a schema (a JSON string; "{}" = any value): xgrammar's own JSON-schema grammar,
    strict mode, with its lookahead assertions removed. Raises if xgrammar cannot build or re-read it."""
    import xgrammar as xgr

    grammar = xgr.Grammar.from_json_schema(schema_json, any_whitespace=JSON_ANY_WHITESPACE, strict_mode=True)
    ebnf = strip_lookaheads(str(grammar))
    xgr.Grammar.from_ebnf(ebnf)                         # the printed grammar must parse again
    return ebnf


def value_format(schema, legacy=False):
    """One argument value: JSON matching `schema` (True = any JSON), or a <raw_code> block where any string is
    allowed. legacy=True: the legacy JSONSchemaFormat value (for A/B checks and the fallback)."""
    from xgrammar.structural_tag import AnyTextFormat, JSONSchemaFormat, OrFormat, TagFormat

    value = None
    if not legacy:
        try:
            from xgrammar.structural_tag import GrammarFormat

            value_schema = {} if schema is True else schema
            value = GrammarFormat(grammar=value_ebnf(json.dumps(value_schema, ensure_ascii=False)))
        except Exception as err:  # noqa: BLE001 - keep serving with the old element rather than fail the request
            _warn_once(type(err).__name__, "nenya_olmo3 structural tag: JSON values fall back to JSONSchemaFormat, "
                       "which masks merged closing tokens such as '\")' (%s: %s)", type(err).__name__, str(err)[:200])
    if value is None:
        value = JSONSchemaFormat(json_schema=schema)
    if not raw_allowed(schema):
        return value
    return OrFormat(elements=[value, TagFormat(begin=oc.S_RAW_BEGIN, content=AnyTextFormat(), end=oc.S_RAW_END)])


def kwargs_format(function, params_of, legacy=False):
    """The keyword arguments of one call. `params_of(function)` gives the parameters schema when the tool is strict
    (vLLM's get_function_parameters), or something without "properties" for free-form arguments."""
    from xgrammar.structural_tag import (
        ConstStringFormat,
        OptionalFormat,
        OrFormat,
        RegexFormat,
        SequenceFormat,
        StarFormat,
    )

    params = params_of(function)
    props = params.get("properties") if isinstance(params, dict) else None
    if not isinstance(props, dict):
        # free-form keyword arguments with JSON (or raw) values
        kv = SequenceFormat(elements=[RegexFormat(pattern=LEGACY_KWARG_NAME if legacy else KWARG_NAME),
                                      ConstStringFormat(value="="), value_format(True, legacy)])
        return OptionalFormat(content=SequenceFormat(elements=[
            kv, StarFormat(content=SequenceFormat(elements=[ConstStringFormat(value=", "), kv]))]))
    if not props:
        return ConstStringFormat(value="")
    defs = {k: params[k] for k in ("$defs", "definitions") if k in params}
    names = list(props)
    required = [n for n in names if n in set(params.get("required", []))]

    def kv(name):
        schema = props[name] if isinstance(props[name], dict) else True
        if isinstance(schema, dict) and defs:
            schema = {**schema, **defs}
        return SequenceFormat(elements=[ConstStringFormat(value=f"{name}="), value_format(schema, legacy)])

    # arguments in schema order; required ones present, optional ones may be skipped. The first argument written
    # has no ", " in front, so enumerate which argument comes first.
    first_required = names.index(required[0]) if required else len(names) - 1
    alternatives = []
    for k in range(first_required + 1):
        elements = [kv(names[k])]
        for name in names[k + 1:]:
            sep_kv = SequenceFormat(elements=[ConstStringFormat(value=", "), kv(name)])
            elements.append(sep_kv if name in required else OptionalFormat(content=sep_kv))
        alternatives.append(SequenceFormat(elements=elements) if len(elements) > 1 else elements[0])
    if not required:
        alternatives.append(ConstStringFormat(value=""))
    return alternatives[0] if len(alternatives) == 1 else OrFormat(elements=alternatives)


def _assemble(tools, tool_choice, params_of, single_call, legacy, turn_end=False):
    """The tag. turn_end: an optional TURN_END after the call block where the tag would otherwise end with the block
    (see the module docstring); it changes nothing under auto with several calls allowed."""
    from xgrammar import StructuralTag
    from xgrammar.structural_tag import AnyTextFormat, TagFormat, TagsWithSeparatorFormat, TriggeredTagsFormat

    calls = [TagFormat(begin=f"{t.function.name}(", content=kwargs_format(t.function, params_of, legacy), end=")")
             for t in tools]
    if not calls:
        return StructuralTag(format=AnyTextFormat())
    block = TagFormat(begin=CALLS_BEGIN,
                      content=TagsWithSeparatorFormat(tags=calls, separator="\n", at_least_one=True,
                                                      stop_after_first=tool_choice == "forced" or single_call),
                      end=CALLS_END)
    if tool_choice == "auto":
        # single_call: the reply ends after the first block, so a second <function_calls> block cannot start (the
        # flag is passed only then: other requests build the same objects as before)
        once = {"stop_after_first": True} if single_call else {}
        fmt = TriggeredTagsFormat(triggers=[CALLS_BEGIN], tags=[block], **once)
        return StructuralTag(format=_then_turn_end(fmt) if single_call and turn_end else fmt)
    return StructuralTag(format=_then_turn_end(block) if turn_end else block)


def _then_turn_end(fmt):
    from xgrammar.structural_tag import ConstStringFormat, OptionalFormat, SequenceFormat

    return SequenceFormat(elements=[fmt, OptionalFormat(content=ConstStringFormat(value=TURN_END))])


def _build_error(tag) -> str:
    """The empty string if xgrammar builds a grammar from the tag (as vLLM will), else the error; remembered per
    tag."""
    try:
        import xgrammar as xgr

        spec = tag.model_dump_json()
    except Exception:  # noqa: BLE001 - nothing to check against here
        return ""
    key = hashlib.sha256(spec.encode("utf-8")).hexdigest()
    err = _built.get(key)
    if err is None:
        try:
            xgr.Grammar.from_structural_tag(spec)
            err = ""
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:200]}"
        if len(_built) >= 1024:
            _built.clear()
        _built[key] = err
    return err


def build(tools, tool_choice, params_of, *, single_call=False, legacy=False):
    """The structural tag for `tools` (OpenAI tool objects with .function.name / .parameters) and vLLM's tool_choice
    ("auto", "required" or "forced" for a named tool). "forced" or single_call allows one call per block; "auto"
    lets free text run until <function_calls>, then constrains the block, and with single_call the reply must end
    after that block. Where the reply ends with the block, an optional <|im_end|> may follow it.

    If xgrammar cannot build that tag, the same tag without the optional <|im_end|> is tried, then the
    legacy tag (each fallback logs one warning). legacy=True returns the legacy tag directly."""
    if legacy:
        return _assemble(tools, tool_choice, params_of, single_call, True)
    tag = _assemble(tools, tool_choice, params_of, single_call, False, turn_end=True)
    err = _build_error(tag)
    if not err:
        return tag
    plain = _assemble(tools, tool_choice, params_of, single_call, False)
    if plain.model_dump_json() != tag.model_dump_json() and not _build_error(plain):
        _warn_once("turn_end", "nenya_olmo3 structural tag: xgrammar rejects an optional '%s' after the call block "
                   "(%s); serving the tag without it, so such a turn can end only on the tokenizer's eos "
                   "(<|endoftext|>)", TURN_END, err)
        return plain
    _warn_once("tag", "nenya_olmo3 structural tag: xgrammar cannot build the tag (%s); serving the legacy "
               "tag, which masks merged closing tokens such as '\")'", err)
    return _assemble(tools, tool_choice, params_of, single_call, True)
