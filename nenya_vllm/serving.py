# SPDX-License-Identifier: Apache-2.0
"""HTTP middleware for `vllm serve`: serve/serve.sh passes `--middleware nenya_vllm.serving.NenyaServingMiddleware`.

It acts on POST .../chat/completions only and fixes two things the tool parser cannot:

1. Unescaping without tools. Nenya writes special-token strings in data with a zero-width space after the first
   character (`<\\u200b|im_end|>`), as training escapes them. The tool parser removes that space, but vLLM runs the
   parser only for requests with tools. For a request without tools (no tools, an empty list, or tool_choice "none")
   this middleware unescapes message.content in the JSON reply, and delta.content in a stream. In a stream, text
   that may be the start of an escape (a trailing "<" or "|") waits for the next chunk; the choice's last chunk
   (finish_reason set) carries whatever is left.
2. A trailing assistant message. Nenya's template closes a final assistant answer with <|endoftext|>, Nenya's
   document boundary, so a new turn after it would start in a context the model never saw; and an assistant message
   with tool calls must be followed by their results (OpenAI requires this too). Such a request gets HTTP 400 unless
   it sets continue_final_message (continue that message: vLLM then renders it without its closer). System and
   developer messages after the assistant message do not change this: the template moves them into the system turn,
   so the assistant turn still renders last.

Everything else passes through byte for byte. A plain ASGI middleware: it imports neither vLLM nor Starlette.
"""

from __future__ import annotations

import json
import re

from . import olmo3_chat as oc

ESCAPES = [t[0] + oc.ZWSP + t[1:] for t in oc.SPECIAL_TOKENS]
_ZWSP = oc.ZWSP.encode("utf-8")
_STARTS = "".join(sorted({e[0] for e in ESCAPES}))          # "<|": the first character of every escape
_MAX_HELD = max(len(e) for e in ESCAPES) - 1
# the end of a JSON string whose last character is an escape's first one, e.g. `...<"`
_STRING_ENDS = tuple(json.dumps(c)[1:-1].encode("utf-8") + b'"' for c in _STARTS)
_MAY_CHANGE = re.compile(b"|".join(re.escape(m) for m in (_ZWSP, b"\\u200") + _STRING_ENDS))


def _has_zwsp(data: bytes) -> bool:
    """A zero-width space in JSON bytes: raw UTF-8, or escaped (any \\u200x escape counts; a false hit only costs a
    parse)."""
    return _ZWSP in data or b"\\u200" in data


def _may_change(data: bytes) -> bool:
    """Stream bytes the unescaper may have to change: a zero-width space (as _has_zwsp finds it), or a JSON string
    that ends in "<" or "|", which the next chunk may turn into an escape. Any other "<" or "|" cannot start one
    (the zero-width space would follow it), so such text passes without a parse. A false hit only costs a parse."""
    return _MAY_CHANGE.search(data) is not None


# ---------------------------------------------------------------------------------------------------------------
# Request checks
# ---------------------------------------------------------------------------------------------------------------
def tool_parser_skipped(request: dict) -> bool:
    """True if vLLM will not run the tool parser on the reply: no tools, or tool_choice "none"."""
    return not request.get("tools") or request.get("tool_choice") == "none"


SYSTEM_ROLES = ("system", "developer")          # moved into the system turn (vLLM turns developer into system)


def last_turn(messages):
    """The message the template renders last: the last one that is not a system or developer message (those go into
    the system turn at the top, so [user, assistant, system] renders like [user, assistant]). None if there is none,
    or if that entry is not a dict (vLLM reports a malformed message itself)."""
    for m in reversed(messages):
        if not isinstance(m, dict):
            return None
        if m.get("role") not in SYSTEM_ROLES:
            return m
    return None


def trailing_assistant_problem(request: dict) -> str | None:
    """Why a chat request whose rendering ends with an assistant turn is refused, or None if it is fine. That is a
    conversation whose last message is an assistant message, not counting system or developer messages after it."""
    messages = request.get("messages")
    if not isinstance(messages, list) or request.get("continue_final_message"):
        return None
    last = last_turn(messages)
    if last is None or last.get("role") != "assistant":
        return None
    after = (" (system and developer messages after it go into the system turn, so they do not count)"
             if last is not messages[-1] else "")
    if last.get("tool_calls"):
        return (f"The conversation ends with an assistant message that has tool calls{after}: add a tool message "
                "with the result of each call before asking for the next turn.")
    return (f"The conversation ends with an assistant message{after}. Nenya's chat format closes a final assistant "
            "answer with <|endoftext|>, which ends a document, so a new turn after it would start in a context the "
            "model never saw. Add the next user (or tool) message, or set continue_final_message=true and "
            "add_generation_prompt=false to continue that assistant message.")


# ---------------------------------------------------------------------------------------------------------------
# Unescaping
# ---------------------------------------------------------------------------------------------------------------
def held_escape(text: str) -> int:
    """Characters at the end of `text` a stream must hold back: a possible start of an escaped special-token string
    (all start with "<" or "|"), and never part of a complete one (the escaped |||PHONE_NUMBER||| ends with a "|"
    that could start another)."""
    tail = text[-_MAX_HELD:]
    if oc.ZWSP not in tail:
        # an escape's second character is the space, so without it only a last "<" or "|" can start one
        return 1 if tail and tail[-1] in _STARTS else 0
    best = 0
    for esc in ESCAPES:
        for k in range(min(len(esc) - 1, len(text)), best, -1):
            if text.endswith(esc[:k]):
                best = k
                break
    b = len(text) - best
    moved = best > 0
    while moved:
        moved = False
        for esc in ESCAPES:
            s = text.find(esc, max(0, b - len(esc) + 1))
            if 0 <= s < b < s + len(esc):
                b, moved = s, True
    return len(text) - b


def _dumps(obj) -> bytes:
    # FastAPI's JSONResponse and vLLM's stream chunks: compact separators, UTF-8 kept
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def unescape_chat_response(body: bytes) -> bytes:
    """A complete /v1/chat/completions JSON reply with every choice's message.content unescaped."""
    if not _has_zwsp(body):
        return body
    try:
        data = json.loads(body)
    except ValueError:
        return body
    changed = False
    choices = data.get("choices") if isinstance(data, dict) else None
    for choice in choices if isinstance(choices, list) else []:
        msg = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            new = oc.unescape_text(msg["content"])
            if new != msg["content"]:
                msg["content"], changed = new, True
    return _dumps(data) if changed else body


class StreamUnescaper:
    """Unescapes delta.content across the server-sent events of one streamed reply (any number of choices)."""

    def __init__(self):
        self.buffer = b""
        self.pending = {}                       # choice index -> raw text held back
        self.last_chunk = None

    def feed(self, data: bytes, more: bool) -> bytes:
        """Bytes of the response body in, bytes to send out. An event split across body messages waits whole."""
        if not self.buffer and not self.pending and data.endswith(b"\n\n") and not _may_change(data):
            return data                         # whole events with nothing to unescape or hold: as they are
        self.buffer += data
        out = []
        while True:
            k = self.buffer.find(b"\n\n")
            if k < 0:
                break
            event, self.buffer = self.buffer[:k], self.buffer[k + 2:]
            out.append(self._event(event))
        if not more:
            out.append(self._flush())
            out.append(self.buffer)
            self.buffer = b""
        return b"".join(out)

    def _event(self, event: bytes) -> bytes:
        raw = event + b"\n\n"
        if not event.startswith(b"data:"):
            return raw
        payload = event[5:].strip()
        if payload == b"[DONE]":
            return self._flush() + raw
        if not self.pending and not _may_change(payload):
            return raw
        try:
            chunk = json.loads(payload)
        except ValueError:
            return raw
        if not isinstance(chunk, dict) or not isinstance(chunk.get("choices"), list):
            return raw
        self.last_chunk = chunk
        changed = False
        for choice in chunk["choices"]:
            if not isinstance(choice, dict):
                continue
            idx = choice.get("index", 0)
            delta = choice.get("delta")
            if delta is None and idx in self.pending:
                delta = choice["delta"] = {}
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                continue
            text = self.pending.pop(idx, "") + (content or "")
            if choice.get("finish_reason") is None:
                k = held_escape(text)
                if k:
                    self.pending[idx] = text[len(text) - k:]
                    text = text[:len(text) - k]
            new = oc.unescape_text(text)
            if content is None and not new:
                continue
            if new != content:
                delta["content"], changed = new, True
        return b"data: " + _dumps(chunk) + b"\n\n" if changed else raw

    def _flush(self) -> bytes:
        """Text still held back (no last chunk carried it): sent as one more chunk."""
        if not any(self.pending.values()):
            self.pending.clear()
            return b""
        base = self.last_chunk or {}
        chunk = {k: base[k] for k in ("id", "object", "created", "model") if k in base}
        chunk["choices"] = [{"index": i, "delta": {"content": oc.unescape_text(t)}, "finish_reason": None}
                            for i, t in sorted(self.pending.items()) if t]
        self.pending.clear()
        return b"data: " + _dumps(chunk) + b"\n\n"


# ---------------------------------------------------------------------------------------------------------------
# The ASGI middleware
# ---------------------------------------------------------------------------------------------------------------
def _header(message, name: bytes) -> bytes:
    for k, v in message.get("headers") or []:
        if k.lower() == name:
            return v.lower()
    return b""


class _Replay:
    """receive() for the app: the request messages already read, then the client's own receive (disconnects)."""

    def __init__(self, messages, receive):
        self.messages, self.receive = list(messages), receive

    async def __call__(self):
        if self.messages:
            return self.messages.pop(0)
        return await self.receive()


class _UnescapingSend:
    def __init__(self, send):
        self.send = send
        self.mode = None                        # "json" (held until complete), "sse", or "pass"
        self.start = None
        self.parts = []
        self.stream = StreamUnescaper()

    async def __call__(self, message):
        kind = message.get("type")
        if kind == "http.response.start":
            ctype = _header(message, b"content-type")
            if message.get("status") == 200 and ctype.startswith(b"application/json"):
                self.mode, self.start = "json", message
                return
            self.mode = "sse" if ctype.startswith(b"text/event-stream") else "pass"
            await self.send(message)
            return
        if kind != "http.response.body" or self.mode in (None, "pass"):
            await self.send(message)
            return
        more = bool(message.get("more_body", False))
        if self.mode == "json":
            self.parts.append(message.get("body", b""))
            if more:
                return
            body = unescape_chat_response(b"".join(self.parts))
            start = dict(self.start)
            start["headers"] = [(k, v) for k, v in self.start.get("headers") or [] if k.lower() != b"content-length"]
            start["headers"].append((b"content-length", str(len(body)).encode()))
            await self.send(start)
            await self.send({"type": "http.response.body", "body": body, "more_body": False})
            return
        out = self.stream.feed(message.get("body", b""), more)
        if out or not more:
            await self.send({"type": "http.response.body", "body": out, "more_body": more})


async def _send_error(send, status: int, message: str) -> None:
    body = json.dumps({"error": {"message": message, "type": "BadRequestError", "param": "messages",
                                 "code": status}}).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body, "more_body": False})


class NenyaServingMiddleware:
    """ASGI middleware (see the module docstring). `vllm serve --middleware nenya_vllm.serving.NenyaServingMiddleware`
    adds it with app.add_middleware(), which calls NenyaServingMiddleware(app)."""

    def __init__(self, app, unescape: bool = True, refuse_trailing_assistant: bool = True):
        self.app = app
        self.unescape = unescape
        self.refuse_trailing_assistant = refuse_trailing_assistant

    async def __call__(self, scope, receive, send):
        if (scope.get("type") != "http" or scope.get("method") != "POST"
                or not scope.get("path", "").rstrip("/").endswith("/chat/completions")):
            await self.app(scope, receive, send)
            return
        messages = []
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request" or not message.get("more_body", False):
                break
        try:
            request = json.loads(b"".join(m.get("body", b"") for m in messages if m.get("type") == "http.request"))
        except ValueError:
            request = None                      # vLLM reports the bad body itself
        if isinstance(request, dict) and self.refuse_trailing_assistant:
            problem = trailing_assistant_problem(request)
            if problem:
                await _send_error(send, 400, problem)
                return
        replay = _Replay(messages, receive)
        if self.unescape and isinstance(request, dict) and tool_parser_skipped(request):
            await self.app(scope, replay, _UnescapingSend(send))
        else:
            await self.app(scope, replay, send)
