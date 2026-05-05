from __future__ import annotations

import contextlib
import json
import re
from typing import Any, cast

from openai import InternalServerError, OpenAI
from openai.types.chat import ChatCompletionMessageParam

_MAX_CONTEXT_MESSAGES = 40  # system + first task + last N; prevents KV-cache RAM blowup


def _trim_context(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep system msg + first user task + most recent exchanges to stay within context window."""
    if len(messages) <= _MAX_CONTEXT_MESSAGES:
        return messages
    head = messages[:2]  # system prompt + first user task
    tail = messages[-(_MAX_CONTEXT_MESSAGES - 2) :]
    return head + tail


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction; falls back to wrapping plain text as an answer."""
    with contextlib.suppress(json.JSONDecodeError):
        return cast(dict[str, Any], json.loads(text))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        with contextlib.suppress(json.JSONDecodeError):
            return cast(dict[str, Any], json.loads(text[start : end + 1]))
    return {"action": "answer", "content": text}


def call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call llama-server and return a parsed dict.

    Uses plain-text completion so models that emit <think>…</think> blocks before
    JSON don't fail. Strips thinking blocks, then falls back to best-effort extraction.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            temperature=0.2,
            max_tokens=4096,
        )
    except InternalServerError as exc:
        raise RuntimeError(
            "Inference server returned 500. If this keeps happening try "
            'flash_attn = "off" or a smaller ctx_size in .locoder.toml, then restart.'
        ) from exc
    raw = _strip_thinking((resp.choices[0].message.content or "{}").strip())
    return _extract_json(raw)
