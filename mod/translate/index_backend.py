"""Official Index-Translate prompting on isolated Ollama.

Reference: https://huggingface.co/IndexTeam/Index-Translate-9B
The official source-language labels are Chinese; automatic detection is local.
"""
from __future__ import annotations
import json
import re
import urllib.request
from model_catalog import OLLAMA_URL
from translategemma_backend import language as detect_language

TARGET = "\u97e9\u6587"
SOURCES = {
    "en": "\u82f1\u6587", "de": "\u5fb7\u6587", "fr": "\u6cd5\u6587",
    "ja": "\u65e5\u6587", "zh": "\u4e2d\u6587", "es": "\u897f\u73ed\u7259\u6587",
    "it": "\u610f\u5927\u5229\u6587", "pt": "\u8461\u8404\u7259\u6587",
    "ru": "\u4fc4\u6587", "ar": "\u963f\u62c9\u4f2f\u6587",
    "hi": "\u5370\u5730\u6587", "ko": "\u97e9\u6587",
}
KOREAN = re.compile(r"[\uac00-\ud7a3]")


def prompt(text: str, source_code: str) -> str:
    title = SOURCES.get(source_code, "")
    return (
        f"\u8bf7\u5c06\u4ee5\u4e0b{title}\u6587\u672c\u7ffb\u8bd1\u4e3a{TARGET}\uff0c"
        "\u76f4\u63a5\u8f93\u51fa\u7ffb\u8bd1\u7ed3\u679c\uff0c"
        "\u4e0d\u8981\u8fdb\u884c\u4efb\u4f55\u89e3\u91ca\u3002\n\n"
        + text
    )


def translate(text: str, model_tag: str, source_code: str, timeout: int = 600) -> str:
    payload = {
        "model": model_tag,
        "messages": [{"role": "user", "content": prompt(text, source_code)}],
        "stream": False, "think": False, "keep_alive": "3m",
        "options": {"temperature": 0, "num_ctx": 8192,
                    "num_predict": min(3072, max(512, int(len(text)*2.4)))},
    }
    request = urllib.request.Request(
        OLLAMA_URL + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    result = str(data.get("message", {}).get("content") or "").strip()
    result = re.sub(r"(?s)^<think>.*?</think>\s*", "", result).strip()
    if not data.get("done") or data.get("done_reason") == "length":
        raise RuntimeError("Index-Translate output incomplete or truncated")
    if not KOREAN.search(result):
        raise ValueError(f"Index-Translate failed to generate Korean: {model_tag}")
    return result
