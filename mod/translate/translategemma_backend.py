"""Official TranslateGemma Ollama prompt + offline source language selection.

Uses the vendor-documented two-newline prompt and ISO codes, not generic LLM
translation instructions. All inference stays on the project's localhost Ollama.
"""
from __future__ import annotations
import json
import re
import urllib.request

from model_catalog import OLLAMA_URL

KOREAN = re.compile(r"[\uac00-\ud7a3]")
LANG_CODES = {
    "English": "en", "German": "de", "French": "fr",
    "Japanese": "ja", "Chinese (Simplified)": "zh", "Chinese (Traditional)": "zh",
    "Spanish": "es", "Italian": "it", "Portuguese": "pt", "Russian": "ru",
    "Arabic": "ar", "Hindi": "hi", "Vietnamese": "vi",
    "Indonesian": "id", "Thai": "th", "Turkish": "tr", "Dutch": "nl",
    "Polish": "pl", "Czech": "cs", "Ukrainian": "uk", "Hebrew": "he",
    "Persian": "fa", "Bengali": "bn", "Filipino": "tl", "Khmer": "km",
    "Tamil": "ta", "Urdu": "ur", "Swahili": "sw",
}
ISO_NAMES = {code: name for name, code in LANG_CODES.items()}
ISO_NAMES["zh"] = "Chinese"
ISO_NAMES["tl"] = "Filipino"
ISO_NAMES["fa"] = "Persian"


def language(text: str, name: str) -> tuple[str, str]:
    if name != "auto":
        if name not in LANG_CODES:
            raise ValueError(f"TranslateGemma source language unsupported: {name}")
        return name, LANG_CODES[name]
    try:
        from langdetect import DetectorFactory, detect
    except ImportError as exc:
        raise RuntimeError("Missing offline language detector. Run out/install.bat --python") from exc
    if len(text.strip()) < 40:
        raise ValueError("Auto language identification needs >=40 characters; set --source-language")
    DetectorFactory.seed = 0
    code = detect(text[:5000])
    code = {"zh-cn": "zh", "zh-tw": "zh", "pt-br": "pt"}.get(code, code)
    if code not in ISO_NAMES:
        raise ValueError(f"Unrecognized source language '{code}', use --source-language")
    return ISO_NAMES[code], code


def prompt(text: str, lang: str, code: str) -> str:
    return (
        f"You are a professional {lang} ({code}) to Korean (ko) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {lang} "
        "text while adhering to Korean grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the Korean translation, without any additional explanations "
        f"or commentary. Please translate the following {lang} text into Korean:\n\n"
        + text
    )


def translate(text: str, model_tag: str, lang: str, code: str, timeout: int = 600) -> str:
    body = {
        "model": model_tag,
        "messages": [{"role": "user", "content": prompt(text, lang, code)}],
        "stream": False, "think": False, "keep_alive": "3m",
        "options": {"temperature": 0, "num_ctx": 4096,
                    "num_predict": min(2048, max(512, int(len(text)*2.4)))},
    }
    request = urllib.request.Request(
        OLLAMA_URL + "/api/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    output = str(result.get("message", {}).get("content") or "").strip()
    output = re.sub(r"(?s)^<think>.*?</think>\s*", "", output).strip()
    if not result.get("done") or result.get("done_reason") == "length":
        raise RuntimeError("TranslateGemma returned incomplete or truncated output")
    if not KOREAN.search(output):
        raise ValueError(f"TranslateGemma produced no Korean output for {model_tag}")
    return output
