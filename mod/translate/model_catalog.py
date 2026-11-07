"""LzTranslate model catalog and isolated-runtime paths.

Historical chrF2 values are reference-similarity measurements, NOT accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
_DEFAULT_APP = (Path(Path(__file__).resolve().anchor) / "APP" / "LzTranslate"
                if os.name == "nt" else Path.home() / ".local" / "share" / "LzTranslate")
APP = Path(os.environ.get("LZTRANSLATE_APP_HOME") or _DEFAULT_APP)
MODEL_DIR = APP / "models"
ENGINE_DIR = APP / "engine"
OLLAMA_EXE = ENGINE_DIR / "ollama" / "ollama.exe"
OLLAMA_URL = os.environ.get("LZTRANSLATE_OLLAMA_URL", "http://127.0.0.1:12545").rstrip("/")
LANGUAGES = [
    "auto", "English", "Japanese", "Chinese (Simplified)",
    "Chinese (Traditional)", "German", "French", "Spanish",
    "Italian", "Portuguese", "Russian", "Arabic", "Hindi",
    "Vietnamese", "Indonesian", "Thai", "Turkish", "Dutch", "Polish",
    "Czech", "Ukrainian", "Hebrew", "Persian", "Bengali", "Filipino",
    "Khmer", "Tamil", "Urdu", "Swahili", "Korean",
]

@dataclass(frozen=True)
class Model:
    id: str
    title: str
    runner: str
    tag: str = ""
    s18: float | None = None
    s26: float | None = None
    gpt: float | None = None
    note: str = ""

    def suffix(self, ext: str = "srt") -> str:
        return f".{self.id}.ko.{ext}"

MODELS = (
    Model("mi12", "MiLMMT-46 12B Q5", "xiaomi",
          "hf.co/mradermacher/MiLMMT-46-12B-v1.0-GGUF:Q5_K_M", 32.76, 31.90, 22.50),
    Model("hy7", "Hy-MT2 7B Q4", "generic",
          "hf.co/tencent/Hy-MT2-7B-GGUF:Q4_K_M", 30.73, 31.14, 22.97),
    Model("hy30", "Hy-MT2 30B-A3B Q4", "generic",
          "hf.co/tencent/Hy-MT2-30B-A3B-GGUF:Q4_K_M", 30.55, 30.44, 26.56),
    Model("mi4", "MiLMMT-46 4B Q8", "xiaomi",
          "hf.co/mradermacher/MiLMMT-46-4B-v1.0-GGUF:Q8_0", 30.06, 29.79, 22.43),
    Model("hy18", "Hy-MT2 1.8B Q4 (custom)", "generic",
          "hy-mt2-1.8b-fixed:bench", 29.41, 28.36, 19.90,
          note="Needs local custom Modelfile and model import"),
    Model("gemma4", "Gemma 4 26B", "generic",
          "gemma4:26b", 29.33, 29.18, 29.64),
    Model("index35q4", "Index-Translate 35B-A3B Q4", "index",
          "index35q4:quality", note="Official IndexTeam ModelScope GGUF; SHA256 verified before import"),
    Model("index9", "Index-Translate 9B Q8", "index",
          "index-translate-9b-q8:ocr", note="Official translation-specialist Q8_0; 76-item long WMT24++ leader"),
    Model("tg27", "TranslateGemma 27B Q4", "translategemma",
          "translategemma:27b", note="Google TranslateGemma official 27B Q4_K_M"),
    Model("tg27q6", "TranslateGemma 27B Q6", "translategemma",
          "translategemma-27b-q6:quality", note="mradermacher Q6_K SHA256 verified via HF LFS"),
    Model("tg27q5", "TranslateGemma 27B Q5", "translategemma",
          "hf.co/mradermacher/translategemma-27b-it-GGUF:Q5_K_M",
          note="High fidelity GGUF Q5_K_M; ~20 GB, benchmark before default"),
    Model("tg27q8", "TranslateGemma 27B Q8", "translategemma",
          "translategemma:27b-it-q8_0", note="May offload to CPU on 24GB VRAM"),
    Model("tg12q8", "TranslateGemma 12B Q8", "translategemma",
          "translategemma:12b-it-q8_0"),
    Model("tg4q8", "TranslateGemma 4B Q8", "translategemma",
          "translategemma:4b-it-q8_0"),
    Model("qwen35", "Qwen3.5 27B", "generic",
          "qwen3.5:27b", 27.19, 26.56, 24.32),
    Model("tower9", "Tower+ 9B Q5", "tower",
          "hf.co/mradermacher/Tower-Plus-9B-GGUF:Q5_K_M",
          note="CC-BY-NC research only"),
    Model("gemmax9", "GemmaX2 9B Q5", "gemmax",
          "hf.co/mradermacher/GemmaX2-28-9B-v0.2-GGUF:Q5_K_M"),
    Model("madlad10", "MADLAD-400 10B Q5", "madlad",
          note="Requires local llama.cpp CUDA server"),
    Model("nllb33", "NLLB-200 3.3B INT8", "nllb",
          note="CC-BY-NC research only; separate CTranslate2 runtime"),
    Model("chimera7", "Hunyuan-MT Chimera 7B Q5", "chimera",
          "hf.co/mradermacher/Hunyuan-MT-Chimera-7B-GGUF:Q5_K_M",
          note="Requires two genuine first-stage translations"),
)
BY_ID = {m.id: m for m in MODELS}

MODEL_REPOS = {
    "index9": ("IndexTeam/Index-Translate-9B-GGUF", ("Index-Translate-9B.Q8_0.gguf",)),
    "tower9": ("mradermacher/Tower-Plus-9B-GGUF", ("Tower-Plus-9B.Q5_K_M.gguf",)),
    "gemmax9": ("mradermacher/GemmaX2-28-9B-v0.2-GGUF", ("GemmaX2-28-9B-v0.2.Q5_K_M.gguf",)),
    "chimera7": ("mradermacher/Hunyuan-MT-Chimera-7B-GGUF", ("Hunyuan-MT-Chimera-7B.Q5_K_M.gguf",)),
    "madlad10": ("thirteenbit/madlad400-10b-mt-gguf", ("model-q5_k_m.gguf",)),
    "nllb33": ("OpenNMT/nllb-200-3.3B-ct2-int8",
               ("model.bin", "config.json", "generation_config.json",
                "shared_vocabulary.json", "special_tokens_map.json",
                "tokenizer.json", "tokenizer_config.json")),
    "mi12": ("mradermacher/MiLMMT-46-12B-v1.0-GGUF", ("MiLMMT-46-12B-v1.0.Q5_K_M.gguf",)),
    "mi4": ("mradermacher/MiLMMT-46-4B-v1.0-GGUF", ("MiLMMT-46-4B-v1.0.Q8_0.gguf",)),
}
MODEL_ALIASES = {"mi12": "milmmt12", "mi4": "milmmt4"}

def score(value: float | None) -> str:
    return "\u2014" if value is None else f"{value:.2f}"

def get_installed() -> set[str]:
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=3) as response:
            payload = json.load(response)
        return {str(x.get("name", "")).casefold() for x in payload.get("models", [])}
    except (OSError, ValueError, TimeoutError):
        # Ollama may be stopped; inspect its isolated manifest store only.
        found: set[str] = set()
        manifest_roots = [
            MODEL_DIR / "ollama" / "manifests-v2",
            MODEL_DIR / "ollama" / "manifests",
        ]
        for item in MODELS:
            if not item.tag:
                continue
            name, _, version = item.tag.partition(":")
            version = version or "latest"
            parts = name.split("/")
            if len(parts) == 1:
                subpath = Path("registry.ollama.ai") / "library" / parts[0] / version
            elif len(parts) >= 3:
                subpath = Path(*parts) / version
            else:
                continue
            if any((root / subpath).is_file() for root in manifest_roots):
                found.add(item.tag.casefold())
        return found

def file_verified(alias: str) -> bool:
    report = MODEL_DIR / f"{alias}_verified.json"
    if not report.is_file():
        return False
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        filenames = data.get("files")
        if not filenames:
            return False
        return all((MODEL_DIR / alias / name).is_file() for name in filenames)
    except (OSError, ValueError, TypeError):
        return False

def available(model: Model, tags: set[str] | None = None) -> bool:
    if model.runner == "nllb":
        return file_verified("nllb33")
    if model.runner == "madlad":
        return file_verified("madlad10") and bool(llama_server_exe())
    return bool(model.tag) and model.tag.casefold() in (get_installed() if tags is None else tags)

def llama_server_exe() -> Path | None:
    paths = [
        ENGINE_DIR / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
        ENGINE_DIR / "llama.cpp" / "build" / "bin" / "llama-server.exe",
        ENGINE_DIR / "llama-server.exe",
    ]
    return next((p for p in paths if p.is_file()), None)

def status_rows() -> list[tuple[Model, bool]]:
    tags = get_installed()
    return [(m, available(m, tags)) for m in MODELS]

def live_quality(model: Model) -> tuple[float | None, float | None, float | None]:
    return model.s18, model.s26, model.gpt
