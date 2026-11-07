"""LzTranslate batch CLI: SRT/TXT/MD to Korean, without overwriting source.

The model server is isolated at 127.0.0.1:12545 with its own model store.
Use python pipeline.py --check or --dry-run before a real job.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from compare_srt import read_srt, validate_alignment
from model_catalog import (
    APP, BY_ID, LANGUAGES, MODEL_DIR, MODELS, OLLAMA_EXE, OLLAMA_URL,
    available, get_installed, status_rows,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
EXTENSIONS = {".srt", ".txt", ".md"}
KOREAN = re.compile(r"[\uac00-\ud7a3]")

def say(*items: object) -> None:
    print(*items, flush=True)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def ollama_request(path: str, payload: dict | None = None, timeout: int = 5) -> dict:
    request = urllib.request.Request(
        OLLAMA_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)

def server_running() -> bool:
    try:
        response = ollama_request("/api/version", timeout=2)
        return isinstance(response.get("version"), str)
    except (OSError, ValueError, TimeoutError):
        return False

@contextlib.contextmanager
def server(need: bool):
    """Reuse the dedicated server or start/stop only the one this run owns."""
    if not need:
        yield
        return
    if server_running():
        say("[SERVER] Dedicated Ollama endpoint already available:", OLLAMA_URL)
        yield
        return
    if not OLLAMA_EXE.is_file():
        raise FileNotFoundError(
            f"Isolated Ollama missing: {OLLAMA_EXE}. Run out\\install.bat --engine"
        )
    log_dir = APP / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / ("ollama_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".log")
    environment = os.environ.copy()
    environment["OLLAMA_HOST"] = OLLAMA_URL.removeprefix("http://")
    environment["OLLAMA_MODELS"] = str(MODEL_DIR / "ollama")
    environment["OLLAMA_NO_CLOUD"] = "1"
    with log_file.open("x", encoding="utf-8") as logfile:
        process = subprocess.Popen(
            [str(OLLAMA_EXE), "serve"], cwd=str(OLLAMA_EXE.parent),
            env=environment, stdin=subprocess.DEVNULL,
            stdout=logfile, stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(60):
                if server_running():
                    break
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Isolated Ollama exited with {process.returncode}; log: {log_file}"
                    )
                time.sleep(0.5)
            else:
                raise TimeoutError(f"Isolated Ollama did not start; log: {log_file}")
            say("[SERVER] Started isolated Ollama:", OLLAMA_URL)
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            capture_output=True, timeout=10, check=False,
                        )
                    else:
                        process.kill()
                    process.wait(timeout=10)

def collect_inputs(args: argparse.Namespace) -> list[Path]:
    values = [Path(p) for p in args.input]
    values.extend(Path(p) for p in args.files)
    if args.folder:
        if not args.folder.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {args.folder}")
        globber = args.folder.rglob if args.recursive else args.folder.glob
        values.extend(
            p for p in globber("*")
            if p.suffix.lower() in EXTENSIONS
            and ".ko." not in p.name.casefold()
        )
    if not values:
        raise ValueError("Choose --input PATH or --folder DIR")
    unique = {}
    for item in values:
        src = item.expanduser().resolve(strict=True)
        if not src.is_file() or src.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Expected SRT, TXT or MD file: {src}")
        unique[str(src).casefold()] = src
    return sorted(unique.values(), key=lambda p: str(p).casefold())

def output_path(src: Path, model_id: str, output_dir: Path | None) -> Path:
    suffix = BY_ID[model_id].suffix(src.suffix.lower().lstrip("."))
    return (output_dir or src.parent) / (src.stem + suffix)

def validate_source(src: Path) -> None:
    if src.suffix.lower() == ".srt":
        read_srt(src)
    elif not src.read_text(encoding="utf-8-sig").strip():
        raise ValueError(f"Empty text input: {src}")

def validate_existing(src: Path, output: Path) -> None:
    if src.suffix.lower() == ".srt":
        validate_alignment(read_srt(src), read_srt(output))
    elif not output.read_text(encoding="utf-8-sig").strip():
        raise ValueError(f"Existing empty translation: {output}")

def translate_text(source: Path, output: Path, model, language: str,
                   input_hash: str) -> None:
    """Translate text paragraphs independently; keep blank paragraph separators."""
    if model.runner not in ("generic", "xiaomi", "tower", "gemmax", "translategemma", "index"):
        raise ValueError(f"Text translation unsupported by {model.id}; choose an SRT.")
    original = source.read_text(encoding="utf-8-sig")
    if model.runner == "translategemma":
        from translategemma_backend import language as identify_source
        from translategemma_backend import translate as specialist_translate
        detected_name, detected_code = identify_source(original, language)
    if model.runner == "index":
        from index_backend import detect_language as identify_index_source
        from index_backend import translate as index_translate
        _, detected_code = identify_index_source(original, language)
    sections = re.split(r"(\r?\n[ \t]*\r?\n)", original)
    results = []
    for section in sections:
        if not section.strip():
            results.append(section)
            continue
        if model.runner == "translategemma":
            results.append(specialist_translate(
                section, model.tag, detected_name, detected_code
            ))
            continue
        if model.runner == "index":
            results.append(index_translate(section, model.tag, detected_code))
            continue
        if model.runner == "xiaomi":
            prompt = f"Translate this from {language} to Korean:\n{language}: {section}\nKorean:"
        elif model.runner == "gemmax":
            prompt = f"Translate this from {language} to Korean:\n{language}: {section}\nKorean:"
        else:
            prompt = (
                "Translate the following text into natural Korean. "
                "Preserve names, numbers and meaning. Output only the translation.\n"
                f"{('Detect the source language automatically.' if language == 'auto' else 'Source language: ' + language)}\n\n{section}"
            )
        request = {
            "model": model.tag, "prompt": prompt, "stream": False,
            "keep_alive": "3m",
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 2048},
        }
        if model.runner in ("xiaomi", "gemmax"):
            request["raw"] = True
        response = ollama_request("/api/generate", request, timeout=600)
        if response.get("error") or not response.get("done"):
            raise RuntimeError(f"Model generation failed: {model.id}")
        if response.get("done_reason") == "length":
            raise RuntimeError(f"Output truncated by model token limit: {model.id}")
        text = str(response.get("response", "")).strip()
        text = re.sub(r"(?s)^<think>.*?</think>\s*", "", text).strip()
        if not KOREAN.search(text):
            raise ValueError(f"Non-Korean or empty translation: model={model.id}")
        results.append(text)
    if sha256(source) != input_hash:
        raise RuntimeError("Input changed during translation")
    with output.open("x", encoding="utf-8", newline="") as stream:
        stream.write("".join(results).rstrip() + "\n")
    validate_existing(output, output)

def command_for(model, source: Path, output: Path, language: str,
                batch_size: int, width: int, style: Path | None) -> list[str]:
    python = str(sys.executable)
    if model.runner == "translategemma":
        return [
            python, "-B", str(HERE / "translategemma_srt.py"),
            "--model", model.tag, "--source", str(source),
            "--output", str(output), "--source-language", language,
            "--width", str(width),
        ]
    if model.runner == "index":
        return [
            python, "-B", str(HERE / "index_srt.py"),
            "--model", model.tag, "--source", str(source),
            "--output", str(output), "--source-language", language,
            "--width", str(width),
        ]
    if model.runner == "generic":
        command = [
            python, "-B", str(HERE / "translate.py"), str(source),
            "--model", model.tag, "--output", str(output),
            "--source-language", language,
            "--batch-size", str(batch_size), "--width", str(width),
        ]
        if style:
            command += ["--md", str(style)]
        return command
    if model.runner == "xiaomi":
        alias = "milmmt12" if model.id == "mi12" else "milmmt4"
        return [
            python, "-B", str(HERE / "translate_specialist_srt.py"),
            "--model", alias, "--source", str(source),
            "--source-language", language, "--output", str(output),
        ]
    if model.runner == "nllb":
        venv_python = APP / "engine" / "venv" / "Scripts" / "python.exe"
        if not venv_python.is_file():
            raise FileNotFoundError("NLLB Python environment missing. Run install.bat --python")
        python = str(venv_python)
    command = [
        python, "-B", str(HERE / "translate_extra.py"),
        "--model", model.id, "--source", str(source),
        "--source-language", language, "--output", str(output),
    ]
    return command

def execute(command: list[str], model_id: str) -> None:
    environment = os.environ.copy()
    environment["LZTRANSLATE_OLLAMA_URL"] = OLLAMA_URL
    environment["TRANSLATE_PROFILE_DIR"] = str(APP / "metrics")
    environment["HF_HOME"] = str(APP / "models" / ".hf")
    say("[MODEL_START]", model_id)
    started = time.monotonic()
    result = subprocess.run(
        command, cwd=str(HERE), env=environment,
        stdin=subprocess.DEVNULL, timeout=7200, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Model {model_id} failed; exit={result.returncode}")
    say("[MODEL_DONE]", model_id, "seconds", round(time.monotonic()-started, 2))

def jobs(args: argparse.Namespace) -> int:
    sources = collect_inputs(args)
    ids = list(dict.fromkeys(s.strip() for s in args.models.split(",") if s.strip()))
    if not ids or any(m not in BY_ID for m in ids):
        raise ValueError(f"Invalid models: {args.models}. Use --check to view models.")
    if not 1 <= args.batch_size <= 8 or not 20 <= args.width <= 80:
        raise ValueError("Batch size must be 1..8 and width 20..80")
    if args.source_language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {args.source_language}")
    if args.md and not args.md.is_file():
        raise FileNotFoundError(args.md)
    folder = args.output_dir.expanduser().resolve(strict=True) if args.output_dir else None
    if folder and not folder.is_dir():
        raise NotADirectoryError(folder)
    tasks = []
    for src in sources:
        validate_source(src)
        for name in ids:
            model = BY_ID[name]
            if src.suffix.lower() != ".srt" and model.runner in ("nllb", "madlad", "chimera"):
                raise ValueError(f"Model {name} only accepts SRT; input: {src}")
            if args.source_language == "auto" and model.runner in (
                "xiaomi", "tower", "gemmax", "nllb", "chimera",
            ):
                raise ValueError(f"Select explicit --source-language for {name}")
            dst = output_path(src, name, folder)
            if str(dst).casefold() == str(src).casefold():
                raise ValueError("Output and input have same name")
            if dst.exists():
                validate_existing(src, dst)
                status = "SKIP_EXISTING"
            else:
                status = "PLANNED"
            tasks.append((src, model, dst, status))
            say("[PLAN]", status, src.name, model.id, "=>", dst)
    if args.dry_run:
        say("[PLAN_ONLY]", len(tasks), "jobs; no inference or downloads performed")
        return 0
    need_ollama = any(m.runner not in ("nllb", "madlad") and status == "PLANNED"
                      for _, m, _, status in tasks)
    with server(need_ollama):
        tags = get_installed()
        for _, model, _, status in tasks:
            if status == "PLANNED" and not available(model, tags):
                raise RuntimeError(
                    f"Model {model.id} unavailable. Run out\\install.bat --model {model.id}"
                )
        logdir = APP / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        summary = []
        for src, model, dst, status in tasks:
            if status == "SKIP_EXISTING":
                summary.append({"input": str(src), "model": model.id, "status": status})
                continue
            fingerprint = sha256(src)
            if src.suffix.lower() == ".srt":
                if model.runner == "chimera":
                    candidates = []
                    for other in ("mi12", "hy30", "hy7"):
                        m = BY_ID[other]
                        if not available(m, tags):
                            continue
                        cp = output_path(src, other, folder)
                        if cp.is_file():
                            validate_existing(src, cp)
                        else:
                            execute(command_for(
                                m, src, cp, args.source_language,
                                args.batch_size, args.width, args.md,
                            ), other)
                            validate_existing(src, cp)
                        candidates.append(cp)
                        if len(candidates) >= 2:
                            break
                    if len(candidates) < 2:
                        raise RuntimeError("Chimera requires at least two installed base models")
                    command = command_for(
                        model, src, dst, args.source_language,
                        args.batch_size, args.width, args.md,
                    )
                    for candidate in candidates:
                        command += ["--candidate", str(candidate)]
                else:
                    command = command_for(
                        model, src, dst, args.source_language,
                        args.batch_size, args.width, args.md,
                    )
                execute(command, model.id)
            else:
                translate_text(src, dst, model, args.source_language, fingerprint)
            if sha256(src) != fingerprint:
                raise RuntimeError("Input changed during translation: " + str(src))
            validate_existing(src, dst)
            record = {
                "input": str(src), "input_sha256": fingerprint,
                "output": str(dst), "output_sha256": sha256(dst),
                "model": model.id, "status": "COMPLETED",
            }
            summary.append(record)
            say("[PASS]", model.id, dst, record["output_sha256"])
        name = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        receipt = logdir / ("job_" + name + ".json")
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump({"status": "COMPLETED", "jobs": summary}, stream,
                      ensure_ascii=False, indent=2)
            stream.write("\n")
    say("[DONE]", len(summary), "jobs; results logged under", APP / "logs")
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local Korean translation: SRT subtitles, OCR text TXT/MD.",
        epilog=("Examples:\n"
                "  translate.bat \"captions.srt\"\n"
                "  translate.bat \"ocr.txt\"\n"
                "  translate.bat \"ocr.md\" --source-language Japanese\n"
                "  translate.bat --folder \"input\" --recursive --dry-run\n"
                "Output is written beside the input unless --output-dir is used.\n"
                "Existing files are never overwritten."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", help="One or more input paths; filename alone uses the top measured model")
    parser.add_argument("--input", action="append", default=[], help="Input file; repeatable")
    parser.add_argument("--folder", type=Path, help="Scan this folder for SRT/TXT/MD")
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories")
    parser.add_argument("--models", default="hy30", help="Comma-separated model IDs (default: hy30, highest measured COMET-DA on WMT24++ 76-item long-form sample)")
    parser.add_argument("--source-language", default="auto", choices=LANGUAGES,
                        help="Input language (default: auto; specialist models may need an explicit source)")
    parser.add_argument("--output-dir", type=Path, help="Existing directory; default input folder")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=42)
    parser.add_argument("--md", type=Path, help="Style instructions for generic Ollama")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true", help="Report installed local environment")
    args = parser.parse_args(argv)
    if args.check:
        say("[ENGINE]", OLLAMA_EXE, "present=", OLLAMA_EXE.is_file())
        say("[ENDPOINT]", OLLAMA_URL, "running=", server_running())
        say("[MODEL_HOME]", MODEL_DIR)
        for model, installed in status_rows():
            say("[MODEL]", model.id, "READY" if installed else "MISSING", model.title)
        return 0
    try:
        return jobs(args)
    except (OSError, RuntimeError, ValueError, TimeoutError,
            subprocess.TimeoutExpired, UnicodeError) as error:
        say("[FAILED]", type(error).__name__ + ":", error)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
