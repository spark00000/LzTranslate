"""Bootstrap an isolated LzTranslate runtime in the configured application root.

Official Ollama Windows ZIP, optional fresh llama.cpp Git checkout, Python
environment, and explicitly selected model assets. No global installers.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from model_catalog import (
    APP, ENGINE_DIR, MODEL_DIR, MODELS, BY_ID, MODEL_ALIASES,
    MODEL_REPOS, OLLAMA_EXE, OLLAMA_URL, file_verified,
    get_installed, llama_server_exe,
)
from pipeline import server

GITHUB_API = "https://api.github.com/repos/ollama/ollama/releases/latest"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def command(*args, cwd: Path | None = None, env: dict | None = None):
    print("[COMMAND]", " ".join(map(str, args)), flush=True)
    subprocess.run([str(x) for x in args], cwd=str(cwd) if cwd else None,
                   env=env, check=True)

def engine() -> None:
    if OLLAMA_EXE.is_file():
        print("[SKIP] Isolated engine already present:", OLLAMA_EXE, flush=True)
        return
    dest = ENGINE_DIR / "ollama"
    if dest.exists():
        raise FileExistsError(f"Partial Ollama folder exists; inspect it before reinstall: {dest}")
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        GITHUB_API,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "LzTranslate-install"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    asset = next(
        (a for a in release.get("assets", [])
         if a.get("name") == "ollama-windows-amd64.zip"), None
    )
    if not asset:
        raise RuntimeError("Official Windows AMD64 Ollama asset missing from release")
    digest = str(asset.get("digest") or "")
    if not digest.startswith("sha256:"):
        raise RuntimeError("Official release does not publish a verifiable SHA256")
    expected_size = int(asset["size"])
    url = str(asset["browser_download_url"])
    if not url.startswith("https://github.com/ollama/ollama/releases/download/"):
        raise RuntimeError("Unexpected Ollama download domain")
    archive = ENGINE_DIR / "ollama-windows-amd64.zip"
    partial = archive.with_suffix(".zip.part")
    if archive.exists() or partial.exists():
        raise FileExistsError(f"Previous installer archive exists: {archive} / {partial}")
    print("[DOWNLOAD]", release.get("tag_name"), "bytes=", expected_size, flush=True)
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "LzTranslate-install"}),
            timeout=120,
        ) as response, partial.open("xb") as file:
            shutil.copyfileobj(response, file, 4 * 1024 * 1024)
        if partial.stat().st_size != expected_size:
            raise ValueError("Ollama ZIP length differs from official release metadata")
        actual = sha256(partial)
        if actual.casefold() != digest.removeprefix("sha256:").casefold():
            raise ValueError("Ollama ZIP SHA256 mismatch")
        partial.rename(archive)
    finally:
        if partial.is_file():
            partial.unlink()
    dest.mkdir(parents=False, exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            path = (dest / item.filename).resolve()
            if not path.is_relative_to(dest.resolve()):
                raise ValueError("Unsafe ZIP member path")
        zipped.extractall(dest)
    if not OLLAMA_EXE.is_file():
        raise RuntimeError("Ollama ZIP extraction finished without ollama.exe")
    print("[PASS] Isolated Ollama:", OLLAMA_EXE, flush=True)
    print("[PASS] SHA256:", sha256(archive), flush=True)

def python_env() -> None:
    exe = ENGINE_DIR / "venv" / "Scripts" / "python.exe"
    if not exe.exists():
        candidates = [
            Path(os.environ.get("LZTRANSLATE_PYTHON312", "")),
            Path.home() / "AppData" / "Roaming" / "uv" / "python" /
            "cpython-3.12-windows-x86_64-none" / "python.exe",
        ]
        base = next((x for x in candidates if x.is_file()), None)
        if base is None:
            raise FileNotFoundError(
                "CPython 3.12 not found; set LZTRANSLATE_PYTHON312 to its python.exe"
            )
        ENGINE_DIR.mkdir(parents=True, exist_ok=True)
        command(str(base), "-m", "venv", str(exe.parents[1]))
    command(str(exe), "-m", "pip", "install", "--disable-pip-version-check",
            "huggingface-hub>=0.34,<2", "ctranslate2>=4.4,<5",
            "transformers>=4.50,<6", "sentencepiece>=0.2,<1",
            "protobuf>=5,<7", "sacrebleu==2.5.1",
            "langdetect==1.0.9")
    print("[PASS] Translation Python runtime:", exe, flush=True)

def llama() -> None:
    if llama_server_exe():
        print("[SKIP] llama.cpp server already present:", llama_server_exe(), flush=True)
        return
    folder = ENGINE_DIR / "llama.cpp"
    if folder.exists():
        if not (folder / ".git").exists():
            raise RuntimeError(f"Unexpected existing llama.cpp directory: {folder}")
        print("[RESUME] Existing cloned repository; no automatic git pull", flush=True)
    else:
        command("git", "clone", "--depth", "1", LLAMA_REPO, str(folder))
    command("cmake", "-S", str(folder), "-B", str(folder / "build"),
            "-DGGML_CUDA=ON", "-DLLAMA_BUILD_SERVER=ON")
    command("cmake", "--build", str(folder / "build"), "--config", "Release",
            "--target", "llama-server", "--parallel", "8")
    if not llama_server_exe():
        raise RuntimeError("llama.cpp build did not create llama-server.exe")
    print("[PASS] CUDA llama-server:", llama_server_exe(), flush=True)

def repo_assets(model_id: str) -> None:
    alias = MODEL_ALIASES.get(model_id, model_id)
    repo, names = MODEL_REPOS[model_id]
    base = MODEL_DIR / alias
    if file_verified(alias):
        print("[SKIP] Verified local Hugging Face weights:", alias, flush=True)
        return
    python_env()
    env_py = ENGINE_DIR / "venv" / "Scripts" / "python.exe"
    # Invoke this source file via the venv, isolating huggingface_hub dependencies.
    command(str(env_py), "-B", str(Path(__file__).resolve()),
            "--fetch-weights", model_id)

def fetch_weights(model_id: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    alias = MODEL_ALIASES.get(model_id, model_id)
    repo, names = MODEL_REPOS[model_id]
    base = MODEL_DIR / alias
    base.mkdir(parents=True, exist_ok=True)
    remote = HfApi().model_info(repo, files_metadata=True)
    files: dict[str, dict] = {}
    for name in names:
        item = next((x for x in remote.siblings if x.rfilename == name), None)
        if not item:
            raise FileNotFoundError(f"Upstream model file missing: {repo}/{name}")
        lfs = item.lfs
        remote_sha = (lfs.get("sha256") if isinstance(lfs, dict)
                      else getattr(lfs, "sha256", None)) if lfs else None
        expected_size = int(item.size or 0)
        if expected_size > 100 * 1024 * 1024 and not remote_sha:
            raise RuntimeError(f"No published SHA256 for large model file: {name}")
        if (base / name).is_file():
            actual_file = base / name
            print("[CHECK] Existing downloaded file:", actual_file, flush=True)
        else:
            actual_file = Path(hf_hub_download(
                repo_id=repo, filename=name, local_dir=str(base),
                force_download=False,
            ))
        actual_sha = sha256(actual_file)
        if expected_size and actual_file.stat().st_size != expected_size:
            raise ValueError(f"Model file size mismatch: {name}")
        if remote_sha and actual_sha.casefold() != remote_sha.casefold():
            raise ValueError(f"Model file SHA256 mismatch: {name}")
        files[name] = {
            "bytes": actual_file.stat().st_size,
            "actual_sha256": actual_sha,
            "upstream_sha256": remote_sha,
        }
        print("[PASS]", model_id, name, actual_sha, flush=True)
    receipt = MODEL_DIR / (alias + "_verified.json")
    report = {
        "model": model_id, "repository": repo,
        "repository_revision": remote.sha, "files": files,
    }
    if receipt.exists():
        raise FileExistsError("Existing unverified receipt must be inspected: " + str(receipt))
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    if not file_verified(alias):
        raise RuntimeError("Failed to validate local model receipt")
    print("[PASS] Verified weights:", model_id, flush=True)

def import_local(model_id: str) -> None:
    model = BY_ID[model_id]
    alias = MODEL_ALIASES.get(model_id, model_id)
    weights = MODEL_DIR / alias
    gguf = next((x for x in weights.glob("*.gguf") if x.is_file()), None) \
        if weights.is_dir() else None
    with server(True):
        if model.tag.casefold() in get_installed():
            print("[SKIP] Ollama model already installed:", model.tag, flush=True)
            return
        if gguf:
            desc = weights / "LzTranslate.Modelfile"
            templates = {
                "mi12": "MiLMMT12_q5.Modelfile",
                "mi4": "MiLMMT4_q8.Modelfile",
                "tower9": "TowerPlus9_Q5.Modelfile",
                "gemmax9": "GemmaX2_9B_Q5.Modelfile",
                "chimera7": "Chimera7_Q5.Modelfile",
            }
            template = Path(__file__).resolve().parent / templates.get(model_id, "")
            if not desc.exists():
                if template.is_file():
                    lines = template.read_text(encoding="utf-8").splitlines()
                    lines = ["FROM " + str(gguf) if row.startswith("FROM ") else row
                             for row in lines]
                    desc.write_text("\n".join(lines) + "\n", encoding="utf-8")
                else:
                    desc.write_text("FROM " + str(gguf) + "\n", encoding="utf-8")
            environment = os.environ.copy()
            environment["OLLAMA_HOST"] = OLLAMA_URL.removeprefix("http://")
            environment["OLLAMA_MODELS"] = str(MODEL_DIR / "ollama")
            command(str(OLLAMA_EXE), "create", model.tag, "-f", str(desc),
                    cwd=OLLAMA_EXE.parent, env=environment)
        else:
            environment = os.environ.copy()
            environment["OLLAMA_HOST"] = OLLAMA_URL.removeprefix("http://")
            environment["OLLAMA_MODELS"] = str(MODEL_DIR / "ollama")
            if model_id == "hy18":
                template = Path(__file__).resolve().parent / "HyMT2_1_8B_fixed.Modelfile"
                if not template.is_file():
                    raise FileNotFoundError("Original hy18 Modelfile missing")
                command(str(OLLAMA_EXE), "create", model.tag, "-f", str(template),
                        cwd=OLLAMA_EXE.parent, env=environment)
            else:
                command(str(OLLAMA_EXE), "pull", model.tag,
                        cwd=OLLAMA_EXE.parent, env=environment)
        if model.tag.casefold() not in get_installed():
            raise RuntimeError("Ollama model not listed after installation: " + model.tag)
        print("[PASS] Model installed:", model_id, model.tag, flush=True)

def install_model(model_id: str) -> None:
    model = BY_ID[model_id]
    if model_id == "index9":
        # Fresh-install from the IndexTeam official GGUF with LFS SHA256 verification.
        # Existing verified Ollama tag is used without replacing its assets.
        if model.tag.casefold() in get_installed():
            print("[SKIP] Verified Index-Translate 9B already installed", flush=True)
            return
        repo_assets(model_id)
        engine()
        import_local(model_id)
        return
    if model.runner in ("nllb", "madlad"):
        if model.runner == "nllb":
            python_env()
        repo_assets(model_id)
        if model.runner == "madlad":
            llama()
    else:
        engine()
        import_local(model_id)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install isolated LzTranslate Ollama, llama.cpp and model assets"
    )
    parser.add_argument("--engine", action="store_true", help="Install official portable Ollama")
    parser.add_argument("--python", action="store_true", help="Install Python 3.12 model runtime")
    parser.add_argument("--llama", action="store_true", help="Fresh git clone/build CUDA llama.cpp")
    parser.add_argument("--model", choices=sorted(BY_ID), help="Download/import one chosen model")
    parser.add_argument("--fetch-weights", choices=sorted(MODEL_REPOS),
                        help=argparse.SUPPRESS)
    parser.add_argument("--check", action="store_true", help="Only inspect available tools")
    args = parser.parse_args()
    try:
        if args.check:
            print("[ENGINE]", OLLAMA_EXE, OLLAMA_EXE.is_file())
            print("[PYTHON]", (ENGINE_DIR/"venv"/"Scripts"/"python.exe").is_file())
            print("[LLAMA]", llama_server_exe())
            print("[MODELS]", MODEL_DIR)
            return 0
        if args.fetch_weights:
            fetch_weights(args.fetch_weights)
            return 0
        if not any((args.engine, args.python, args.llama, args.model)):
            parser.print_help()
            return 0
        if args.engine:
            engine()
        if args.python:
            python_env()
        if args.llama:
            llama()
        if args.model:
            install_model(args.model)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError,
            zipfile.BadZipFile, urllib.error.URLError) as error:
        print("[FAILED]", type(error).__name__ + ":", error, file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
