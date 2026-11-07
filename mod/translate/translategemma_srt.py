"""Dedicated TranslateGemma SRT runner, strict source/timestamp conservation.

Each subtitle is translated with the official model prompt and a detected
source language. Use a full document for source-language detection so that
short English/German subtitle fragments are not identified independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from compare_srt import read_srt, validate_alignment
from translate import render
from translategemma_backend import language, translate


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def run(args) -> int:
    src = args.source.resolve(strict=True)
    if not src.is_file() or src.suffix.casefold() != ".srt":
        raise ValueError("Input must be an existing UTF-8 .srt file")
    dst = args.output.resolve()
    if src == dst or dst.exists():
        raise FileExistsError(f"Input/output protection: {dst}")
    if not dst.parent.is_dir():
        raise ValueError(f"Destination directory missing: {dst.parent}")
    if not 20 <= args.width <= 80:
        raise ValueError("--width must be 20..80")
    cues = read_srt(src)
    original_hash = digest(src)
    detected, code = language(" ".join(x[2] for x in cues), args.source_language)
    start = time.monotonic()
    print("[TG_SOURCE]", detected, code, "cues", len(cues), "model", args.model, flush=True)
    translated = {}
    for n, stamp, text in cues:
        output = translate(text, args.model, detected, code)
        if "\n" in output:
            output = " ".join(part.strip() for part in output.splitlines() if part.strip())
        translated[n] = output
        print("[TG_PROGRESS]", len(translated), "/", len(cues),
              round(time.monotonic()-start, 2), flush=True)
    if digest(src) != original_hash:
        raise RuntimeError("Source modified during translation")
    final = render(cues, translated, args.width)
    with dst.open("x", encoding="utf-8", newline="") as writer:
        writer.write(final)
    validate_alignment(cues, read_srt(dst))
    print("[TG_PASS]", dst, "SHA256", digest(dst), "source_unchanged", digest(src)==original_hash, flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="TranslateGemma official-prompt Korean SRT runner")
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source-language", default="auto")
    ap.add_argument("--width", type=int, default=42)
    args = ap.parse_args()
    try:
        return run(args)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print("[TG_FAILED]", type(exc).__name__, str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
