"""Index-Translate subtitle runner with verified cue/timestamp preservation."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import time
from compare_srt import read_srt, validate_alignment
from translate import render
from index_backend import detect_language, translate


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args) -> int:
    source = args.source.resolve(strict=True)
    destination = args.output.resolve()
    if not source.is_file() or source.suffix.lower() != ".srt":
        raise ValueError("A valid SRT input is required")
    if destination == source or destination.exists():
        raise FileExistsError("Original/existing output protected: " + str(destination))
    if not destination.parent.is_dir():
        raise NotADirectoryError(destination.parent)
    if not 20 <= args.width <= 80:
        raise ValueError("Width must be in 20..80")
    original = read_srt(source)
    original_hash = sha256(source)
    lang, code = detect_language(" ".join(row[2] for row in original), args.source_language)
    print("[INDEX_START]", args.model, "language", lang, "cues", len(original), flush=True)
    predictions = {}
    for ident, timestamp, text in original:
        result = translate(text, args.model, code)
        predictions[ident] = " ".join(result.splitlines()).strip()
        print("[INDEX_PROGRESS]", len(predictions), "/", len(original), flush=True)
    if sha256(source) != original_hash:
        raise RuntimeError("Original SRT changed during translation")
    translated = render(original, predictions, args.width)
    with destination.open("x", encoding="utf-8", newline="") as f:
        f.write(translated)
    validate_alignment(original, read_srt(destination))
    print("[INDEX_PASS]", destination, "sha256", sha256(destination), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Official Index-Translate SRT translation to Korean")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-language", default="auto")
    parser.add_argument("--model", default="index-translate-9b-q8:ocr")
    parser.add_argument("--width", type=int, default=42)
    args = parser.parse_args()
    try:
        return run(args)
    except (RuntimeError, ValueError, OSError, UnicodeError) as err:
        print("[INDEX_FAIL]", type(err).__name__, str(err), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
