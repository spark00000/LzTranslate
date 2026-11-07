"""Resumable and SHA-256-verified TranslateGemma 27B Q6_K GGUF downloader.

The source is the publicly verified mradermacher quantization of the official
Google TranslateGemma 27B model, hosted on Hugging Face. The exact full-file
SHA256 is taken from the Hugging Face model tree LFS content oid. All target
files and checkpoints stay in the configured application directory, and existing files
are protected. Download chunks use HTTP Range and local SHA256 checks.

Source: https://huggingface.co/mradermacher/translategemma-27b-it-GGUF
Official Google source: https://huggingface.co/google/translategemma-27b-it
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import threading
import time
import urllib.request

from model_catalog import APP
ROOT = APP / "models" / "translategemma27q6_hf"
FILENAME = "translategemma-27b-it.Q6_K.gguf"
SIZE = 22_166_989_920
SHA256 = "77734a1c6ef99aa29c92bc42dc0b859ecb37e3606ae5a4e69facde21419391c8"
URL = "https://huggingface.co/mradermacher/translategemma-27b-it-GGUF/resolve/main/" + FILENAME
CHUNK = 64 * 1024 * 1024
PART = ROOT / (FILENAME + ".part")
FINAL = ROOT / FILENAME
STATE = ROOT / "download_checkpoint.json"
STATE_TMP = ROOT / "download_checkpoint.json.tmp"
RECEIPT = ROOT / "verified.json"
NUM_CHUNKS = math.ceil(SIZE / CHUNK)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(state: dict) -> None:
    if STATE_TMP.exists():
        raise RuntimeError("Unresolved previous checkpoint temporary file; stop")
    with STATE_TMP.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(state, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(STATE_TMP, STATE)


def validate_done(state: dict) -> None:
    for item, previous_hash in state["completed"].items():
        idx = int(item)
        if not 0 <= idx < NUM_CHUNKS:
            raise RuntimeError("Invalid checkpoint range")
        start = idx * CHUNK
        size = min(CHUNK, SIZE - start)
        h = hashlib.sha256()
        with PART.open("rb") as f:
            f.seek(start)
            remaining = size
            while remaining:
                block = f.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise RuntimeError("Missing bytes in completed range")
                h.update(block)
                remaining -= len(block)
        if h.hexdigest() != previous_hash:
            raise RuntimeError(f"Previously completed range {idx} failed SHA-256; stop")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=360,
                    help="Maximum seconds to issue new range requests (default 360)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not 10 <= args.seconds <= 1200 or not 1 <= args.workers <= 12:
        ap.error("Seconds must be 10..1200 and workers 1..12")
    if FINAL.exists():
        if FINAL.stat().st_size != SIZE or sha256(FINAL) != SHA256:
            raise RuntimeError("Existing final file failed official SHA-256; refusing overwrite")
        print("[ALREADY_VERIFIED]", FINAL, SHA256, flush=True)
        return 0

    if not ROOT.exists():
        ROOT.mkdir(parents=True, exist_ok=False)
    if STATE_TMP.exists():
        raise RuntimeError("Incomplete checkpoint transaction; stop and inspect")
    expected = {"schema": 1, "url": URL, "size": SIZE,
                "sha256": SHA256, "chunk_bytes": CHUNK, "completed": {}}
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        for field in ("schema", "url", "size", "sha256", "chunk_bytes"):
            if state.get(field) != expected[field]:
                raise RuntimeError(f"Checkpoint source mismatch: {field}")
        if not PART.is_file() or PART.stat().st_size != SIZE:
            raise RuntimeError("Existing partial file invalid; stop")
        print("[RESUME_VERIFY]", len(state["completed"]), "ranges", flush=True)
        validate_done(state)
    else:
        if PART.exists() or RECEIPT.exists():
            raise FileExistsError("Existing target without valid checkpoint; stop")
        with PART.open("xb") as f:
            f.truncate(SIZE)
        state = expected
        save(state)

    todo = queue.Queue()
    for i in range(NUM_CHUNKS):
        if str(i) not in state["completed"]:
            todo.put(i)
    lock = threading.Lock()
    stop = threading.Event()
    errors: list[str] = []
    deadline = time.monotonic() + args.seconds
    began = time.monotonic()
    last_report = [0.0]

    def worker() -> None:
        while not stop.is_set() and time.monotonic() < deadline:
            try:
                idx = todo.get_nowait()
            except queue.Empty:
                return
            start = idx * CHUNK
            end = min(SIZE, start + CHUNK) - 1
            expected_range = f"bytes {start}-{end}/{SIZE}"
            for attempt in range(4):
                if stop.is_set():
                    return
                try:
                    req = urllib.request.Request(
                        URL, method="GET",
                        headers={
                            "Range": f"bytes={start}-{end}",
                            "Accept-Encoding": "identity",
                            "User-Agent": "LzTranslate-official-weights/1.3",
                        },
                    )
                    checksum = hashlib.sha256()
                    got = 0
                    with urllib.request.urlopen(req, timeout=60) as r:
                        content_range = r.headers.get("Content-Range")
                        if r.status != 206 or content_range != expected_range:
                            raise RuntimeError(
                                f"Expected 206 {expected_range}; got {r.status} {content_range}")
                        with PART.open("r+b", buffering=0) as out:
                            out.seek(start)
                            while got < end - start + 1:
                                data = r.read(min(1024 * 1024, end - start + 1 - got))
                                if not data:
                                    raise IOError("Server ended range prematurely")
                                out.write(data)
                                checksum.update(data)
                                got += len(data)
                            out.flush()
                            os.fsync(out.fileno())
                    if got != end - start + 1:
                        raise RuntimeError("Incomplete range length")
                    with lock:
                        state["completed"][str(idx)] = checksum.hexdigest()
                        save(state)
                        finished = len(state["completed"])
                        now = time.monotonic()
                        if now - last_report[0] > 15 or finished == NUM_CHUNKS:
                            progress = sum(
                                min(CHUNK, SIZE - int(k)*CHUNK)
                                for k in state["completed"])
                            print("[PROGRESS]", finished, "/", NUM_CHUNKS,
                                  "bytes", progress, "/", SIZE,
                                  "percent", round(progress * 100 / SIZE, 2),
                                  "elapsed_sec", round(now - began, 1), flush=True)
                            last_report[0] = now
                    break
                except Exception as e:
                    if attempt == 3:
                        with lock:
                            errors.append(
                                f"Range {idx} failed 4 attempts: {type(e).__name__}: {e}")
                        stop.set()
                        return
                    time.sleep(min(2 ** attempt, 5))
            todo.task_done()

    print("[START]", URL, "bytes", SIZE, "expected_sha256", SHA256,
          "workers", args.workers, "remaining_chunks", todo.qsize(), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = [executor.submit(worker) for _ in range(args.workers)]
        for task in tasks:
            task.result()

    if errors:
        raise RuntimeError("; ".join(errors))
    if len(state["completed"]) != NUM_CHUNKS:
        print("[INCOMPLETE_SAFE_TO_RESUME]", len(state["completed"]),
              "/", NUM_CHUNKS, "verified ranges", flush=True)
        return 0

    print("[VERIFY_ALL_BYTES]", SIZE, "bytes", flush=True)
    actual_sha = sha256(PART)
    if actual_sha != SHA256:
        raise RuntimeError(f"Whole file SHA-256 mismatch {actual_sha} != {SHA256}")
    if FINAL.exists() or RECEIPT.exists():
        raise FileExistsError("Final output collision")
    os.replace(PART, FINAL)
    receipt = {
        "official_sha256": SHA256,
        "actual_sha256": actual_sha,
        "bytes": SIZE,
        "source": URL,
        "official_manifest":
            "https://huggingface.co/api/models/mradermacher/translategemma-27b-it-GGUF/tree/main?recursive=true",
        "verified_full_file": True,
    }
    with RECEIPT.open("x", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print("[OFFICIAL_MODEL_VERIFIED]", FINAL, "sha256", actual_sha, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
