"""Reference-based semantic translation evaluation using Unbabel WMT22 COMET-DA.

Independent local model; CPU-only inference in
the optional evaluation environment, model isolated under models/comet.
Uses original English source, official Google Korean reference and
a saved, SHA-verified hypothesis from each candidate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

from model_catalog import APP

ROOT = APP / "logs" / "quality_20261106"
CORPUS = ROOT / "official_wmt24pp_long_sample.json"
CORPUS_SHA = "8a4b82395eeb2bc488475f137d110e1351697c4fb49d32faf85003d2dc60b02f"
EVALUATOR = APP / "models" / "comet"
CHECKPOINT = (EVALUATOR / "weights" / "models--Unbabel--wmt22-comet-da"
              / "snapshots" / "2760a223ac957f30acfb18c8aa649b01cf1d75f2"
              / "checkpoints" / "model.ckpt")
PREDICTIONS = {
    "gemma4": "54294ca0cce3d9dfb5406e142825f75e42451cb75b86cb502e6dd82a1667e870",
    "index35q4": "76692f6932683f189d675432ce23e7ffe1e34f338479ec0d39181444947466a2",
    "index9": "6b9df6fcefa1ce9e4986dbb9b8ee4272b1e5060b1ec14c8e6632769587bc7507",
    "tg27q4": "5af3af145f2ff52d47caafd4586ee09a88eaee37afae478abfc98eccf0272a59",
    "hy7": "da6a0f8df5016d5fe0ea787f6514a871ab28ec70af35a258ff7b814470a91091",
    "hy30": "0babd995d82cbbbee8b838c7059113b0de77344968adee346d0d95d87c995907",
    "tg12q8": "e8a49e0cae22535bc46fe3e62e53e04e5ae1852982297a809f24a5857d43d21d",
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(16*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def verify(model_ids):
    assert CORPUS.is_file() and sha256(CORPUS) == CORPUS_SHA
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))["samples"]
    assert len(corpus) == 76
    data = {}
    for model in model_ids:
        path = ROOT / f"predictions_{model}.jsonl"
        if not path.is_file() or sha256(path) != PREDICTIONS[model]:
            raise RuntimeError(f"Saved translations changed: {model}")
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(corpus):
            raise RuntimeError(f"Wrong count: {model}")
        for original, pred in zip(corpus, rows):
            if original["id"] != pred["id"] or pred["status"] != "PASS":
                raise RuntimeError(f"Unaligned entry in {model}")
        data[model] = rows
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"Local COMET checkpoint missing: {CHECKPOINT}")
    return corpus, data

def score(model_ids, batch_size: int):
    corpus, predictions = verify(model_ids)
    for name in model_ids:
        if (ROOT / f"comet_{name}.json").exists():
            raise FileExistsError(f"Existing semantic score, no overwrite: {name}")
    os.environ["HF_HOME"] = str(EVALUATOR / "hf-home")
    os.environ["XDG_CACHE_HOME"] = str(EVALUATOR / "xdg-cache")
    os.environ["TORCH_HOME"] = str(EVALUATOR / "torch-home")
    import torch
    from comet import load_from_checkpoint
    torch.set_num_threads(12)
    print("[COMET_LOAD]", CHECKPOINT, flush=True)
    stamp = time.monotonic()
    critic = load_from_checkpoint(str(CHECKPOINT))
    print("[COMET_READY]", round(time.monotonic()-stamp, 2), "sec", flush=True)
    ckpt_hash = sha256(CHECKPOINT)
    items = []
    for name in model_ids:
        for origin, pred in zip(corpus, predictions[name]):
            items.append({
                "src": origin["source"], "mt": pred["translation"],
                "ref": origin["reference"],
            })
    print("[COMET_EVAL]", len(items), "pairs", flush=True)
    result = critic.predict(items, batch_size=batch_size, gpus=0)
    values = [float(value) for value in result.scores]
    if len(values) != len(items):
        raise RuntimeError("COMET returned inconsistent number of judgments")
    for idx, name in enumerate(model_ids):
        segment_values = values[idx*len(corpus):(idx+1)*len(corpus)]
        details = [
            {"id": item["id"], "domain": item["domain"], "kind": item["kind"],
             "score": round(value, 8)}
            for item, value in zip(corpus, segment_values)
        ]
        record = {
            "schema": "lztranslate_comet_wmt22_da_v1",
            "metric": "Unbabel/wmt22-comet-da reference-based (higher is better)",
            "metric_checkpoint_sha256": ckpt_hash,
            "benchmark_sample_sha256": CORPUS_SHA,
            "translation_sha256": PREDICTIONS[name],
            "model": name,
            "score_mean": round(statistics.mean(segment_values), 6),
            "score_by_kind": {
                kind: round(statistics.mean(value for item, value in zip(corpus, segment_values)
                                            if item["kind"] == kind), 6)
                for kind in ("segment", "document")
            },
            "count": len(segment_values),
            "scores": details,
        }
        path = ROOT / f"comet_{name}.json"
        with path.open("x", encoding="utf-8") as out:
            json.dump(record, out, ensure_ascii=False, indent=2)
            out.write("\n")
        print("[COMET_PASS]", name, record["score_mean"],
              "doc", record["score_by_kind"]["document"], "sha256", sha256(path), flush=True)

def main() -> int:
    ap = argparse.ArgumentParser(description="Local COMET-DA semantic evaluation on official WMT24++")
    ap.add_argument("--models", default="index9,gemma4,tg27q4,hy7",
                    help="Comma separated IDs from saved raw translations")
    ap.add_argument("--batch-size", type=int, default=2)
    opts = ap.parse_args()
    ids = list(dict.fromkeys(s.strip() for s in opts.models.split(",") if s.strip()))
    if not ids or any(k not in PREDICTIONS for k in ids):
        ap.error("Unsupported model ID")
    if not 1 <= opts.batch_size <= 8:
        ap.error("Batch size must be 1..8")
    try:
        score(ids, opts.batch_size)
        return 0
    except Exception as exc:
        print("[COMET_FAIL]", type(exc).__name__, str(exc), file=sys.stderr, flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
