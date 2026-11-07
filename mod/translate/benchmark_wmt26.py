"""Independent 2026 WMT long-document EN->KO comparison on this PC.

This is separate from WMT24++ and uses the official 2026 WMT general MT
document corpus plus human-annotation-derived SYNTHETIC references. The
synthetic references are not direct human reference translations.

Commands:
  python benchmark_wmt26.py --prepare
  python benchmark_wmt26.py --model hy30
  python benchmark_wmt26.py --model index35q4
  python benchmark_wmt26.py --model tg27q4
  <eval-python> benchmark_wmt26.py --score

Output goes only to the configured application log directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.request

from model_catalog import APP

ROOT = APP / "logs" / "quality_20261106"
SOURCE = ROOT / "wmt26-genmt_official.jsonl"
SOURCE_SHA = "de286ce336f10367bedfe0611764b2569b538a9c57db20467fc8c1fa84838291"
REFERENCE = ROOT / "wmt26_synthetic_references_official.jsonl"
REFERENCE_SHA = "e40250b86dea6b090d126920ac2e6f0769cb8ca8466cd7676c686456a6a515b1"
SAMPLE = ROOT / "wmt26_long24.json"
HOST = "http://127.0.0.1:12545"
EVAL_MODEL = (APP / "models" / "comet" / "weights" /
              "models--Unbabel--wmt22-comet-da" / "snapshots" /
              "2760a223ac957f30acfb18c8aa649b01cf1d75f2" /
              "checkpoints" / "model.ckpt")
CANDIDATES = {
    "hy30": ("hf.co/tencent/Hy-MT2-30B-A3B-GGUF:Q4_K_M", "generic"),
    "index35q4": ("index35q4:quality", "index"),
    "index9": ("index-translate-9b-q8:ocr", "index"),
    "tg27q4": ("translategemma:27b", "translategemma"),
    "tg27q6": ("translategemma-27b-q6:quality", "translategemma"),
    "tg12q8": ("translategemma:12b-it-q8_0", "translategemma"),
}
QUOTAS = {"news": 7, "social": 8, "software": 4, "speech": 5}
KOREAN = re.compile(r"[\uac00-\ud7a3]")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare() -> dict:
    if digest(SOURCE) != SOURCE_SHA or digest(REFERENCE) != REFERENCE_SHA:
        raise RuntimeError("Original WMT26 source or human-derived reference hash changed")
    raw = [json.loads(x) for x in SOURCE.read_text(encoding="utf-8").splitlines()]
    by_id = {x["doc_id"]: x for x in raw if
             x["src_lang"] == "eng_Latn" and x["tgt_lang"] == "kor_Hang"}
    references = [
        x for x in (json.loads(t) for t in REFERENCE.read_text(encoding="utf-8").splitlines())
        if x["doc_id"] in by_id
    ]
    bins = {key: [] for key in QUOTAS}
    for ref in references:
        origin = by_id[ref["doc_id"]]
        if origin["domain"] not in bins:
            continue
        if origin["src_text"].strip() != ref["source"].strip():
            raise RuntimeError("WMT26 reference aligned to different original")
        if not ref["hypothesis"].strip():
            raise RuntimeError("WMT26 empty reference")
        bins[origin["domain"]].append({
            "id": origin["doc_id"], "domain": origin["domain"],
            "source": origin["src_text"], "reference": ref["hypothesis"],
            "source_chars": len(origin["src_text"]),
        })
    chosen = []
    for domain, n in QUOTAS.items():
        records = sorted(bins[domain], key=lambda d: d["id"])
        if len(records) < n:
            raise RuntimeError(f"Not enough WMT26 {domain} references")
        chosen += [records[int((i + .5) * len(records) / n)] for i in range(n)]
    if len(chosen) != 24 or len({x["id"] for x in chosen}) != 24:
        raise RuntimeError("WMT26 subset invalid")
    obj = {
        "source": "https://github.com/wmt-conference/wmt26-general-mt",
        "source_sha256": SOURCE_SHA,
        "human_assessed_synthetic_reference_sha256": REFERENCE_SHA,
        "kind": "2026 WMT General MT full-document source, derived synthetic reference",
        "selection": "7 news, 8 social, 4 software, 5 speech; deterministic equidistant IDs",
        "samples": chosen,
    }
    if SAMPLE.exists():
        if json.loads(SAMPLE.read_text(encoding="utf-8")) != obj:
            raise RuntimeError("Existing WMT26 frozen sample differs")
    else:
        with SAMPLE.open("x", encoding="utf-8", newline="\n") as out:
            json.dump(obj, out, ensure_ascii=False, indent=2)
            out.write("\n")
    print("[WMT26_SAMPLES]", len(chosen), "SHA256", digest(SAMPLE),
          "domain_lengths", {k: [x["source_chars"] for x in chosen if x["domain"] == k]
                              for k in QUOTAS}, flush=True)
    return obj


def infer(model: str, source: str) -> str:
    model_id, kind = CANDIDATES[model]
    if kind == "index":
        prompt = (
            "请将以下英语文档准确翻译为韩语，保留全部事实、数字、专有名词、"
            "段落结构和HTML标签，不要添加或删减信息。直接输出译文，不要解释。\n\n"
            + source)
    elif kind == "translategemma":
        prompt = (
            "You are a professional English (en) to Korean (ko) translator. "
            "Your goal is to accurately convey the meaning and nuances of the original English "
            "text while adhering to Korean grammar, vocabulary, and cultural sensitivities.\n"
            "Produce only the Korean translation, without any additional explanations "
            "or commentary. Please translate the following English text into Korean:\n\n"
            + source)
    else:
        prompt = ("Translate the following complete English document accurately into Korean. "
                  "Preserve every fact, number, named entity, paragraph break and HTML element. "
                  "Do not add or omit information. Output Korean translation only:\n\n" + source)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False, "think": False, "keep_alive": "4m",
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 4096},
    }
    req = urllib.request.Request(
        HOST + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=480) as response:
        reply = json.load(response)
    txt = str(reply.get("message", {}).get("content") or "").strip()
    if not reply.get("done") or reply.get("done_reason") == "length":
        raise RuntimeError("Model stopped before completing WMT26 document")
    if not KOREAN.search(txt):
        raise RuntimeError("Output is not Korean")
    return txt


def benchmark(model: str) -> None:
    obj = prepare()
    samples = obj["samples"]
    file = ROOT / f"wmt26_predictions_{model}.jsonl"
    if file.exists():
        data = [json.loads(x) for x in file.read_text(encoding="utf-8").splitlines()]
        if len(data) != len(samples):
            raise FileExistsError("Existing partial benchmark protected; do not overwrite")
        if any(x["id"] != y["id"] or y["status"] != "PASS"
               or y["sample_sha256"] != digest(SAMPLE)
               for x, y in zip(samples, data)):
            raise RuntimeError("Existing result does not match frozen sample")
        print("[EXISTING_COMPLETE]", file, digest(file), flush=True)
        return
    with urllib.request.urlopen(HOST + "/api/tags", timeout=10) as r:
        tags = {x["name"].casefold() for x in json.load(r)["models"]}
    if CANDIDATES[model][0].casefold() not in tags:
        raise RuntimeError("Model not installed in ISOLATED Ollama endpoint")
    with file.open("x", encoding="utf-8", newline="\n") as out:
        for i, sample in enumerate(samples):
            before = time.monotonic()
            korean = infer(model, sample["source"])
            entry = {
                "id": sample["id"], "model": model,
                "sample_sha256": digest(SAMPLE),
                "status": "PASS", "translation": korean,
                "seconds": round(time.monotonic() - before, 3),
            }
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out.flush()
            print("[WMT26_PASS]", model, i+1, "/", len(samples),
                  entry["seconds"], flush=True)
    print("[WMT26_COMPLETE]", model, "SHA256", digest(file), flush=True)


def score(models: list[str]) -> None:
    from comet import load_from_checkpoint
    import torch
    samples = prepare()["samples"]
    scores = {}
    for model in models:
        path = ROOT / f"wmt26_predictions_{model}.jsonl"
        if not path.is_file():
            print("[MISSING_MODEL]", model, flush=True)
            continue
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(samples) or any(
            a["id"] != b["id"] or b["status"] != "PASS"
            or b["sample_sha256"] != digest(SAMPLE)
            for a,b in zip(samples, rows)
        ):
            raise RuntimeError(f"Benchmark results invalid for {model}")
        scores[model] = rows
    if not scores:
        raise RuntimeError("No fully translated candidates")
    out = ROOT / "wmt26_long24_comet.json"
    if out.exists():
        raise FileExistsError("Existing WMT26 COMET result protected")
    torch.set_num_threads(12)
    estimator = load_from_checkpoint(str(EVAL_MODEL))
    estimates = {}
    for model, rows in scores.items():
        content = [
            {"src": sample["source"], "ref": sample["reference"], "mt": row["translation"]}
            for sample, row in zip(samples, rows)]
        raw = estimator.predict(content, batch_size=2, gpus=0)
        values = [float(x) for x in raw.scores]
        if len(values) != len(samples):
            raise RuntimeError("COMET evaluator returned wrong length")
        estimates[model] = {
            "mean": round(statistics.mean(values), 6),
            "document_scores": [
                {"id": s["id"], "score": round(v, 8)}
                for s,v in zip(samples, values)],
            "predictions_sha256": digest(ROOT / f"wmt26_predictions_{model}.jsonl"),
        }
        print("[WMT26_COMET]", model, estimates[model]["mean"], flush=True)
    obj = {
        "schema": "LzTranslate-WMT26-human-derived-synthetic-reference-24-docs",
        "benchmark_sha256": digest(SAMPLE),
        "evaluator_checkpoint_sha256": digest(EVAL_MODEL),
        "models": estimates,
        "limitations": "Human-annotation-derived synthetic references (not independent human direct reference); 24 fixed documents and English-to-Korean only.",
    }
    with out.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(obj, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print("[WMT26_REPORT]", out, digest(out), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--model", choices=tuple(CANDIDATES))
    group.add_argument("--score", action="store_true")
    ap.add_argument("--models", default="hy30,index35q4,tg27q4,index9",
                    help="Comma-separated models to evaluate")
    args = ap.parse_args()
    if args.prepare:
        prepare()
    elif args.model:
        benchmark(args.model)
    else:
        score([s.strip() for s in args.models.split(",") if s.strip()])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("[WMT26_ERROR]", type(e).__name__, str(e), file=sys.stderr, flush=True)
        raise
