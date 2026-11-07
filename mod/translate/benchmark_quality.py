"""Reproducible WMT24++ Korean translation quality evaluation.

Runs only local endpoints. Raw references/model outputs remain in
the configured application log directory. WMT24++ official source and checksum are fixed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

from model_catalog import APP

RUN = APP / "logs" / "quality_20261106"
SOURCE = APP / "logs" / "wmt24pp_en-ko_KR.jsonl"
SOURCE_SHA256 = "4520124f3769cd711aab0c939aeb849a225dce51b761eeb419f46e555b02d7e5"
DATA_URL = "https://huggingface.co/datasets/google/wmt24pp/resolve/main/en-ko_KR.jsonl"
SAMPLE = RUN / "official_wmt24pp_long_sample.json"
KOREAN = re.compile(r"[\uac00-\ud7a3]")
DOMAINS = ("news", "social", "speech", "literary")
LOCAL_API = os.environ.get("LZTRANSLATE_OLLAMA_URL", "http://127.0.0.1:12545").rstrip("/")
GLOBAL_API = os.environ.get("LZTRANSLATE_GLOBAL_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
MODELS = {
    "tg4q8": ("translategemma:4b-it-q8_0", "local", "translategemma"),
    "tg12q8": ("translategemma:12b-it-q8_0", "local", "translategemma"),
    "tg27q4": ("translategemma:27b", "local", "translategemma"),
    "tg27q6": ("translategemma-27b-q6:quality", "local", "translategemma"),
    "tg27q5": ("hf.co/mradermacher/translategemma-27b-it-GGUF:Q5_K_M", "local", "translategemma"),
    "tg27q8": ("translategemma:27b-it-q8_0", "local", "translategemma"),
    "gemma4": ("gemma4:26b", "global", "generic"),
    "index35q4": ("index35q4:quality", "local", "index"),
    "index9": ("index-translate-9b-q8:ocr", "global", "index"),
    "hy7": ("hf.co/tencent/Hy-MT2-7B-GGUF:Q4_K_M", "global", "generic"),
    "hy30": ("hf.co/tencent/Hy-MT2-30B-A3B-GGUF:Q4_K_M", "global", "generic"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def prepare() -> dict:
    if not SOURCE.is_file() or sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("Official WMT24++ file is missing or its SHA256 changed")
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    # Human post-edited references, with canary and bad-source records excluded.
    selected = []
    for d in DOMAINS:
        good = sorted(
            (r for r in rows if r["domain"] == d and not r["is_bad_source"]
             and 250 <= len(r["source"]) <= 1250 and len(r["target"]) >= 30),
            key=lambda x: int(x["segment_id"]),
        )
        if len(good) < 16:
            raise RuntimeError(f"Too few long WMT24++ records in {d}: {len(good)}")
        for ix in [int((i + .5) * len(good)/16) for i in range(16)]:
            row = good[ix]
            selected.append({
                "id": f"{d}:{row['segment_id']}", "kind": "segment", "domain": d,
                "source": row["source"], "reference": row["target"],
                "official_segments": [int(row["segment_id"])],
            })

    # Genuine contiguous source paragraphs from the SAME official document.
    docs = defaultdict(list)
    for row in rows:
        if (row["domain"] in ("news", "literary") and not row["is_bad_source"]
            and row["source"].strip() and row["target"].strip()):
            docs[(row["domain"], str(row["document_id"]))].append(row)
    for domain in ("news", "literary"):
        passages = []
        for (kind, doc_id), seq in sorted(docs.items()):
            if kind != domain:
                continue
            seq.sort(key=lambda x: int(x["segment_id"]))
            for j in range(0, len(seq) - 2, 3):
                triplet = seq[j:j+3]
                txt = "\n\n".join(str(a["source"]) for a in triplet)
                ref = "\n\n".join(str(a["target"]) for a in triplet)
                if 750 <= len(txt) <= 2400 and len(ref) >= 250:
                    passages.append({
                        "id": f"document:{domain}:{doc_id}:{triplet[0]['segment_id']}",
                        "kind": "document", "domain": domain,
                        "source": txt, "reference": ref, "document_id": doc_id,
                        "official_segments": [int(a["segment_id"]) for a in triplet],
                    })
        if len(passages) < 6:
            raise RuntimeError(f"Too few document-level official passages: {domain}: {len(passages)}")
        # Deterministic spread across the source documents.
        passages.sort(key=lambda r: r["id"])
        selected.extend(passages[int((i + .5)*len(passages)/6)] for i in range(6))

    ids = [x["id"] for x in selected]
    if len(set(ids)) != len(ids) or len(selected) != 76:
        raise RuntimeError("Incorrect official sample count/duplicate")
    corpus = {
        "version": 1,
        "source_url": DATA_URL, "official_source_sha256": SOURCE_SHA256,
        "selection": "16 >=250-char records per domain + 6 3-paragraph passages per news/literary domain; bad sources/canary excluded; deterministic equidistant sampling",
        "samples": selected,
    }
    RUN.mkdir(parents=True, exist_ok=True)
    if SAMPLE.exists():
        if json.loads(SAMPLE.read_text(encoding="utf-8")) != corpus:
            raise RuntimeError("Existing frozen corpus differs from the verified source")
    else:
        SAMPLE.write_text(json.dumps(corpus, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("[SAMPLE]", len(selected), "SHA256", sha256(SAMPLE), flush=True)
    return corpus


def request(api: str, path: str, payload: dict, timeout: int = 360) -> dict:
    r = urllib.request.Request(
        api + path, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=timeout) as stream:
        return json.load(stream)


def official_prompt(source: str) -> str:
    # The official Ollama TranslateGemma library prompt, including TWO newlines.
    return (
        "You are a professional English (en) to Korean (ko) translator. "
        "Your goal is to accurately convey the meaning and nuances of the original "
        "English text while adhering to Korean grammar, vocabulary, and cultural sensitivities.\n"
        "Produce only the Korean translation, without any additional explanations "
        "or commentary. Please translate the following English text into Korean:\n\n"
        + source
    )


def translate(model: str, source: str) -> tuple[str, dict]:
    tag, runtime, style = MODELS[model]
    api = LOCAL_API if runtime == "local" else GLOBAL_API
    if style == "translategemma":
        prompt = official_prompt(source)
    elif style == "index":
        prompt = "请将以下英文文本翻译为韩文，直接输出翻译结果，不要进行任何解释。\n\n" + source
    else:
        prompt = ("Translate the following English text into natural Korean. "
                  "Preserve all facts, numbers, and names. Output only the Korean translation, "
                  "without explanation.\nEnglish text:\n" + source)
    payload = {
        "model": tag, "messages": [{"role": "user", "content": prompt}],
        "stream": False, "think": False, "keep_alive": "3m",
        "options": {"temperature": 0, "num_ctx": 4096,
                    "num_predict": min(1600, max(600, int(len(source)*2.0)))},
    }
    before = time.monotonic()
    answer = request(api, "/api/chat", payload, timeout=600)
    seconds = round(time.monotonic()-before, 3)
    output = str(answer.get("message", {}).get("content") or "").strip()
    output = re.sub(r"(?s)^<think>.*?</think>\s*", "", output)
    if not answer.get("done") or answer.get("done_reason") == "length":
        raise RuntimeError("Model response incomplete or truncated")
    if not KOREAN.search(output):
        raise RuntimeError("No Korean characters in model translation")
    return output, {
        "elapsed_sec": seconds,
        "load_sec": round(answer.get("load_duration", 0)/1e9, 3),
        "eval_tokens": answer.get("eval_count", 0),
        "prompt_tokens": answer.get("prompt_eval_count", 0),
        "eval_sec": round(answer.get("eval_duration", 0)/1e9, 3),
    }


def path_model(model: str) -> Path:
    return RUN / f"predictions_{model}.jsonl"


def validate_predictions(model: str, samples: list[dict], require_complete=True):
    output = path_model(model)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != len(samples) if require_complete else len(rows) > len(samples)):
        raise RuntimeError(f"Incomplete or excessive model rows: {model} {len(rows)}/{len(samples)}")
    for reference, row in zip(samples, rows):
        if (row.get("id") != reference["id"]
            or row.get("model") != model
            or row.get("sample_sha256") != sha256(SAMPLE)
            or row.get("status") != "PASS" or not KOREAN.search(row.get("translation", ""))):
            raise RuntimeError(f"Invalid benchmark output for {model}, id {reference['id']}")
    return rows


def benchmark(model: str, resume_sha256: str | None):
    data = prepare()
    samples = data["samples"]
    target = path_model(model)
    previous = []
    if target.exists():
        if not resume_sha256:
            previous = validate_predictions(model, samples)
            print("[ALREADY_COMPLETE]", model, len(previous), sha256(target))
            return
        if sha256(target).lower() != resume_sha256.lower():
            raise RuntimeError("Resume SHA256 does not match saved prediction file")
        previous = validate_predictions(model, samples, require_complete=False)
        if not 0 < len(previous) < len(samples):
            raise RuntimeError("Only incomplete files may be resumed")
    elif resume_sha256:
        raise RuntimeError("Cannot resume: no existing prediction file")
    tag, runtime, _ = MODELS[model]
    api = LOCAL_API if runtime == "local" else GLOBAL_API
    with urllib.request.urlopen(api+"/api/tags", timeout=8) as response:
        tags = {x["name"].lower() for x in json.load(response)["models"]}
    if tag.lower() not in tags:
        raise RuntimeError(f"Model is not installed on endpoint {api}: {tag}")
    print("[START]", model, tag, "rows", len(samples), "preserving", len(previous), flush=True)
    with target.open("a" if previous else "x", encoding="utf-8") as f:
        for i, item in enumerate(samples[len(previous):], start=len(previous)):
            try:
                text, stats = translate(model, item["source"])
                row = {
                    "id": item["id"], "model": model,
                    "sample_sha256": sha256(SAMPLE),
                    "status": "PASS", "translation": text, "stats": stats,
                }
                f.write(json.dumps(row, ensure_ascii=False)+"\n")
                f.flush()
                print("[PASS]", model, i+1, "/", len(samples),
                      round(stats["elapsed_sec"], 2), "sec", flush=True)
            except Exception as exc:
                print("[FAIL]", model, item["id"], str(exc), flush=True)
                raise
    finished = validate_predictions(model, samples)
    print("[BENCH_COMPLETE]", model, len(finished), "SHA256", sha256(target), flush=True)


def compare(model_ids: list[str]):
    from sacrebleu.metrics import CHRF, BLEU
    data = prepare()
    rows = data["samples"]
    metric = CHRF(word_order=2, char_order=6, beta=2)
    bleu = BLEU(effective_order=True)
    results = []
    for name in model_ids:
        path = path_model(name)
        if not path.is_file():
            print("[SKIP_MISSING]", name, flush=True)
            continue
        pred = validate_predictions(name, rows)
        scores = {}
        for kind in ("segment", "document"):
            chosen = [(a, b) for a, b in zip(rows, pred) if a["kind"] == kind]
            hyp = [b["translation"] for a, b in chosen]
            ref = [a["reference"] for a, b in chosen]
            scores[kind] = {
                "n": len(chosen),
                "chrfpp": round(metric.corpus_score(hyp, [ref]).score, 3),
                "bleu": round(bleu.corpus_score(hyp, [ref]).score, 3),
            }
        hyp = [x["translation"] for x in pred]
        ref = [x["reference"] for x in rows]
        results.append({
            "model": name,
            "chrFpp": round(metric.corpus_score(hyp, [ref]).score, 3),
            "BLEU": round(bleu.corpus_score(hyp, [ref]).score, 3),
            "kind": scores,
            "seconds_mean": round(statistics.mean(x["stats"]["elapsed_sec"] for x in pred), 2),
            "translations_sha256": sha256(path),
        })
    results.sort(key=lambda x:(-x["chrFpp"],-x["kind"]["document"]["chrfpp"]))
    if not results:
        raise RuntimeError("No complete predictions")
    out = RUN / "quality_comparison.json"
    svg = RUN / "quality_comparison.svg"
    if out.exists() or svg.exists():
        raise FileExistsError("Previous comparison exists, refusing overwrite")
    evidence = {
        "dataset_url": DATA_URL, "source_sha256": SOURCE_SHA256,
        "sample_sha256": sha256(SAMPLE), "count": len(rows),
        "metric": "sacrebleu chrF++ (word_order=2, beta=2), BLEU; reference similarity only",
        "caution": "Not a human MQM evaluation; WMT24++ may be training-contaminated; results depend on quantization, prompts, source domain.",
        "results": results,
    }
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    width=940;h=130+45*len(results)
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="25" y="38" font-family="Segoe UI,Arial" font-size="22">WMT24++ long-form EN-KO quality: chrF++</text>',
           '<text x="25" y="60" font-family="Segoe UI,Arial" font-size="12">64 long segments + 12 real multi-paragraph passages; model-generated Korean vs official reference</text>']
    maximum=max(55,max(x["chrFpp"] for x in results)+5)
    for i,x in enumerate(results):
        y=82+i*45;v=x["chrFpp"];length=int(v/maximum*575)
        lines.append(f'<text x="24" y="{y+19}" font-family="Segoe UI,Arial" font-size="14">{i+1}. {x["model"]}</text>')
        lines.append(f'<rect x="205" y="{y}" width="575" height="26" rx="4" fill="#e8ecf0"/>')
        lines.append(f'<rect x="205" y="{y}" width="{length}" height="26" rx="4" fill="{"#247b79" if i==0 else "#435b77"}"/>')
        lines.append(f'<text x="793" y="{y+19}" font-family="Segoe UI,Arial" font-size="15">{v:.2f}</text>')
    lines.append("</svg>")
    svg.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("[RESULT]",out,"SHA256",sha256(out),flush=True)
    print("[CHART]",svg,"SHA256",sha256(svg),flush=True)
    for x in results:print("[RANK]",x["model"],x["chrFpp"],x["BLEU"],"document",x["kind"]["document"]["chrfpp"],flush=True)


def main():
    ap=argparse.ArgumentParser(description="Local-only, long-form official WMT24++ English-to-Korean evaluation")
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--model", choices=sorted(MODELS))
    group.add_argument("--report", action="store_true")
    group.add_argument("--list", action="store_true")
    ap.add_argument("--resume-from-sha256", help="explicit content hash for safely resuming an interrupted file")
    args=ap.parse_args()
    try:
        if args.list:
            for key, values in MODELS.items():print(key,*values)
        elif args.prepare:
            prepare()
        elif args.report:
            compare(list(MODELS))
        else:
            benchmark(args.model,args.resume_from_sha256)
        return 0
    except Exception as exc:
        print("[ERROR]",type(exc).__name__,str(exc),file=sys.stderr,flush=True)
        return 1


if __name__=="__main__":
    sys.exit(main())
