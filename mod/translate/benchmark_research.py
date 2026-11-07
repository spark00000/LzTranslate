"""WMT24++ EN->KO reproducible local-model research benchmark.

Official Google reference: https://huggingface.co/datasets/google/wmt24pp
Artifact root: configured application log directory
Metric: sacrebleu chrF++ (character ngrams + word bigrams), not accuracy.

Only the local Ollama HTTP API is used for inference; no model store or
process owned by another project is changed. Run one model per invocation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

from model_catalog import APP, BY_ID, MODEL_DIR

OFFICIAL_URL = "https://huggingface.co/datasets/google/wmt24pp/resolve/main/en-ko_KR.jsonl"
DATASET_SHA256 = "4520124f3769cd711aab0c939aeb849a225dce51b761eeb419f46e555b02d7e5"
ROOT = APP / "logs"
SOURCE = ROOT / "wmt24pp_en-ko_KR.jsonl"
SAMPLE = ROOT / "wmt24pp_enko_sample_20261105.json"
PREFIX = "wmt24pp_enko_20261105"
API = os.environ.get("LZTRANSLATE_BENCH_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DOMAINS = ("news", "social", "speech", "literary")
LANG = "English"
KOREAN = re.compile(r"[\uac00-\ud7a3]")
SPECS = {
    "mi4":  ("hf.co/mradermacher/MiLMMT-46-4B-v1.0-GGUF:Q8_0", "xiaomi"),
    "mi12": ("hf.co/mradermacher/MiLMMT-46-12B-v1.0-GGUF:Q5_K_M", "xiaomi"),
    "hy18": ("hy-mt2-1.8b-fixed:bench", "generic"),
    "hy7":  ("hf.co/tencent/Hy-MT2-7B-GGUF:Q4_K_M", "generic"),
    "hy30": ("hf.co/tencent/Hy-MT2-30B-A3B-GGUF:Q4_K_M", "generic"),
    "tg27": ("translategemma:27b", "generic"),
    "gemma4": ("gemma4:26b", "generic"),
    "qwen35": ("qwen3.5:27b", "generic"),
    "tower9": ("hf.co/mradermacher/Tower-Plus-9B-GGUF:Q5_K_M", "tower"),
    "gemmax9": ("hf.co/mradermacher/GemmaX2-28-9B-v0.2-GGUF:Q5_K_M", "gemmax"),
    "chimera7": ("hf.co/mradermacher/Hunyuan-MT-Chimera-7B-GGUF:Q5_K_M", "chimera"),
    "index9": ("index-translate-9b-q8:ocr", "index"),
    "nllb33": (str(MODEL_DIR / "nllb33"), "nllb"),
    "madlad10": (str(MODEL_DIR / "madlad10"), "madlad"),
}
DEFAULT_PEERS = ("mi12", "hy7")

def digest(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            hash_.update(part)
    return hash_.hexdigest()

def select_dataset(per_domain: int) -> list[dict]:
    if digest(SOURCE) != DATASET_SHA256:
        raise RuntimeError(f"Official dataset SHA256 differs: {SOURCE}")
    by_domain = {domain: [] for domain in DOMAINS}
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        domain = row.get("domain")
        text = str(row.get("source", ""))
        translation = str(row.get("target", ""))
        if (domain not in by_domain or bool(row.get("is_bad_source"))
            or not 40 <= len(text) <= 240 or not 8 <= len(translation) <= 360):
            continue
        by_domain[domain].append({
            "id": int(row["segment_id"]),
            "domain": domain,
            "source": text,
            "reference": translation,
            "document": str(row.get("document_id", "")),
        })
    chosen = []
    for domain in DOMAINS:
        pool = by_domain[domain]
        pool.sort(key=lambda x: x["id"])
        if len(pool) < per_domain:
            raise ValueError(f"Insufficient {domain} samples ({len(pool)})")
        # Equidistant, deterministic IDs; never pick model-dependent sentences.
        idxs = [int((i + 0.5) * len(pool) / per_domain)
                for i in range(per_domain)]
        chosen.extend(pool[i] for i in idxs)
    return chosen

def sample_data(per_domain: int) -> dict:
    expected = {
        "schema": "lztranslate_wmt24pp_sample_v1",
        "dataset_url": OFFICIAL_URL,
        "dataset_sha256": DATASET_SHA256,
        "scope": "WMT24++ en-ko_KR; 4 domains; exclude canary, marked bad sources, extreme lengths",
        "per_domain": per_domain,
        "items": select_dataset(per_domain),
    }
    if SAMPLE.is_file():
        existing = json.loads(SAMPLE.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(f"Locked evaluation set differs. Refusing overwrite: {SAMPLE}")
    else:
        with SAMPLE.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    return expected

def post(path: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def prompt_for(model_id: str, source: str, peers: list[str]) -> tuple[str, bool]:
    runner = SPECS[model_id][1]
    if runner in ("xiaomi", "gemmax"):
        return f"Translate this from English to Korean:\nEnglish: {source}\nKorean:", True
    if runner == "index":
        # Matches official Index-Translate single-user prompt format.
        return ("请将以下英文文本翻译为韩文，直接输出翻译结果，不要进行任何解释。\n\n" + source, False)
    if runner == "chimera":
        if len(peers) < 2:
            raise ValueError("Chimera requires >= 2 independently generated peers")
        triple = "###"
        return (
            f"Analyze the following multiple Korean translations of the English segment and "
            f"generate a single refined Korean translation. Output only the refined translation.\n"
            f"English: {source}\n"
            + "\n".join(f"Candidate {i}: {s}" for i, s in enumerate(peers, 1))
        ), False
    if runner == "tower":
        return ("Translate the following text from English into natural Korean. "
                "Preserve meaning, names, and numbers. Output only the translation.\n\n" + source), False
    return (
        "Translate the following English text into natural Korean. "
        "Preserve all facts, numbers, and names. Output only the Korean translation, "
        "without explanation.\nEnglish text:\n" + source
    ), False

def generate_ollama(model_id: str, source: str, peers: list[str]) -> tuple[str, dict]:
    tag, runner = SPECS[model_id]
    prompt, raw = prompt_for(model_id, source, peers)
    body = {
        "model": tag, "prompt": prompt, "stream": False, "keep_alive": "2m",
        "options": {"temperature": 0, "top_k": 1, "repeat_penalty": 1.0,
                    "num_ctx": 4096, "num_predict": 512},
    }
    if raw:
        body["raw"] = True
    if runner in ("generic", "index"):
        body["think"] = False
    start = time.monotonic()
    data = post("/api/generate", body)
    elapsed = round(time.monotonic() - start, 3)
    if data.get("error") or not data.get("done"):
        raise RuntimeError("Ollama returned an incomplete/error response")
    if data.get("done_reason") == "length":
        raise RuntimeError("Ollama translation truncated by token budget")
    translated = str(data.get("response") or "").strip()
    translated = re.sub(r"(?s)^<think>.*?</think>\s*", "", translated).strip()
    translated = re.sub(r"</s>|<\|im_end\|>", "", translated).strip(' "“”\r\n')
    translated = " ".join(translated.splitlines()).strip()
    if not KOREAN.search(translated):
        raise ValueError("No Korean characters in translation response")
    return translated, {
        "wall_seconds": elapsed,
        "prompt_tokens": data.get("prompt_eval_count"),
        "generated_tokens": data.get("eval_count"),
        "load_seconds": round(data.get("load_duration", 0) / 1e9, 3),
        "decode_seconds": round(data.get("eval_duration", 0) / 1e9, 3),
    }

def predictions_file(model_id: str) -> Path:
    return ROOT / f"{PREFIX}_{model_id}.jsonl"

def load_predictions(model_id: str, sample: dict) -> list[dict]:
    p = predictions_file(model_id)
    items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    if len(items) != len(sample["items"]):
        raise RuntimeError(f"{model_id} incomplete predictions: {len(items)} rows")
    for expected, item in zip(sample["items"], items):
        if (item["id"] != expected["id"] or item["domain"] != expected["domain"]
            or item.get("dataset_sha256") != DATASET_SHA256):
            raise RuntimeError(f"{model_id} prediction dataset mismatch at {expected['id']}")
        if item.get("status") != "PASS" or not KOREAN.search(item.get("prediction", "")):
            raise RuntimeError(f"{model_id} has invalid prediction at {expected['id']}")
    return items

def run_model(model_id: str, per_domain: int, resume_hash: str | None = None) -> int:
    sample = sample_data(per_domain)
    dest = predictions_file(model_id)
    previous: list[dict] = []
    if dest.exists() and resume_hash is None:
        saved = load_predictions(model_id, sample)
        print("[EXISTING_VALID]", model_id, "rows", len(saved), "sha256", digest(dest), flush=True)
        return 0
    if resume_hash is not None:
        if not dest.is_file() or digest(dest).casefold() != resume_hash.casefold():
            raise RuntimeError("Explicit resume SHA256 check failed")
        previous = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
        if not 1 <= len(previous) < len(sample["items"]):
            raise RuntimeError("Resumable file is not a partial prediction list")
        for expected, row in zip(sample["items"], previous):
            if (row.get("id") != expected["id"]
                or row.get("domain") != expected["domain"]
                or row.get("model") != model_id
                or row.get("dataset_sha256") != DATASET_SHA256
                or row.get("status") != "PASS"
                or not KOREAN.search(row.get("prediction", ""))):
                raise RuntimeError("Existing partial results invalid; refusing append")
        print("[RESUME_VERIFIED]", model_id, "preserving first", len(previous),
              "rows; sha256", resume_hash, flush=True)
    if SPECS[model_id][1] == "madlad":
        raise RuntimeError("MADLAD requires its separate llama.cpp runner; not an Ollama benchmark")
    if SPECS[model_id][1] == "nllb":
        from translate_extra import NLLB
        nllb = NLLB("English")
    else:
        nllb = None
        with urllib.request.urlopen(API + "/api/tags", timeout=8) as response:
            listed = {m.get("name", "").casefold() for m in json.load(response)["models"]}
        if SPECS[model_id][0].casefold() not in listed:
            raise RuntimeError(f"{model_id} missing from benchmark Ollama: {API}")
    first = time.monotonic()
    peers_by_model = {}
    if SPECS[model_id][1] == "chimera":
        for peer in DEFAULT_PEERS:
            peers_by_model[peer] = load_predictions(peer, sample)
    print("[BENCH_START]", model_id, "rows", len(sample["items"]),
          "model", SPECS[model_id][0], flush=True)
    with dest.open("a" if previous else "x", encoding="utf-8") as out:
        for index, item in enumerate(sample["items"][len(previous):], start=len(previous)):
            start = time.monotonic()
            row = {
                "id": item["id"], "domain": item["domain"],
                "model": model_id, "model_tag": SPECS[model_id][0],
                "dataset_sha256": DATASET_SHA256,
            }
            try:
                peers = [peers_by_model[p][index]["prediction"]
                         for p in DEFAULT_PEERS] if peers_by_model else []
                if nllb is None:
                    text, stats = generate_ollama(model_id, item["source"], peers)
                else:
                    text = nllb(item["source"])
                    stats = {"wall_seconds": round(time.monotonic()-start, 3)}
                text = " ".join(text.strip().splitlines()).strip()
                if not KOREAN.search(text):
                    raise ValueError("No Korean characters in translation")
                row.update({"status": "PASS", "prediction": text,
                            "statistics": stats})
            except (OSError, ValueError, RuntimeError, TimeoutError,
                    urllib.error.HTTPError, urllib.error.URLError) as error:
                row.update({"status": "FAIL", "error": f"{type(error).__name__}: {error}"[:500]})
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print("[BENCH_FAIL]", model_id, item["id"], row["error"], flush=True)
                return 2
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print("[BENCH_PROGRESS]", model_id, index+1, len(sample["items"]),
                  "seconds", row["statistics"]["wall_seconds"], flush=True)
    load_predictions(model_id, sample)
    print("[BENCH_PASS]", model_id, "elapsed", round(time.monotonic()-first, 3),
          "sha256", digest(dest), flush=True)
    return 0

def aggregate(per_domain: int) -> None:
    from sacrebleu.metrics import CHRF
    metric = CHRF(word_order=2, char_order=6, beta=2)
    sample = sample_data(per_domain)
    valid = {}
    for model in SPECS:
        file = predictions_file(model)
        if file.exists():
            try:
                valid[model] = load_predictions(model, sample)
            except (OSError, RuntimeError, ValueError) as exc:
                print("[EXCLUDED]", model, str(exc), flush=True)
    if not valid:
        raise ValueError("No complete model prediction files to compare")
    results = []
    for model, records in valid.items():
        pred = [r["prediction"] for r in records]
        ref = [r["reference"] for r in sample["items"]]
        grouped = {}
        for d in DOMAINS:
            inds = [i for i, x in enumerate(sample["items"]) if x["domain"] == d]
            grouped[d] = round(metric.corpus_score(
                [pred[i] for i in inds], [[ref[i] for i in inds]]
            ).score, 3)
        elapsed = [r["statistics"]["wall_seconds"] for r in records]
        results.append({
            "model": model,
            "model_tag": SPECS[model][0],
            "coverage": len(records), "total": len(sample["items"]),
            "chrfpp": round(metric.corpus_score(pred, [ref]).score, 3),
            "domain_chrfpp": grouped,
            "seconds_total": round(sum(elapsed), 3),
            "seconds_mean": round(sum(elapsed) / len(elapsed), 3),
            "predictions_sha256": digest(predictions_file(model)),
        })
    results.sort(key=lambda row: (-row["chrfpp"], row["seconds_mean"]))
    receipt = {
        "schema": "lztranslate_wmt24pp_results_v1",
        "dataset": OFFICIAL_URL,
        "dataset_sha256": DATASET_SHA256,
        "selected_sample": str(SAMPLE),
        "sample_sha256": digest(SAMPLE),
        "metric": "sacrebleu 2.5.1 chrF++ corpus-level word_order=2, char_order=6, beta=2; 0..100",
        "scope": "English->Korean, official WMT24++ post-edited references; sentence-level subset only",
        "warning": "Small stratified subset, surface similarity not semantic accuracy; data may overlap model training. External leaderboard results are not comparable.",
        "total_samples": len(sample["items"]),
        "models_measured": len(results),
        "models_unmeasured": [m for m in SPECS if m not in valid],
        "results": results,
    }
    result_file = ROOT / (PREFIX + "_comparison.json")
    if result_file.exists():
        raise FileExistsError(f"Comparison file already present: {result_file}")
    with result_file.open("x", encoding="utf-8") as out:
        json.dump(receipt, out, ensure_ascii=False, indent=2)
        out.write("\n")
    draw_chart(results, ROOT / (PREFIX + "_comparison.svg"))
    print("[COMPARISON]", result_file, "top", results[0]["model"],
          "chrF++", results[0]["chrfpp"], "model_count", len(results), flush=True)
    for model in results:
        print("[RANK]", model["model"], "chrF++", model["chrfpp"],
              "sec_per_item", model["seconds_mean"], flush=True)

def aggregate_combined() -> None:
    """Combine the fixed 16- and 32-item sets; first-run rows win overlap.

    References and predictions are independently verified for each underlying
    sample/prediction manifest. Four repeated IDs are counted only once.
    """
    import random
    from sacrebleu.metrics import BLEU, CHRF

    prefixes = ("wmt24pp_enko_20261105", "wmt24pp_enko_20261105_p8")
    finalists = ("gemma4", "index9", "hy18", "hy7", "tower9", "mi4")
    records: dict[tuple[str, int], dict] = {}
    evidence: dict[str, dict] = {}
    overlaps = 0
    for prefix in prefixes:
        manifest_file = ROOT / (prefix.replace("wmt24pp_enko_20261105",
                                                "wmt24pp_enko_sample_20261105") + ".json")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest["dataset_sha256"] != DATASET_SHA256:
            raise RuntimeError("Combined dataset SHA mismatch")
        prediction_by_model: dict[str, list[dict]] = {}
        sha: dict[str, str] = {}
        for model in finalists:
            path = ROOT / (prefix + f"_{model}.jsonl")
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
            if len(rows) != len(manifest["items"]):
                raise RuntimeError(f"Incomplete combined predictions for {prefix}/{model}")
            for item, row in zip(manifest["items"], rows):
                if (row.get("id") != item["id"] or row.get("domain") != item["domain"]
                    or row.get("status") != "PASS"
                    or row.get("dataset_sha256") != DATASET_SHA256
                    or not KOREAN.search(row.get("prediction", ""))):
                    raise RuntimeError(f"Combined prediction mismatch: {prefix}/{model}")
            prediction_by_model[model] = rows
            sha[model] = digest(path)
        evidence[prefix] = {"manifest_sha256": digest(manifest_file), "predictions_sha256": sha}
        for i, original in enumerate(manifest["items"]):
            key = (original["domain"], original["id"])
            if key in records:
                if (records[key]["source"] != original["source"]
                    or records[key]["reference"] != original["reference"]):
                    raise RuntimeError(f"Official duplicate reference mismatch: {key}")
                overlaps += 1
                continue
            records[key] = {
                "domain": original["domain"], "source": original["source"],
                "reference": original["reference"],
                "predictions": {m: prediction_by_model[m][i]["prediction"] for m in finalists},
                "seconds": {m: prediction_by_model[m][i]["statistics"]["wall_seconds"]
                            for m in finalists},
            }
    ordered = [records[k] for k in sorted(records)]
    if len(ordered) != 44 or overlaps != 4:
        raise RuntimeError(f"Unexpected unique/overlap count {len(ordered)}/{overlaps}")
    metric, bleu = CHRF(word_order=2, char_order=6, beta=2), BLEU(effective_order=True)
    references = [r["reference"] for r in ordered]
    results = []
    for model in finalists:
        hypothesis = [r["predictions"][model] for r in ordered]
        per_domain = {}
        for domain in DOMAINS:
            subset = [r for r in ordered if r["domain"] == domain]
            per_domain[domain] = round(metric.corpus_score(
                [r["predictions"][model] for r in subset],
                [[r["reference"] for r in subset]],
            ).score, 3)
        results.append({
            "model": model,
            "chrfpp": round(metric.corpus_score(hypothesis, [references]).score, 3),
            "bleu": round(bleu.corpus_score(hypothesis, [references]).score, 3),
            "domain_chrfpp": per_domain,
            "seconds_mean": round(sum(r["seconds"][model] for r in ordered)/len(ordered), 3),
            "coverage": len(ordered),
        })
    results.sort(key=lambda r: (-r["chrfpp"], r["seconds_mean"]))
    leader = results[0]["model"]
    rng = random.Random(20261105)
    intervals = {}
    for challenger in ("index9", "hy18"):
        delta = []
        for _ in range(500):
            sample = [ordered[rng.randrange(len(ordered))] for _ in ordered]
            refs = [r["reference"] for r in sample]
            superior = metric.corpus_score([r["predictions"][leader] for r in sample], [refs]).score
            inferior = metric.corpus_score([r["predictions"][challenger] for r in sample], [refs]).score
            delta.append(superior - inferior)
        delta.sort()
        intervals[challenger] = [round(delta[12], 3), round(delta[487], 3)]
    dest = ROOT / "wmt24pp_enko_20261105_combined44_comparison.json"
    svg = ROOT / "wmt24pp_enko_20261105_combined44_comparison.svg"
    if dest.exists() or svg.exists():
        raise FileExistsError("Combined report or chart already exists, no overwrite")
    report = {
        "schema": "lztranslate_wmt24pp_combined44_v1",
        "dataset": OFFICIAL_URL,
        "dataset_sha256": DATASET_SHA256,
        "source_evidence": evidence,
        "total_input_records": 48, "duplicates_ignored": overlaps, "unique": len(ordered),
        "duplicate_policy": "first (16-item) set has priority; later 32-item duplicate discarded",
        "domain_counts": {d: sum(r["domain"] == d for r in ordered) for d in DOMAINS},
        "metric": "sacrebleu 2.5.1 corpus chrF++ (word_order=2) and corpus BLEU",
        "winner": leader,
        "winner_semantics": "highest observed chrF++ on only 44 unique selected sentences; NOT globally proven best",
        "bootstrap_95pct_delta_to_challenger": intervals,
        "bootstrap": {"resamples": 500, "seed": 20261105, "paired": True},
        "results": results,
    }
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    draw_chart(results, svg)
    for item in results:
        print("[COMBINED_RANK]", item["model"], item["chrfpp"], item["bleu"], flush=True)
    print("[COMBINED_PASS]", "unique", len(ordered), "winner", leader,
          "report_sha256", digest(dest), "chart_sha256", digest(svg), flush=True)


def draw_chart(rows: list[dict], dest: Path) -> None:
    w = 1060
    margin = 230
    step = 37
    h = 118 + step * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="26" y="35" font-family="Segoe UI,Arial" font-size="21" font-weight="bold" fill="#222">WMT24++ EN to KO — chrF++ benchmark</text>',
        '<text x="26" y="58" font-family="Segoe UI,Arial" font-size="12" fill="#555">Higher is better. Official WMT24++ subset; not semantic translation accuracy.</text>',
    ]
    maxval = max(40, max(x["chrfpp"] for x in rows) * 1.16)
    barmax = 660
    for i, row in enumerate(rows):
        y = 87 + step * i
        width = round(barmax * row["chrfpp"] / maxval)
        label = html.escape(row["model"])
        color = "#384f69" if i else "#196b67"
        parts += [
            f'<text x="24" y="{y+17}" font-family="Segoe UI,Arial" font-size="14" fill="#263341">{i+1}. {label}</text>',
            f'<rect x="{margin}" y="{y}" width="{barmax}" height="22" rx="4" fill="#eceff1"/>',
            f'<rect x="{margin}" y="{y}" width="{width}" height="22" rx="4" fill="{color}"/>',
            f'<text x="{margin+barmax+12}" y="{y+16}" font-family="Segoe UI,Arial" font-size="14" fill="#263341">{row["chrfpp"]:.2f}</text>',
        ]
    parts.append("</svg>")
    with dest.open("x", encoding="utf-8") as out:
        out.write("\n".join(parts) + "\n")
    print("[CHART]", dest, flush=True)

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Official Google WMT24++ EN->KO batch comparison (stratified subset)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true", help="Pin and prepare the official sample")
    group.add_argument("--model", choices=sorted(SPECS), help="Evaluate one model, no overwrites")
    group.add_argument("--aggregate", action="store_true", help="Score completed models and export SVG")
    group.add_argument("--aggregate-combined", action="store_true", help="Combine and compare the 16+32 sample finalists")
    group.add_argument("--list", action="store_true", help="List supported model IDs")
    parser.add_argument("--per-domain", type=int, default=4, help="Equal samples per domain (default 4)")
    parser.add_argument("--resume-from-sha256", help="Explicit hash of existing partial prediction file")
    args = parser.parse_args()
    if not 1 <= args.per_domain <= 8:
        parser.error("--per-domain must be 1..8")
    global SAMPLE, PREFIX
    if args.per_domain != 4:
        SAMPLE = ROOT / f"wmt24pp_enko_sample_20261105_p{args.per_domain}.json"
        PREFIX = f"wmt24pp_enko_20261105_p{args.per_domain}"
    try:
        if args.list:
            for name, (tag, runner) in SPECS.items():
                print(name, runner, tag)
            return 0
        if args.prepare:
            data = sample_data(args.per_domain)
            print("[SAMPLE_PASS]", SAMPLE, "rows", len(data["items"]),
                  "sha256", digest(SAMPLE))
            return 0
        if args.aggregate:
            aggregate(args.per_domain)
            return 0
        if args.aggregate_combined:
            aggregate_combined()
            return 0
        return run_model(args.model, args.per_domain, args.resume_from_sha256)
    except (OSError, RuntimeError, ValueError,
            urllib.error.URLError, TimeoutError, ImportError) as exc:
        print("[BENCH_ERROR]", type(exc).__name__, str(exc), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
