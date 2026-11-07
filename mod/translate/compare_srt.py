"""Reference-relative SRT comparison, with no external packages.

chrF2 here means character n-gram F-score (orders 1..6, beta=2), whitespace
removed, corpus-level matching. The result is a TEXT-SIMILARITY score
against one reference, NOT a measure of absolute translation correctness.
It must not be relabeled as translation accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys


STAMP = re.compile(r"^\d{2,}:\d{2}:\d{2},\d{3} --> \d{2,}:\d{2}:\d{2},\d{3}$")


def read_srt(path: Path) -> list[tuple[int, str, str]]:
    raw = path.read_text(encoding="utf-8-sig")
    blocks = [b.strip() for b in re.split(r"\r?\n\s*\r?\n", raw.strip()) if b.strip()]
    if not blocks:
        raise ValueError(f"Empty SRT: {path}")
    cues = []
    for expected, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if (len(lines) < 3 or lines[0].strip() != str(expected) or
                STAMP.fullmatch(lines[1].strip()) is None):
            raise ValueError(f"Invalid SRT cue {expected}: {path}")
        caption = " ".join(x.strip() for x in lines[2:] if x.strip())
        if not caption:
            raise ValueError(f"Empty cue {expected}: {path}")
        cues.append((expected, lines[1].strip(), caption))
    return cues


def validate_alignment(source: list, comparison: list) -> None:
    if len(source) != len(comparison):
        raise ValueError(f"Cue count mismatch: {len(source)} vs {len(comparison)}")
    for original, candidate in zip(source, comparison):
        if original[:2] != candidate[:2]:
            raise ValueError(f"Cue index or timestamp differs at cue {original[0]}")


def chr_f2(candidate: list, reference: list) -> float:
    """Corpus char n-gram F2, character orders 1 through 6, range 0..100."""
    if not candidate or len(candidate) != len(reference):
        raise ValueError("Invalid candidate/reference length")
    scores = []
    for order in range(1, 7):
        overlap = candidate_total = reference_total = 0
        for (_, _, hypothesis), (_, _, truth) in zip(candidate, reference):
            hypothesis = "".join(hypothesis.split())
            truth = "".join(truth.split())
            hyp_counter = Counter(
                hypothesis[i:i+order] for i in range(max(0, len(hypothesis)-order+1)))
            ref_counter = Counter(
                truth[i:i+order] for i in range(max(0, len(truth)-order+1)))
            overlap += sum(min(v, ref_counter.get(k, 0)) for k, v in hyp_counter.items())
            candidate_total += sum(hyp_counter.values())
            reference_total += sum(ref_counter.values())
        precision = overlap / candidate_total if candidate_total else 0
        recall = overlap / reference_total if reference_total else 0
        scores.append((5*precision*recall/(4*precision+recall)) if precision and recall else 0)
    return round(100 * sum(scores) / 6, 2)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(source: Path, reference: Path, candidates: dict[str, Path]) -> dict:
    original = read_srt(source)
    truth = read_srt(reference)
    validate_alignment(original, truth)
    data = {
        "schema": "translate_eval_v1",
        "metric": "corpus_char_ngram_F2_orders_1_to_6_no_whitespace",
        "meaning": "similarity_to_GPT_reference_not_absolute_translation_accuracy",
        "evaluated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "reference": str(reference.resolve()),
        "reference_sha256": sha256(reference),
        "cue_count": len(original),
        "results": [],
    }
    for name, path in candidates.items():
        item = {"model": name, "path": str(path.resolve()), "sha256": sha256(path)}
        try:
            cues = read_srt(path)
            validate_alignment(original, cues)
            item["cues"] = len(cues)
            item["timestamp_and_order_preserved"] = True
            item["hangul_cues"] = sum(bool(re.search(r"[\uac00-\ud7a3]", text)) for _, _, text in cues)
            item["chr_f2_reference_similarity_pct"] = chr_f2(cues, truth)
        except (UnicodeError, ValueError, OSError) as error:
            item["status"] = "INVALID"
            item["error"] = str(error)
        else:
            item["status"] = "VALID"
        data["results"].append(item)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SRT alignment and GPT-reference chrF2 comparison")
    parser.add_argument("--source", required=True, type=Path, help="English SRT")
    parser.add_argument("--reference", required=True, type=Path, help="GPT Korean SRT")
    parser.add_argument("--candidate", action="append", required=True, help="MODEL=LOCAL_SRT (repeat)")
    parser.add_argument("--report-file", type=Path, help="Optional JSON path, must not exist")
    arguments = parser.parse_args(argv)
    candidates: dict[str, Path] = {}
    for pair in arguments.candidate:
        if "=" not in pair:
            parser.error(f"Invalid --candidate, expected MODEL=PATH: {pair}")
        name, path = pair.split("=", 1)
        if not name.strip() or not path.strip() or name in candidates:
            parser.error(f"Invalid/duplicate candidate: {pair}")
        candidates[name] = Path(path)
    try:
        result = evaluate(arguments.source, arguments.reference, candidates)
        print("GPT-referenced chrF2 text similarity (NOT absolute quality)")
        print(f"Cues: {result['cue_count']}")
        for item in result["results"]:
            print(f"  {item['model']:21s}  {item.get('chr_f2_reference_similarity_pct', 'INVALID')}%  "
                  f"status={item['status']} timestamps={item.get('timestamp_and_order_preserved', False)}")
        if arguments.report_file:
            with arguments.report_file.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"Report: {arguments.report_file}")
        return 0 if all(x["status"] == "VALID" for x in result["results"]) else 1
    except (OSError, ValueError, UnicodeError) as error:
        print(f"[EVAL_ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
