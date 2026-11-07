from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

from telemetry import Profile

BASE = os.environ.get("LZTRANSLATE_OLLAMA_URL", "http://127.0.0.1:12545").rstrip("/")
PREMIUM_MODEL = "hf.co/tencent/Hy-MT2-30B-A3B-GGUF:Q4_K_M"
MODEL_ALIASES = {"hy-mt2:30b": PREMIUM_MODEL, "hymt2:30b": PREMIUM_MODEL}
INFERENCE_STATS = {"requests": 0, "prompt_tokens": 0, "generated_tokens": 0,
                   "model_load_ns": 0, "prompt_eval_ns": 0, "generation_ns": 0,
                   "model_total_ns": 0}
STAMP = re.compile(r"^\d{2,}:\d{2}:\d{2},\d{3} --> \d{2,}:\d{2}:\d{2},\d{3}$")
KO = re.compile(r"[\uac00-\ud7a3]")
LABEL = re.compile(r"^\[(\d+)\]\s*(.*?)\s*$")

def read_srt(path):
    text = path.read_text(encoding="utf-8-sig")
    blocks = [b.strip() for b in re.split(r"\r?\n\s*\r?\n", text.strip()) if b.strip()]
    if not blocks:raise ValueError("Empty subtitle file")
    cues = []
    for n,block in enumerate(blocks,1):
        lines = block.splitlines()
        if len(lines)<3 or lines[0].strip()!=str(n) or not STAMP.fullmatch(lines[1].strip()):
            raise ValueError(f"Malformed source cue {n}")
        caption = " ".join(x.strip() for x in lines[2:] if x.strip())
        if not caption:raise ValueError(f"Empty source cue {n}")
        cues.append((n,lines[1].strip(),caption))
    return cues

def ollama(endpoint,payload=None,timeout=300):
    req=urllib.request.Request(BASE+endpoint,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type":"application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Local Ollama HTTP {e.code}: {e.read(800).decode('utf-8','replace')}") from e
    except (urllib.error.URLError,TimeoutError) as e:
        raise RuntimeError(f"Local Ollama is unavailable: {e}") from e

def infer(prompt,model,tokens):
    payload={"model":model,"prompt":prompt,"stream":False,
        "keep_alive":"12m","options":{"temperature":0,"num_ctx":2048,
        "num_predict":tokens,"repeat_penalty":1.05}}
    if model.casefold()==PREMIUM_MODEL.casefold():
        payload["options"].update({"temperature":0.7,"top_p":1.0,"top_k":-1,
                                   "repeat_penalty":1.0})
    elif "hy-mt2-7b-gguf" in model.lower() or model.lower().startswith("hy-mt2-1.8b-"):
        payload["options"].update({"temperature":0.7,"top_p":0.6,"top_k":20,
                                   "repeat_penalty":1.05})
    if model.lower().startswith(("gemma4:","qwen3.5:","qwen3.6:")):
        payload["think"]=False
    result=ollama("/api/generate",payload)
    INFERENCE_STATS["requests"] += 1
    for field, key in (("prompt_eval_count", "prompt_tokens"),
                       ("eval_count", "generated_tokens"),
                       ("load_duration", "model_load_ns"),
                       ("prompt_eval_duration", "prompt_eval_ns"),
                       ("eval_duration", "generation_ns"),
                       ("total_duration", "model_total_ns")):
        value=result.get(field)
        if isinstance(value,(int,float)) and value >= 0:
            INFERENCE_STATS[key] += value
    if result.get("error") or not result.get("done") or not result.get("response"):
        raise RuntimeError(f"Local inference failed: {str(result.get('error'))[:120]}")
    answer=result["response"].strip()
    if model.casefold()==PREMIUM_MODEL.casefold():
        answer=re.sub(r"\s*<eos:[0-9a-f]+>\s*$","",answer,flags=re.I).strip()
    return answer

def batch_translate(cues,model,style):
    prompt=("Translate each numbered subtitle from its original language into natural Korean dialogue. "
      "Preserve every [ID] number and the order. Return exactly one [ID] Korean line "
      "for each input line. Keep humor, proper nouns and character voices natural. "
      "No commentary, timestamps or formatting.\n")
    if style:prompt+="Style: "+style[:1300]+"\n"
    prompt+="\n"+"\n".join(f"[{n}] {caption}" for n,_,caption in cues)
    raw=infer(prompt,model,min(900,max(230,len(prompt)//2)))
    lines=[x.strip() for x in raw.splitlines() if x.strip()]
    result={}
    if len(lines)==len(cues):
        for line,(n,_stamp,source) in zip(lines,cues):
            m=LABEL.fullmatch(line)
            if not m or int(m[1])!=n or not KO.search(m[2]) or m[2].casefold()==source.casefold():
                break
            result[n]=m[2].strip()
        if len(result)==len(cues):return result,False
    print(f"[FALLBACK] cues {cues[0][0]}-{cues[-1][0]}",flush=True)
    result={}
    for n,_stamp,source in cues:
        prompt="Translate this subtitle from its original language into natural Korean dialogue. Output ONLY the Korean translation.\n"
        if style:prompt+="Style: "+style[:1000]+"\n"
        output=infer(prompt+"\n"+source,model,350)
        output=re.sub(r"^\[\d+\]\s*","",output).strip(' "“”')
        if "\n" in output:output=" ".join(x.strip() for x in output.splitlines() if x.strip())
        if not KO.search(output) or output.casefold()==source.casefold() or "\x60\x60\x60" in output:
            raise ValueError(f"Invalid translation for cue {n}: {output[:120]}")
        result[n]=output
    return result,True

def wrap_caption(text,width):
    result=[];current=""
    for word in text.split():
        nxt=(current+" "+word).strip()
        if current and len(nxt)>width:
            result.append(current);current=word
        else:current=nxt
    if current:result.append(current)
    return "\r\n".join(result)

def render(cues,translated,width):
    blocks=[f"{n}\r\n{stamp}\r\n{wrap_caption(translated[n],width)}" for n,stamp,_ in cues]
    out="\r\n\r\n".join(blocks)+"\r\n"
    check=[s.strip() for s in re.split(r"\r?\n\s*\r?\n",out.strip()) if s.strip()]
    if len(check)!=len(cues):raise ValueError("Translated cue count changed")
    for (n,stamp,_),block in zip(cues,check):
        lines=block.splitlines()
        if len(lines)<3 or lines[0].strip()!=str(n) or lines[1].strip()!=stamp or not KO.search(" ".join(lines[2:])):
            raise ValueError(f"Translated SRT structure mismatch at cue {n}")
    return out

def run(a):
    model=MODEL_ALIASES.get(a.model.lower(),a.model)
    source=a.input.resolve(strict=True)
    if source.suffix.lower()!=".srt" or source.stem.lower().endswith(".ko") or not source.is_file():
        raise ValueError("Input must be an existing original-language SRT")
    suffix=(".hymt2.ko.srt" if model==PREMIUM_MODEL else
            ".gemma4.ko.srt" if model.lower().startswith("gemma4:") else
            ".qwen35.ko.srt" if model.lower().startswith("qwen3.5:") else
            ".qwen36.ko.srt" if model.lower().startswith("qwen3.6:") else
            ".hymt2-7b.ko.srt" if "Hy-MT2-7B-GGUF" in model else
            ".hymt2-1.8b.ko.srt" if model.lower().startswith("hy-mt2-1.8b") else
            ".ko.srt")
    target=(a.output or source.with_name(source.stem+suffix)).resolve()
    if os.path.normcase(str(source))==os.path.normcase(str(target)):
        raise ValueError("Refusing to overwrite input SRT")
    if target.exists():raise FileExistsError(f"EXISTING_OUTPUT_PROTECTED: {target}")
    if not target.parent.is_dir():raise ValueError(f"Output directory does not exist: {target.parent}")
    if not 1<=a.batch_size<=8:raise ValueError("--batch-size must be 1..8")
    if not 20<=a.width<=80:raise ValueError("--width must be 20..80")
    cues=read_srt(source)
    src_digest=hashlib.sha256(source.read_bytes()).hexdigest()
    models={str(m.get("name")).casefold():str(m.get("name"))
            for m in ollama("/api/tags",timeout=12).get("models",[]) if m.get("name")}
    if model.casefold() not in models:raise ValueError(f"Local Ollama model is not installed: {model}")
    model=models[model.casefold()]
    style=a.md.read_text(encoding="utf-8-sig") if a.md else ""
    if a.source_language.lower()!="auto":
        style="Source language: "+a.source_language[:100]+"\n"+style
    print(f"[INPUT] cues={len(cues)} model={model} source={source} target={target}",flush=True)
    if a.dry_run:
        print("[DRY_RUN] Input validated, local model is present, output is available",flush=True)
        return 0
    translated={};fallbacks=0;started=time.monotonic()
    for offset in range(0,len(cues),a.batch_size):
        chunk=cues[offset:offset+a.batch_size]
        result,fallback=batch_translate(chunk,model,style)
        translated.update(result);fallbacks+=int(fallback)
        print(f"[PROGRESS] {len(translated)}/{len(cues)} elapsed={time.monotonic()-started:.1f}s",flush=True)
    output=render(cues,translated,a.width)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=src_digest:
        raise RuntimeError("Source changed while translating")
    with target.open("x",encoding="utf-8",newline="") as stream:
        stream.write(output)
    if target.read_bytes().decode("utf-8")!=output:
        raise RuntimeError("Written SRT verification failed")
    print("[RESULT] "+json.dumps({"status":"COMPLETED","model":model,"cues":len(cues),
        "output":str(target),"elapsed_seconds":round(time.monotonic()-started,3),
        "fallback_batches":fallbacks,"sha256":hashlib.sha256(target.read_bytes()).hexdigest()},
        ensure_ascii=False),flush=True)
    return 0

def main():
    p=argparse.ArgumentParser(description="Local GPU SRT -> Korean translator via Ollama")
    p.add_argument("input",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--model",default="translategemma:27b")
    p.add_argument("--source-language",default="auto",help="Original subtitle language or auto (default)")
    p.add_argument("--md",type=Path)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--width",type=int,default=42)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    t=time.monotonic()
    print(f"[TIMING][TRANSLATE] START {time.strftime('%Y-%m-%d %H:%M:%S')}",flush=True)
    with Profile(a.model, str(a.input), enabled=not a.dry_run) as monitor:
        try:
            result=run(a)
            monitor.status="COMPLETED" if result==0 else "FAILED"
            return result
        except (OSError,ValueError,RuntimeError,UnicodeError) as e:
            print(f"[ERROR] {type(e).__name__}: {e}",file=sys.stderr,flush=True)
            return 1
        finally:
            monitor.inference=dict(INFERENCE_STATS)
            if INFERENCE_STATS["generation_ns"]:
                rate=INFERENCE_STATS["generated_tokens"]*1e9/INFERENCE_STATS["generation_ns"]
                print(f"[INFERENCE] requests={INFERENCE_STATS['requests']} generated_tokens={INFERENCE_STATS['generated_tokens']} tokens_per_second={rate:.2f}",flush=True)
            print(f"[TIMING][TRANSLATE] FINISH {time.strftime('%Y-%m-%d %H:%M:%S')} elapsed={time.monotonic()-t:.3f}s",flush=True)

if __name__=="__main__":sys.exit(main())
