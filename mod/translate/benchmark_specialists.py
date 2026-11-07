"""Benchmark translation-specialized LLMs with their documented, model-specific raw prompts.

- MiLMMT-v1 uses Xiaomi's exact source-name prompt, raw completion (not chat).
- Seed-X-PPO uses the required <ko> target token, raw completion (not chat).
- Uses preexisting, hashed NTREX human Korean references (no fabricated GPT scores).
- Keeps existing translators, model tags and user SRT files unchanged.
- Every run creates uniquely named exclusive output; failure evidence is retained.
"""
from __future__ import annotations
import argparse,collections,datetime as dt,hashlib,json,pathlib,re,statistics,sys,time,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parent
PREV=ROOT.parent/"multilingual_ntrex_20261029"
TRANSLATE=ROOT.parents[1]
sys.path.insert(0,str(TRANSLATE))
from compare_srt import chr_f2
from telemetry import Profile
from model_catalog import OLLAMA_URL

MODELS={
  "milmmt12":"hf.co/mradermacher/MiLMMT-46-12B-v1.0-GGUF:Q5_K_M",
  "milmmt4":"hf.co/mradermacher/MiLMMT-46-4B-v1.0-GGUF:Q8_0",
  "seed7":"hf.co/mradermacher/Seed-X-PPO-7B-GGUF:Q4_K_M",
}
LANGS={
 "eng-US":"English","ita":"Italian","por":"Portuguese","fil":"Tagalog",
 "fas":"Persian","ben":"Bengali","nld":"Dutch","pol":"Polish",
 "ces":"Czech","heb":"Hebrew","ukr":"Ukrainian","khm":"Khmer",
 "tam":"Tamil","urd":"Urdu","swa":"Swahili",
 "jpn":"Japanese","zho-CN":"Chinese (Simplified)","deu":"German",
 "fra":"French","spa":"Spanish","rus":"Russian","arb":"Arabic",
 "hin":"Hindi","vie":"Vietnamese","ind":"Indonesian",
 "tha":"Thai","tur":"Turkish",
}
MILLMMT_SUPPORTED=set(LANGS)-{"swa"}
SEED_SUPPORTED={"eng-US","ita","por","nld","pol","ces","ukr","jpn",
 "zho-CN","deu","fra","spa","rus","arb","vie","ind","tha","tur"}
SHARED_LANGS=SEED_SUPPORTED
KOREAN=re.compile(r"[\uac00-\ud7a3]")
def sha256(p:pathlib.Path)->str:
 return hashlib.sha256(p.read_bytes()).hexdigest()
def load_cases(scope, alias):
 data=[]
 for filename in ("probe_v1.json","probe_extra_v1.json"):
  src=PREV/filename
  d=json.loads(src.read_text(encoding="utf-8"))
  data.extend(d["cases"])
 assert len(data)==81
 permitted=SHARED_LANGS if scope=="shared18" else (
  SEED_SUPPORTED if alias=="seed7" else MILLMMT_SUPPORTED)
 filtered=[x for x in data if x["language_code"] in permitted]
 assert len({(x["language_code"],x["id"]) for x in filtered})==len(filtered)
 return filtered
def prompt(alias,c):
 src_lang=LANGS[c["language_code"]]
 txt=c["source"]
 if alias.startswith("milmmt"):
  return f"Translate this from {src_lang} to Korean:\n{src_lang}: {txt}\nKorean:"
 return f"Translate the following {src_lang} sentence into Korean:\n{txt} <ko>"
def request(model:str, prm:str)->dict:
 payload={
 "model":model,"prompt":prm,"raw":True,"stream":False,
 "keep_alive":"3m",
 "options":{"temperature":0,"top_k":1,"num_ctx":4096,
             "num_predict":256,"repeat_penalty":1.0},
 }
 r=urllib.request.Request(OLLAMA_URL + "/api/generate",
                         data=json.dumps(payload,ensure_ascii=False).encode("utf-8"),
                         headers={"Content-Type":"application/json"})
 started=time.perf_counter()
 with urllib.request.urlopen(r,timeout=105) as resp: data=json.load(resp)
 if data.get("error"):raise ValueError(str(data["error"])[:220])
 text=str(data.get("response") or "").strip()
 text=re.sub(r"(?s)^<think>.*?</think>\s*","",text).strip()
 text=re.sub(r"</?s>|<\|im_end\|>","",text).strip()
 text=text.strip(" \"“”").strip()
 if "\n" in text:text=text.splitlines()[0].strip()
 if not text:raise ValueError("empty translation")
 n=data.get("eval_count") or 0
 ns=data.get("eval_duration") or 0
 return {"output_ko":text,"contains_korean":bool(KOREAN.search(text)),
  "elapsed_seconds":round(time.perf_counter()-started,3),
  "total_duration_seconds":round((data.get("total_duration") or 0)/1e9,3),
  "load_seconds":round((data.get("load_duration") or 0)/1e9,3),
  "prompt_tokens":data.get("prompt_eval_count"),
  "generated_tokens":n,"generated_tokens_per_second":round(n*1e9/ns,2) if ns else None,
  "done_reason":data.get("done_reason"),}
def score(rows):
 by=collections.defaultdict(list)
 for x in rows: by[x["language_code"]].append(x)
 def metric(values):
  return chr_f2([(i,"",x.get("output_ko") if x.get("contains_korean") else "") for i,x in enumerate(values)],
                [(i,"",x["reference_ko"]) for i,x in enumerate(values)])
 speeds=[x["generated_tokens_per_second"] for x in rows if isinstance(x.get("generated_tokens_per_second"),float)]
 return {"corpus_chrf2_reference_similarity_pct":metric(rows),
  "macro_language_chrf2_pct":round(statistics.mean(metric(x) for x in by.values()),2),
  "valid_korean":sum(bool(x.get("contains_korean")) for x in rows),
  "count":len(rows),"languages":len(by),
  "mean_generation_tps":round(statistics.mean(speeds),2) if speeds else None,
  "language_chrf2_pct":{k:metric(x) for k,x in sorted(by.items())}}
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model",choices=MODELS,required=True)
 p.add_argument("--scope",choices=["shared18","supported"],default="shared18")
 p.add_argument("--limit",type=int,default=0,help="0 means full scope")
 p.add_argument("--run",required=True)
 a=p.parse_args()
 if not re.fullmatch(r"[a-zA-Z0-9_-]{3,65}",a.run):p.error("invalid run id")
 full=load_cases(a.scope,a.model)
 cases=full[:a.limit] if a.limit else full
 if not cases:p.error("empty selected benchmark")
 raw_path=ROOT/(a.run+".jsonl")
 report_path=ROOT/(a.run+"_report.json")
 if raw_path.exists() or report_path.exists():p.error("refusing overwrite")
 start=time.perf_counter()
 print("BENCH_START",a.model,a.scope,len(cases),flush=True)
 rows=[]
 with raw_path.open("x",encoding="utf-8") as out:
  with Profile(MODELS[a.model],str(PREV/"probe_v1.json")+";"+str(PREV/"probe_extra_v1.json"),
               enabled=True,engine="ollama_raw_translation_specialist") as monitor:
   for n,c in enumerate(cases,1):
    x={k:c[k] for k in ("id","language_code","source_language","source","reference_ko")}
    try:
     x.update(request(MODELS[a.model],prompt(a.model,c)))
     x["status"]="OK" if x["contains_korean"] else "NOT_KOREAN"
    except Exception as exc:
     x.update({"output_ko":"","contains_korean":False,"status":"ERROR",
              "error":type(exc).__name__+": "+str(exc)[:250]})
    rows.append(x);out.write(json.dumps(x,ensure_ascii=False)+"\n");out.flush()
    print("BENCH_PROGRESS",a.model,n,len(cases),x["language_code"],x["status"],x.get("elapsed_seconds"),flush=True)
   monitor.status="COMPLETED" if all(x["status"]=="OK" for x in rows) else "FAILED"
   monitor.inference={"model":a.model,"cases":len(rows),"summary":score(rows)}
 metrics=score(rows)
 report={
  "model":MODELS[a.model],"model_alias":a.model,"scope":a.scope,
  "created":dt.datetime.now().astimezone().isoformat(),
  "input_manifest_sha256":{"probe_v1":sha256(PREV/"probe_v1.json"),
                           "probe_extra_v1":sha256(PREV/"probe_extra_v1.json")},
  "metric":"corpus chrF2 character 1..6 grams whitespace removed; similarity to human Korean reference, not GPT quality or accuracy",
  "stats":metrics,"wall_seconds":round(time.perf_counter()-start,3),
  "raw_file":str(raw_path)}
 with report_path.open("x",encoding="utf-8") as out:json.dump(report,out,indent=2,ensure_ascii=False);out.write("\n")
 assert json.loads(report_path.read_text(encoding="utf-8"))==report
 print("BENCH_FINISH",a.model,json.dumps({"valid":metrics["valid_korean"],"total":metrics["count"],"chrF2":metrics["corpus_chrf2_reference_similarity_pct"],"elapsed":report["wall_seconds"],"tps":metrics["mean_generation_tps"]}),flush=True)
 print("BENCH_REPORT",str(report_path),flush=True)
 return 0 if metrics["valid_korean"]==len(rows) else 2
if __name__=="__main__":sys.exit(main())
