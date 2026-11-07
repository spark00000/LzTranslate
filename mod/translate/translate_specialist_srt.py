"""Translate subtitle cue-by-cue with an officially prompted translation-specific LLM.

Pure add-on to the existing translator; does not change original ASR/SRT or Ollama
models. Uses raw completion with source-language choice and exclusive-create output.
The source SRT, telemetry, SHA256 and individual-cue response trace are preserved.
"""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,pathlib,re,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parents[1]))
from benchmark_specialists import MODELS,request
from compare_srt import read_srt,validate_alignment
from telemetry import Profile
KO=re.compile(r"[\uac00-\ud7a3]")
LANGUAGE_ALLOWED=re.compile(r"^[A-Za-z][A-Za-z (),.-]{1,75}$")
def make_prompt(alias,language,text):
 if alias.startswith("milmmt"):
  return f"Translate this from {language} to Korean:\n{language}: {text}\nKorean:"
 return f"Translate the following {language} sentence into Korean:\n{text} <ko>"
def sha(path):
 return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description="Translate subtitle SRT using MiLMMT or Seed-X-PPO raw prompts")
 p.add_argument("--model",choices=MODELS,required=True)
 p.add_argument("--source",type=pathlib.Path,required=True)
 p.add_argument("--source-language",required=True,help="e.g. English, German, Japanese, Chinese (Simplified)")
 p.add_argument("--output",type=pathlib.Path,required=True)
 p.add_argument("--max-cues",type=int,default=0,help="For a smoke test, choose number of cues; otherwise all cues")
 a=p.parse_args()
 if not LANGUAGE_ALLOWED.fullmatch(a.source_language):p.error("Invalid source language name")
 if not a.source.is_file():p.error("source SRT not found")
 original=read_srt(a.source)
 if a.max_cues<0:p.error("Invalid max cues")
 cues=original[:a.max_cues] if a.max_cues else original
 if a.output.exists():p.error("Destination exists, refusing overwrite")
 if not a.output.parent.exists():p.error("Destination parent does not exist")
 trace=a.output.with_suffix(a.output.suffix+".jsonl")
 summary=a.output.with_suffix(a.output.suffix+".report.json")
 if trace.exists() or summary.exists():p.error("Trace/summary already exists")
 records=[]
 started=time.perf_counter()
 print("SRT_START",a.model,"input_cues",len(cues),"language",a.source_language,flush=True)
 with trace.open("x",encoding="utf-8") as log:
  with Profile(MODELS[a.model],str(a.source),engine="ollama_raw_translation_specialist") as monitor:
   for n,stamp,caption in cues:
    row={"cue":n,"timecode":stamp,"source":caption}
    try:
     result=request(MODELS[a.model],make_prompt(a.model,a.source_language,caption))
     row.update(result)
     row["status"]="VALID" if result["contains_korean"] else "NOT_KOREAN"
    except Exception as exc:
     row["status"]="ERROR"
     row["error"]=type(exc).__name__+": "+str(exc)[:320]
    records.append(row)
    log.write(json.dumps(row,ensure_ascii=False)+"\n");log.flush()
    if n%10==0 or n==cues[-1][0]:
     print("SRT_PROGRESS",a.model,n,"of",cues[-1][0],"last_status",row["status"],flush=True)
   valid=sum(x["status"]=="VALID" for x in records)
   monitor.status="COMPLETED" if valid==len(cues) else "FAILED"
   monitor.inference={"cues":len(cues),"valid":valid,"token_generation_mean":(round(sum(x["generated_tokens_per_second"] for x in records if x.get("generated_tokens_per_second") is not None)/sum(x.get("generated_tokens_per_second") is not None for x in records),2) if any(x.get("generated_tokens_per_second") is not None for x in records) else None)}
 valid=sum(x["status"]=="VALID" for x in records)
 metrics={"model":a.model,"tag":MODELS[a.model],
  "source":str(a.source),"source_sha256":sha(a.source),"source_language":a.source_language,
  "cue_count":len(cues),"valid_korean":valid,"elapsed_s":round(time.perf_counter()-started,3),
  "created":dt.datetime.now().astimezone().isoformat(),
  "status":"COMPLETED" if valid==len(cues) else "FAILED",
  "caveat":"SRT number/timecode validation and Korean-script presence are structural, not semantic translation-quality assessment."}
 if valid==len(cues):
  parts=[]
  for original_cue,candidate in zip(cues,records):
   number,stamp,_=original_cue
   text=candidate["output_ko"].replace("\n"," ").replace("\r"," ").strip()
   parts.append(f"{number}\r\n{stamp}\r\n{text}")
  with a.output.open("x",encoding="utf-8",newline="") as out:
   out.write("\r\n\r\n".join(parts)+"\r\n")
  validate_alignment(cues,read_srt(a.output))
  metrics["output"]=str(a.output)
  metrics["output_sha256"]=sha(a.output)
 with summary.open("x",encoding="utf-8") as out:
  json.dump(metrics,out,ensure_ascii=False,indent=2);out.write("\n")
 print("SRT_FINISH",json.dumps(metrics,ensure_ascii=True),flush=True)
 return 0 if valid==len(cues) else 2
if __name__=="__main__":sys.exit(main())
