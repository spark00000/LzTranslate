"""Five specialist translation models. Each produces an independent, validated Korean SRT."""
from __future__ import annotations
import argparse,contextlib,datetime,hashlib,json,os,pathlib,re,socket,subprocess,sys,time,urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
APP=pathlib.Path(os.environ.get("LZTRANSLATE_APP_HOME") or (pathlib.Path(pathlib.Path(__file__).resolve().anchor)/"APP"/"LzTranslate"))
MODEL_DIR=APP/"models"
sys.path.insert(0,str(ROOT))
from compare_srt import read_srt,validate_alignment
from telemetry import Profile
from model_catalog import BY_ID
KO=re.compile(r"[\uac00-\ud7a3]")
from model_catalog import llama_server_exe, OLLAMA_URL
LLAMA=(llama_server_exe() or APP/"engine"/"llama.cpp"/"build"/"bin"/"Release"/"llama-server.exe").parent
LANGS={"English":"eng_Latn","Japanese":"jpn_Jpan","Chinese (Simplified)":"zho_Hans","Chinese (Traditional)":"zho_Hant","German":"deu_Latn","French":"fra_Latn","Spanish":"spa_Latn","Italian":"ita_Latn","Portuguese":"por_Latn","Russian":"rus_Cyrl","Arabic":"arb_Arab","Hindi":"hin_Deva","Vietnamese":"vie_Latn","Indonesian":"ind_Latn","Thai":"tha_Thai","Turkish":"tur_Latn","Dutch":"nld_Latn","Polish":"pol_Latn","Czech":"ces_Latn","Ukrainian":"ukr_Cyrl","Hebrew":"heb_Hebr","Persian":"pes_Arab","Bengali":"ben_Beng","Filipino":"tgl_Latn","Khmer":"khm_Khmr","Tamil":"tam_Taml","Urdu":"urd_Arab","Swahili":"swh_Latn"}
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(2*1024*1024),b""):h.update(b)
 return h.hexdigest()
def strip_response(text):
 text=re.sub(r"(?s)^<think>.*?</think>\s*","",text.strip())
 text=re.sub(r"<\|im_end\|>|</s>","",text).strip(' "“”\n')
 if text.startswith("Korean:"):text=text[len("Korean:"):].strip()
 text=" ".join(text.splitlines()).strip()
 if not KO.search(text):raise ValueError("Missing Korean in output: "+text[:100])
 return text
def infer_ollama(model,prompt,raw):
 body={"model":BY_ID[model].tag,"prompt":prompt,"stream":False,"raw":raw,"keep_alive":"5m",
       "options":{"temperature":0,"top_k":1,"num_ctx":2048,"num_predict":256}}
 req=urllib.request.Request(OLLAMA_URL + "/api/generate",
       data=json.dumps(body,ensure_ascii=False).encode(),headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=115) as resp:data=json.load(resp)
 if data.get("error"):raise RuntimeError(str(data["error"])[:300])
 if data.get("done_reason")=="length":raise RuntimeError("Output truncated")
 return data.get("response","")
def form_prompt(model,language,text,peers):
 if model=="gemmax9":return f"Translate this from {language} to Korean:\n{language}: {text}\nKorean:"
 if model=="tower9":return f"Translate the following text from {language} into natural Korean. Preserve meaning, names and numbers. Output only the translation.\n\n{text}"
 if model=="chimera7":
  tick=chr(96)*3
  choices="\n".join(f"{n}. {tick}{p}{tick}" for n,p in enumerate(peers,1))
  return (f"Analyze the following multiple Korean translations of the {language} segment surrounded in triple backticks and generate a single refined Korean translation. Only output the refined translation, do not explain.\n\nThe {language} segment:\n{tick}{text}{tick}\n\nThe multiple Korean translations:\n{choices}")
 raise ValueError("No such prompt")
@contextlib.contextmanager
def madlad_server():
 file=MODEL_DIR/"madlad10"/"model-q5_k_m.gguf"
 if not file.is_file():raise FileNotFoundError(file)
 exe=LLAMA/"llama-server.exe"
 if not exe.is_file():raise FileNotFoundError(exe)
 with socket.socket() as s:
  s.bind(("127.0.0.1",0));port=s.getsockname()[1]
 logs=APP/"logs";logs.mkdir(parents=True,exist_ok=True)
 dest=logs/("madlad_server_"+datetime.datetime.now().strftime("%Y%m%dT%H%M%S")+".log")
 with dest.open("xb") as log:
  process=subprocess.Popen([str(exe),"-m",str(file),"-ngl","99","-c","1024","--host","127.0.0.1","--port",str(port)],
                     cwd=str(LLAMA),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  try:
   ready=False
   for _ in range(150):
    if process.poll() is not None:break
    try:
     with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=2) as resp:ready=json.load(resp).get("status")=="ok"
     if ready:break
    except (OSError,ValueError,TimeoutError):pass
    time.sleep(1)
   if not ready:
    raise RuntimeError("MADLAD server startup failed: "+dest.read_text("utf-8",errors="replace")[-2500:])
   yield port
  finally:
   process.terminate()
   try:process.wait(timeout=8)
   except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
def madlad(port,text):
 body={"prompt":"<2ko> "+text,"n_predict":256,"temperature":0}
 req=urllib.request.Request(f"http://127.0.0.1:{port}/completion",
       data=json.dumps(body,ensure_ascii=False).encode(),headers={"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=115) as resp:return json.load(resp).get("content","")
class NLLB:
 def __init__(self,language):
  if language not in LANGS:raise ValueError("Unknown NLLB source language")
  folder=MODEL_DIR/"nllb33"
  if not (folder/"model.bin").is_file():raise FileNotFoundError(folder)
  sys.path.insert(0,str(APP/"engine"/"vendor312"))
  # Reuse existing local CUDA 12 libraries without installing drivers or changing PATH.
  self._dll_handles=[]
  for directory in (APP/"engine"/"ollama"/"lib"/"ollama"/"cuda_v12",
                    APP/"engine"/"venv"/"Lib"/"site-packages"/"torch"/"lib"):
   if directory.is_dir() and hasattr(os,"add_dll_directory"):
    self._dll_handles.append(os.add_dll_directory(str(directory)))
  import ctranslate2
  from transformers import AutoTokenizer
  self.src=LANGS[language]
  self.tokenizer=AutoTokenizer.from_pretrained(str(folder),local_files_only=True,src_lang=self.src)
  try:
   self.engine=ctranslate2.Translator(str(folder),device="cuda",compute_type="int8")
   print("[NLLB] GPU inference",flush=True)
  except (OSError,RuntimeError) as ex:
   print("[NLLB] CPU fallback because GPU backend unavailable: "+str(ex)[:100],flush=True)
   self.engine=ctranslate2.Translator(str(folder),device="cpu",compute_type="int8",inter_threads=2,intra_threads=8)
 def __call__(self,text):
  self.tokenizer.src_lang=self.src
  tokens=self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
  try:
   outcome=self.engine.translate_batch([tokens],target_prefix=[["kor_Hang"]],beam_size=4,max_decoding_length=256)
  except RuntimeError as e:
   if not any(x in str(e).lower() for x in ("cublas","cudnn","cuda","dll")):raise
   import ctranslate2
   print("[NLLB] GPU runtime unavailable; switching to CPU int8: "+str(e)[:150],flush=True)
   self.engine=ctranslate2.Translator(str(MODEL_DIR/"nllb33"),device="cpu",compute_type="int8",inter_threads=2,intra_threads=8)
   outcome=self.engine.translate_batch([tokens],target_prefix=[["kor_Hang"]],beam_size=4,max_decoding_length=256)
  ids=self.tokenizer.convert_tokens_to_ids(outcome[0].hypotheses[0])
  return self.tokenizer.decode(ids,skip_special_tokens=True)
def run(args):
 m=args.model
 if args.source_language=="auto":raise ValueError("Specific source language required for specialist")
 source=read_srt(args.source)
 if args.output.exists():raise FileExistsError("Output SRT is protected: "+str(args.output))
 if args.max_cues<0:raise ValueError("Invalid cue limit")
 cues=source[:args.max_cues] if args.max_cues else source
 translations=[]
 for x in args.candidate:
  v=read_srt(x);validate_alignment(source,v);translations.append(v)
 if m=="chimera7" and len(translations)<2:raise ValueError("Chimera needs >=2 aligned translation candidates")
 if m!="chimera7" and translations:raise ValueError("Only Chimera accepts candidate SRTs")
 if m=="nllb33" and args.source_language not in LANGS:raise ValueError("NLLB language unknown")
 trace=args.output.with_suffix(".ko.trace.jsonl")
 if trace.exists():raise FileExistsError("Trace already exists: "+str(trace))
 started=time.monotonic()
 answers=[]
 ctx=madlad_server() if m=="madlad10" else contextlib.nullcontext(NLLB(args.source_language) if m=="nllb33" else None)
 with Profile(m,str(args.source),engine="specialist_local") as prof:
  try:
   with ctx as backend,trace.open("x",encoding="utf-8") as logs:
    for idx,(n,stamp,text) in enumerate(cues):
     cand=[v[idx][2] for v in translations]
     if m in ("tower9","gemmax9","chimera7"):raw=infer_ollama(m,form_prompt(m,args.source_language,text,cand),raw=m=="gemmax9")
     elif m=="madlad10":raw=madlad(backend,text)
     else:raw=backend(text)
     output=strip_response(raw)
     answers.append((n,stamp,output))
     logs.write(json.dumps({"cue":n,"output":output},ensure_ascii=False)+"\n");logs.flush()
     if n%10==0 or n==cues[-1][0]:print("[PROGRESS]",m,n,"/",len(cues),flush=True)
   prof.status="COMPLETED"
   prof.inference={"cues":len(answers)}
  except BaseException:
   prof.status="FAILED"
   raise
 assert len(answers)==len(cues)
 with args.output.open("x",encoding="utf-8",newline="") as dest:
  dest.write("\r\n\r\n".join(f"{n}\r\n{stamp}\r\n{txt}" for n,stamp,txt in answers)+"\r\n")
 validate_alignment(cues,read_srt(args.output))
 print("[OUTPUT]",m,str(args.output),"cues",len(cues),"seconds",round(time.monotonic()-started,2),"sha256",digest(args.output),flush=True)
 return 0
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--model",required=True,choices=["tower9","gemmax9","chimera7","madlad10","nllb33"])
 p.add_argument("--source",required=True,type=pathlib.Path)
 p.add_argument("--source-language",required=True)
 p.add_argument("--output",required=True,type=pathlib.Path)
 p.add_argument("--candidate",type=pathlib.Path,action="append",default=[])
 p.add_argument("--max-cues",type=int,default=0)
 a=p.parse_args()
 try:return run(a)
 except (OSError,ValueError,RuntimeError,ImportError,UnicodeError) as e:
  print("[ERROR]",type(e).__name__,str(e),file=sys.stderr,flush=True);return 1
if __name__=="__main__":sys.exit(main())
