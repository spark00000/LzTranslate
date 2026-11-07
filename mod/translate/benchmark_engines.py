"""Same-GGUF Ollama vs llama.cpp server test, Windows local-only.

Runs three short generation trials against the SAME installed GGUF, while
recording system-wide CPU/RAM/GPU samples with telemetry.Profile. Benchmarks
decode throughput (generated tokens per second), NOT translation quality.

Ollama requests use its native generate API; llama.cpp receives a raw
completion prompt. Different model prompt-template handling means these are
useful indicative runtimes, not a controlled claim of percentage superiority.
All reports are exclusive-create and the temporary llama server is stopped.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from compare_srt import read_srt
from telemetry import Profile
from model_catalog import OLLAMA_URL


def post_json(url: str, payload: dict, timeout: int = 150) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def build_prompt(srt: Path) -> str:
    cues = read_srt(srt)[:8]
    return (
        "Translate each numbered English subtitle into natural Korean dialogue. "
        "Preserve every [ID] number and the order. Return exactly one [ID] Korean line "
        "for each input line. No commentary, timestamps or formatting.\n\n" +
        "\n".join(f"[{idx}] {text}" for idx, _stamp, text in cues)
    )


def mean_rate(rows: list[dict]) -> float | None:
    rates = [x.get("tokens_per_second") for x in rows]
    values = [v for v in rates if isinstance(v, (int, float)) and v > 0]
    return round(statistics.mean(values), 3) if values else None


def benchmark_ollama(model: str, prompt: str, runs: int) -> list[dict]:
    values = []
    with Profile(model, "engine_bench_fixed_8cue_prompt", engine="ollama") as profiler:
        for trial in range(1, runs + 1):
            result = post_json(
                OLLAMA_URL + "/api/generate",
                {
                    "model": model, "prompt": prompt, "stream": False,
                    "keep_alive": "2m" if trial < runs else 0,
                    "options": {"temperature": 0, "num_ctx": 2048,
                                "num_predict": 128, "repeat_penalty": 1.0},
                },
            )
            duration_ns = result.get("eval_duration", 0)
            n_tokens = result.get("eval_count", 0)
            item = {
                "trial": trial,
                "generated_tokens": n_tokens,
                "decode_duration_s": round(duration_ns / 1e9, 3),
                "tokens_per_second": round(n_tokens * 1e9 / duration_ns, 3)
                if duration_ns else None,
                "prompt_tokens": result.get("prompt_eval_count"),
                "model_load_seconds": round(result.get("load_duration", 0) / 1e9, 3),
                "text_sample": str(result.get("response", ""))[:120],
            }
            values.append(item)
            print(f"[OLLAMA] trial={trial} tokens={n_tokens} rate={item['tokens_per_second']}", flush=True)
        profiler.status = "COMPLETED"
        profiler.inference = {"runs": values, "average_generation_tps": mean_rate(values)}
    return values


def port_is_in_use(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
        return True
    except urllib.error.HTTPError:
        return True
    except urllib.error.URLError:
        return False


def await_health(proc: subprocess.Popen, port: int, max_seconds: int = 95) -> None:
    started = time.monotonic()
    while time.monotonic() - started < max_seconds:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited during startup: {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(0.6)
    raise TimeoutError("llama-server did not reach ready state")


def benchmark_llama(server_exe: Path, model_path: Path,
                    prompt: str, runs: int, port: int, logdir: Path) -> list[dict]:
    if port_is_in_use(port):
        raise RuntimeError(f"Port {port} is already in use; refusing to interfere")
    values = []
    path = logdir / ("server_" + uuid.uuid4().hex[:10] + ".log")
    with path.open("x", encoding="utf-8") as stderr:
        cmd = [
            str(server_exe), "-m", str(model_path), "-c", "2048",
            "-ngl", "99", "--host", "127.0.0.1",
            "--port", str(port), "--no-webui",
        ]
        with Profile(str(model_path), "engine_bench_fixed_8cue_prompt",
                     engine="llama.cpp") as profiler:
            proc = subprocess.Popen(cmd, cwd=str(server_exe.parent),
                                    stdout=stderr, stderr=subprocess.STDOUT)
            try:
                await_health(proc, port)
                for trial in range(1, runs + 1):
                    start = time.perf_counter()
                    result = post_json(
                        f"http://127.0.0.1:{port}/completion",
                        {
                            "prompt": prompt, "n_predict": 128, "temperature": 0,
                            "stream": False, "cache_prompt": False,
                            "stop": ["<eos:6124c78e>"],
                        },
                    )
                    wall = time.perf_counter() - start
                    timings = result.get("timings") or {}
                    tokens = timings.get("predicted_n") or result.get("tokens_predicted") or 0
                    decoded_ms = timings.get("predicted_ms")
                    rate = (1000 * tokens / decoded_ms) if decoded_ms else timings.get("predicted_per_second")
                    item = {
                        "trial": trial,
                        "generated_tokens": tokens,
                        "decode_duration_s": round(decoded_ms / 1000, 3)
                        if decoded_ms else None,
                        "tokens_per_second": round(rate, 3) if rate else None,
                        "request_elapsed_s": round(wall, 3),
                        "text_sample": str(result.get("content", ""))[:120],
                    }
                    values.append(item)
                    print(f"[LLAMA.CPP] trial={trial} tokens={tokens} rate={item['tokens_per_second']}", flush=True)
                profiler.status = "COMPLETED"
                profiler.inference = {"runs": values, "average_generation_tps": mean_rate(values)}
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=15)
    return values


def main() -> int:
    p = argparse.ArgumentParser(description="Compare the same GGUF in Ollama and llama.cpp")
    p.add_argument("--ollama-model", required=True)
    p.add_argument("--model-file", required=True, type=Path)
    p.add_argument("--llama-server", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--port", type=int, default=18172)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.runs <= 5:
        p.error("--runs must be 1..5")
    for f in [args.model_file, args.llama_server, args.source]:
        if not f.is_file():
            p.error(f"Input missing: {f}")
    if not args.output_dir.is_dir():
        p.error("Existing --output-dir is required")
    prompt = build_prompt(args.source)
    print(f"[BENCH] identical GGUF={args.model_file} prompt_chars={len(prompt)}", flush=True)
    try:
        ollama_results = benchmark_ollama(args.ollama_model, prompt, args.runs)
        llama_results = benchmark_llama(
            args.llama_server, args.model_file, prompt,
            args.runs, args.port, args.output_dir,
        )
        report = {
            "event": "engine_compare_v1",
            "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "model_file": str(args.model_file),
            "model_size_bytes": args.model_file.stat().st_size,
            "ollama_model": args.ollama_model,
            "runs": args.runs,
            "prompt_characters": len(prompt),
            "generation_budget_tokens": 128,
            "context_tokens": 2048,
            "ollama": {"trials": ollama_results, "mean_tps": mean_rate(ollama_results)},
            "llama_cpp": {"trials": llama_results, "mean_tps": mean_rate(llama_results)},
            "caveat": "Same weights and nominal prompt; API model templates differ. Do not equate token rate to translation quality.",
        }
        dest = args.output_dir / ("engine_compare_" + uuid.uuid4().hex[:12] + ".json")
        with dest.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"[BENCH_RESULT] ollama_mean_tps={report['ollama']['mean_tps']} "
              f"llama_cpp_mean_tps={report['llama_cpp']['mean_tps']} report={dest}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError,
            urllib.error.HTTPError, urllib.error.URLError) as error:
        print(f"[BENCH_ERROR] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
