"""Windows translation performance telemetry (standard library only).

Samples system CPU, system RAM and whole-GPU utilization/VRAM/power. These
system-wide counters include other applications: they are NOT model-exclusive.
Every run writes a new JSON file (exclusive-create) so prior evidence is never
overwritten. No system settings or services are modified.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid


class _FILETIME(ctypes.Structure):
    _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.DWORD), ("load_pct", wintypes.DWORD),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page", ctypes.c_ulonglong),
        ("available_page", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended", ctypes.c_ulonglong),
    ]


if os.name == "nt":
    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel.GetSystemTimes.argtypes = [
        ctypes.POINTER(_FILETIME), ctypes.POINTER(_FILETIME),
        ctypes.POINTER(_FILETIME),
    ]
    _kernel.GetSystemTimes.restype = wintypes.BOOL
    _kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
    _kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
else:
    _kernel = None


def _filetime_value(value: _FILETIME) -> int:
    return (int(value.high) << 32) | int(value.low)


def _system_cpu() -> tuple[int, int] | None:
    if _kernel is None:
        return None
    idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
    if not _kernel.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    return _filetime_value(idle), _filetime_value(kernel) + _filetime_value(user)


def _system_ram() -> float | None:
    if _kernel is None:
        return None
    state = _MEMORYSTATUSEX()
    state.length = ctypes.sizeof(state)
    if not _kernel.GlobalMemoryStatusEx(ctypes.byref(state)):
        return None
    return round((state.total_physical - state.available_physical) / (1024 ** 3), 3)


def _gpu_state() -> dict[str, float | None]:
    empty = {"gpu_pct": None, "vram_mib": None, "power_w": None}
    try:
        run = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=False,
        )
        if run.returncode:
            return empty
        line = run.stdout.splitlines()[0]
        values = [x.strip() for x in line.split(",")]
        if len(values) != 3:
            return empty
        numbers = []
        for val in values:
            try:
                numbers.append(float(val))
            except ValueError:
                numbers.append(None)
        return dict(zip(empty.keys(), numbers))
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return empty


def _peak(rows: list[dict], key: str) -> float | None:
    values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    return round(max(values), 3) if values else None


def _average(rows: list[dict], key: str) -> float | None:
    values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
    return round(sum(values) / len(values), 3) if values else None


class Profile:
    """Context manager around one translation, default sample interval 0.75 s."""

    def __init__(self, model: str, source: str, enabled: bool = True,
                 interval: float = 0.75, engine: str = "ollama"):
        self.engine = engine
        self.model = model
        self.source = source
        self.enabled = enabled
        self.interval = max(0.25, float(interval))
        self.status = "FAILED"
        self.inference: dict = {}
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._previous_cpu: tuple[int, int] | None = None
        self._mutex = threading.Lock()

    def _sample(self) -> None:
        now = time.perf_counter()
        cpu = _system_cpu()
        cpu_pct = None
        if cpu is not None and self._previous_cpu is not None:
            previous_idle, previous_total = self._previous_cpu
            idle_delta = max(0, cpu[0] - previous_idle)
            total_delta = cpu[1] - previous_total
            if total_delta > 0:
                cpu_pct = round(max(0, min(100, 100 * (total_delta - idle_delta) / total_delta)), 2)
        if cpu is not None:
            self._previous_cpu = cpu
        row = {
            "t_s": round(now - self.started, 3),
            "cpu_system_pct": cpu_pct,
            "ram_system_used_gib": _system_ram(),
        }
        row.update(_gpu_state())
        with self._mutex:
            self.samples.append(row)

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception as exc:
                print(f"[PROFILE_WARNING] Sampling error: {type(exc).__name__}", file=sys.stderr)

    def __enter__(self) -> Profile:
        if not self.enabled:
            return self
        self.started = time.perf_counter()
        self.started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        self._sample()
        self._worker = threading.Thread(target=self._poll, name="translate-profile", daemon=True)
        self._worker.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if not self.enabled:
            return False
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=6)
        self._sample()
        finished_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        elapsed = round(time.perf_counter() - self.started, 3)
        with self._mutex:
            rows = list(self.samples)
        initial_vram = rows[0].get("vram_mib") if rows else None
        initial_ram = rows[0].get("ram_system_used_gib") if rows else None
        vram_peak = _peak(rows, "vram_mib")
        ram_peak = _peak(rows, "ram_system_used_gib")
        record = {
            "event": "local_translation_profile_v1",
            "engine": self.engine,
            "model": self.model,
            "source": self.source,
            "status": "FAILED" if exc_type is not None else self.status,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed,
            "sample_interval_s": self.interval,
            "samples_count": len(rows),
            "cpu_system_peak_pct": _peak(rows, "cpu_system_pct"),
            "cpu_system_average_pct": _average(rows, "cpu_system_pct"),
            "ram_system_peak_used_gib": ram_peak,
            "ram_system_delta_peak_gib": round(ram_peak - initial_ram, 3) if ram_peak is not None and initial_ram is not None else None,
            "gpu_peak_util_pct": _peak(rows, "gpu_pct"),
            "gpu_mean_util_pct": _average(rows, "gpu_pct"),
            "gpu_peak_vram_mib": vram_peak,
            "gpu_vram_delta_peak_mib": round(vram_peak - initial_vram, 1) if vram_peak is not None and initial_vram is not None else None,
            "gpu_peak_power_w": _peak(rows, "power_w"),
            "inference": self.inference,
            "sample_series": rows,
            "measurement_scope": "system_and_entire_gpu_includes_other_processes",
        }
        print(
            "[PERF][TRANSLATE] "
            f"elapsed={elapsed}s cpu_peak={record['cpu_system_peak_pct']}% "
            f"ram_peak={ram_peak}GiB gpu_peak={record['gpu_peak_util_pct']}% "
            f"vram_peak={vram_peak}MiB samples={len(rows)}",
            flush=True,
        )
        try:
            folder = Path(os.environ.get(
                "TRANSLATE_PROFILE_DIR",
                str(Path(os.environ.get("LZTRANSLATE_APP_HOME") or (Path(Path(__file__).resolve().anchor) / "APP" / "LzTranslate")) / "metrics"),
            ))
            folder.mkdir(parents=True, exist_ok=True)
            name = dt.datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12] + ".json"
            target = folder / name
            with target.open("x", encoding="utf-8") as fh:
                json.dump(record, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            print(f"[PERF][TRANSLATE] report={target}", flush=True)
        except OSError as err:
            print(f"[PROFILE_WARNING] Evidence not saved: {err}", file=sys.stderr)
        return False
