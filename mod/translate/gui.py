"""LzTranslate Tkinter GUI for local batch SRT/TXT/MD translation.

Shows installed models, supports multi-file / whole-folder queues, preview,
bounded cancellation and in-place installation without modifying input files.
"""
from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from model_catalog import APP, BY_ID, LANGUAGES, MODELS, OLLAMA_EXE, status_rows

HERE = Path(__file__).resolve().parent
FILTERS = [
    ("Subtitles and text", "*.srt *.txt *.md"),
    ("Subtitles", "*.srt"),
    ("Plain text", "*.txt *.md"),
    ("All files", "*.*"),
]

class Studio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LzTranslate | Local Batch Translation")
        self.geometry("1040x760")
        self.minsize(820, 620)
        self.files: list[Path] = []
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.language_var = tk.StringVar(value="auto")
        self.output_var = tk.StringVar(value="")
        self.folder_var = tk.StringVar(value="")
        self.recursive_var = tk.BooleanVar(value=False)
        self.batch_var = tk.StringVar(value="4")
        self.width_var = tk.StringVar(value="42")
        self.status_var = tk.StringVar(value="Ready")
        self._layout()
        self.refresh()
        self.after(120, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _layout(self):
        top = ttk.Frame(self, padding=14)
        top.pack(fill="both", expand=True)
        ttk.Label(top, text="Local Translation Studio", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        ttk.Label(
            top,
            text="SRT / TXT / MD → Korean | source files protected | multiple files and models",
        ).pack(anchor="w", pady=(0, 10))
        inputs = ttk.LabelFrame(top, text="1. Select files or input folder", padding=8)
        inputs.pack(fill="both")
        inputs.columnconfigure(0, weight=1)
        inputs.rowconfigure(0, weight=1)
        self.filelist = tk.Listbox(inputs, height=5, selectmode=tk.EXTENDED)
        self.filelist.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(inputs, orient="vertical", command=self.filelist.yview)
        self.filelist.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky="ns")
        actions = ttk.Frame(inputs)
        actions.grid(row=0, column=2, padx=(8, 0), sticky="ns")
        ttk.Button(actions, text="Add files...", command=self._choose_files).pack(fill="x")
        ttk.Button(actions, text="Remove selected", command=self._remove_selected).pack(fill="x", pady=(5, 0))
        ttk.Button(actions, text="Clear list", command=self._clear_files).pack(fill="x", pady=(5, 0))
        folder = ttk.Frame(inputs)
        folder.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(7, 0))
        folder.columnconfigure(1, weight=1)
        ttk.Label(folder, text="Batch folder:").grid(row=0, column=0)
        ttk.Entry(folder, textvariable=self.folder_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(folder, text="Browse...", command=self._choose_folder).grid(row=0, column=2)
        ttk.Checkbutton(folder, text="Recursive", variable=self.recursive_var).grid(row=0, column=3, padx=(8, 0))
        options = ttk.LabelFrame(top, text="2. Translation settings", padding=8)
        options.pack(fill="x", pady=(9, 0))
        options.columnconfigure(1, weight=1)
        ttk.Label(options, text="Source language").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            options, textvariable=self.language_var, values=LANGUAGES,
            state="readonly", width=24,
        ).grid(row=0, column=1, sticky="w", padx=(8, 16))
        ttk.Label(options, text="Cue batch").grid(row=0, column=2)
        ttk.Spinbox(options, from_=1, to=8, width=6, textvariable=self.batch_var).grid(
            row=0, column=3, padx=8
        )
        ttk.Label(options, text="Line width").grid(row=0, column=4)
        ttk.Spinbox(options, from_=20, to=80, width=6, textvariable=self.width_var).grid(
            row=0, column=5, padx=8
        )
        ttk.Label(options, text="Output folder (blank = source folder)").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )
        ttk.Entry(options, textvariable=self.output_var).grid(
            row=2, column=0, columnspan=5, sticky="ew", pady=(2, 2), padx=(0, 8)
        )
        ttk.Button(options, text="Browse...", command=self._choose_output).grid(row=2, column=5)
        models = ttk.LabelFrame(top, text="3. Select one or more local models (Ctrl/Shift)", padding=8)
        models.pack(fill="both", expand=True, pady=(10, 0))
        models.columnconfigure(0, weight=1)
        models.rowconfigure(0, weight=1)
        self.modeltree = ttk.Treeview(
            models, columns=("name", "status", "note"), show="headings",
            selectmode="extended", height=8,
        )
        self.modeltree.heading("name", text="Model ID and name")
        self.modeltree.heading("status", text="Runtime status")
        self.modeltree.heading("note", text="Notes")
        self.modeltree.column("name", width=300, stretch=True)
        self.modeltree.column("status", width=115, stretch=False)
        self.modeltree.column("note", width=330, stretch=True)
        self.modeltree.grid(row=0, column=0, sticky="nsew")
        self.modeltree.tag_configure("missing", foreground="#888888")
        mscroll = ttk.Scrollbar(models, command=self.modeltree.yview)
        self.modeltree.configure(yscrollcommand=mscroll.set)
        mscroll.grid(row=0, column=1, sticky="ns")
        setup = ttk.Frame(models)
        setup.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(setup, text="Refresh status", command=self.refresh).pack(side="left")
        ttk.Button(setup, text="Install Ollama engine", command=self._install_engine).pack(side="left", padx=7)
        ttk.Button(setup, text="Install selected model", command=self._install_models).pack(side="left")
        actions = ttk.Frame(top)
        actions.pack(fill="x", pady=(10, 5))
        self.run_button = ttk.Button(actions, text="Start batch translation", command=lambda: self._run(False))
        self.run_button.pack(side="left")
        self.plan_button = ttk.Button(actions, text="Preview jobs", command=lambda: self._run(True))
        self.plan_button.pack(side="left", padx=6)
        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left")
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=160)
        self.progress.pack(side="right")
        ttk.Label(top, textvariable=self.status_var).pack(anchor="w", pady=(3, 3))
        self.logbox = scrolledtext.ScrolledText(top, height=9, wrap="word", state="disabled")
        self.logbox.pack(fill="both", expand=True)

    def _choose_files(self):
        chosen = filedialog.askopenfilenames(title="Select files", filetypes=FILTERS)
        if chosen:
            existing = {str(x).casefold() for x in self.files}
            for item in chosen:
                path = Path(item)
                if str(path).casefold() not in existing:
                    self.files.append(path)
                    existing.add(str(path).casefold())
            self._sync_files()

    def _remove_selected(self):
        selected = set(self.filelist.curselection())
        self.files = [item for i, item in enumerate(self.files) if i not in selected]
        self._sync_files()

    def _clear_files(self):
        self.files = []
        self._sync_files()

    def _sync_files(self):
        self.filelist.delete(0, tk.END)
        for file in self.files:
            self.filelist.insert(tk.END, str(file))

    def _choose_folder(self):
        path = filedialog.askdirectory(title="Input folder")
        if path:
            self.folder_var.set(path)

    def _choose_output(self):
        path = filedialog.askdirectory(title="Existing output folder")
        if path:
            self.output_var.set(path)

    def refresh(self):
        selected = set(self.modeltree.selection())
        self.modeltree.delete(*self.modeltree.get_children())
        rows = status_rows()
        for model, ready in rows:
            self.modeltree.insert(
                "", "end", iid=model.id,
                values=(f"{model.id} | {model.title}", "Ready" if ready else "Missing", model.note),
                tags=() if ready else ("missing",),
            )
        keep = [name for name in selected if name in BY_ID]
        if keep:
            self.modeltree.selection_set(*keep)
        else:
            default_id = "hy30" if any(model.id == "hy30" and ready for model, ready in rows) else next(
                (model.id for model, ready in rows if ready), "hy30"
            )
            self.modeltree.selection_set(default_id)
        self.status_var.set(
            "Engine installed" if OLLAMA_EXE.is_file() else
            "Engine missing: select Install Ollama engine"
        )

    def _build_args(self, preview: bool) -> list[str]:
        chosen = list(self.modeltree.selection())
        if not chosen:
            raise ValueError("Select at least one model.")
        if not self.files and not self.folder_var.get().strip():
            raise ValueError("Add input files or select a folder.")
        batch = int(self.batch_var.get())
        width = int(self.width_var.get())
        command = [
            sys.executable, "-B", str(HERE / "pipeline.py"),
            "--models", ",".join(chosen), "--source-language", self.language_var.get(),
            "--batch-size", str(batch), "--width", str(width),
        ]
        for path in self.files:
            command += ["--input", str(path)]
        if self.folder_var.get().strip():
            command += ["--folder", self.folder_var.get().strip()]
        if self.recursive_var.get():
            command += ["--recursive"]
        if self.output_var.get().strip():
            command += ["--output-dir", self.output_var.get().strip()]
        if preview:
            command += ["--dry-run"]
        return command

    def _log(self, line):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", str(line).rstrip() + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def _launch(self, argv, label):
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Already running", "Finish or stop the current job first.")
            return
        self.run_button.configure(state="disabled")
        self.plan_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.start(12)
        self.status_var.set(label)
        self._log("[START] " + label)
        thread = threading.Thread(target=self._worker, args=(argv,), daemon=True)
        thread.start()

    def _run(self, preview):
        try:
            args = self._build_args(preview)
        except (ValueError, OSError, tk.TclError) as error:
            messagebox.showerror("Input error", str(error))
            return
        self._launch(args, "Preview" if preview else "Batch translation")

    def _install_engine(self):
        self._launch([sys.executable, "-B", str(HERE / "install.py"), "--engine"], "Install engine")

    def _install_models(self):
        chosen = list(self.modeltree.selection())
        if not chosen:
            messagebox.showerror("Missing selection", "Select a model to install.")
            return
        if len(chosen) > 1:
            messagebox.showinfo("One at a time", "Select one model per download to control disk usage.")
            return
        self._launch(
            [sys.executable, "-B", str(HERE / "install.py"), "--model", chosen[0]],
            "Install model " + chosen[0],
        )

    def _worker(self, argv):
        try:
            kwargs = {
                "cwd": str(HERE), "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                "text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1,
            }
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(argv, **kwargs)
            self.proc = proc
            if proc.stdout:
                for line in proc.stdout:
                    self.events.put(("line", line.rstrip()))
            self.events.put(("done", proc.wait()))
        except (OSError, ValueError) as error:
            self.events.put(("error", str(error)))

    def _stop(self):
        proc = self.proc
        if proc and proc.poll() is None:
            self._log("[STOP] Terminating this job and its child processes.")
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True, timeout=12, check=False,
                )
            else:
                proc.terminate()
        self.stop_button.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "line":
                    self._log(value)
                else:
                    self.progress.stop()
                    self.run_button.configure(state="normal")
                    self.plan_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self._log(f"[EXIT] {event}: {value}")
                    self.proc = None
                    if event == "done":
                        self.refresh()
                    self.status_var.set(
                        "Finished" if event == "done" and value == 0 else "Failed / stopped"
                    )
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(120, self._poll)

    def _close(self):
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("Job running", "Stop the current job and exit?"):
                return
            self._stop()
        self.destroy()

def main():
    if "--self-test" in sys.argv:
        from model_catalog import LANGUAGES
        assert len(MODELS) >= 10
        assert "Korean" in LANGUAGES
        assert HERE.joinpath("pipeline.py").is_file()
        print("GUI_HEADLESS_SELFTEST_OK", len(MODELS))
        return 0
    Studio().mainloop()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
