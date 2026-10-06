"""Optional Tk desktop demo: python -m yolo_pipeline.demo --config config.yaml."""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import for_task, load_config
from .demo_backend import FLOOR, TASK, TASKS, leaderboard, training_history, unique_id
from .models import State
from .store import StateStore


def find_labelme() -> str:
    found = shutil.which("labelme")
    if found:
        return found
    candidates = [Path(sys.executable).parent / "labelme"]
    if len(Path(sys.executable).parents) >= 3:
        candidates.append(Path(sys.executable).parents[2] / "labelme" / "bin" / "labelme")
    return next((str(path) for path in candidates if path.is_file()), "")


class DemoApp:
    def __init__(self, root: tk.Tk, config: dict, labelme: str = ""):
        self.root, self.config = root, config
        Path(config["paths"]["state_db"]).parent.mkdir(parents=True, exist_ok=True)
        self.store = StateStore(config["paths"]["state_db"])
        self.work = Path(config["paths"]["state_db"]).parent / "demo"
        self.labelme = labelme or find_labelme()
        self.labelme_processes = []
        self.process = None
        self.events = queue.Queue()
        self.batch_id = None
        self.runs, self.board_rows, self.model_choices = {}, {}, {}
        self.batch_names = {}
        self.task_choices = {label: task for task, label in TASKS.items() if task in config.get("tasks", {})}
        self.task_selection = tk.StringVar(value=next(iter(self.task_choices)))
        self.stop_file = None
        self.stop_requested = False
        self.job_action = self.job_error = self.last_output = self.history_key = None
        self.status = tk.StringVar(value="Choose an image folder to begin.")
        self.batch_text = tk.StringVar(value="No batch selected")
        self.summary = tk.StringVar(value="Import images → label in Labelme → mark done → train and evaluate")
        self.baseline, self.infer_source, self.infer_model = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.infer_status = tk.StringVar(value="Choose a folder and a model. Results are saved separately.")
        root.title("YOLO MLOps · Demo")
        width = min(1240, root.winfo_screenwidth() - 60)
        height = min(850, root.winfo_screenheight() - 100)
        root.geometry(f"{width}x{height}+20+20")
        root.minsize(min(1000, width), min(640, height))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self.refresh()
        root.after(300, self.poll)

    def _build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f5f9")
        style.configure("TLabel", background="#f3f5f9", foreground="#1c2a3a", font=("sans-serif", 10))
        style.configure("Title.TLabel", font=("sans-serif", 24, "bold"))
        style.configure("Heading.TLabel", font=("sans-serif", 12, "bold"))
        style.configure("TButton", padding=(12, 8), font=("sans-serif", 10))
        style.configure("Accent.TButton", background="#137d87", foreground="white")
        style.map("Accent.TButton", background=[("active", "#10656d"), ("disabled", "#aab8c0")])
        style.configure("Treeview", rowheight=30, font=("sans-serif", 10))
        style.configure("Treeview.Heading", font=("sans-serif", 10, "bold"))
        shell = ttk.Frame(self.root, padding=20)
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x")
        ttk.Label(header, text="MLOps Pipeline", style="Title.TLabel").pack(side="left")
        self.task_combo = ttk.Combobox(header, textvariable=self.task_selection,
                                      values=list(self.task_choices), state="readonly", width=24)
        self.task_combo.pack(side="right")
        self.task_combo.bind("<<ComboboxSelected>>", self.change_task)
        ttk.Label(header, text="Task  ").pack(side="right")
        ttk.Label(shell, textvariable=self.summary, wraplength=1120).pack(anchor="w", pady=(8, 12))
        actions = ttk.Frame(shell)
        actions.pack(fill="x")
        self.import_button = ttk.Button(actions, text="1  Select batch folder", style="Accent.TButton", command=self.select_batch)
        self.import_button.pack(side="left", padx=(0, 8))
        self.label_button = ttk.Button(actions, text="2  Open Labelme", command=self.open_labelme)
        self.label_button.pack(side="left", padx=(0, 8))
        self.train_button = ttk.Button(actions, text="3  Mark done & train", command=self.train)
        self.train_button.pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Inference", command=lambda: self.tabs.select(self.inference_tab)).pack(side="right")
        ttk.Label(shell, textvariable=self.batch_text, wraplength=1140).pack(anchor="w", pady=(12, 5))
        ttk.Label(shell, textvariable=self.status, style="Heading.TLabel", wraplength=1140).pack(anchor="w", pady=(0, 6))
        self.progress = ttk.Progressbar(shell, mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(0, 12))
        body = ttk.Panedwindow(shell, orient="horizontal")
        body.pack(fill="both", expand=True)
        sidebar = ttk.Frame(body, padding=(0, 0, 12, 0))
        body.add(sidebar, weight=1)
        ttk.Label(sidebar, text="Batches", style="Heading.TLabel").pack(anchor="w", pady=(0, 8))
        self.batches = ttk.Treeview(sidebar, columns=("batch", "state"), show="headings", selectmode="browse", height=10)
        for column, title, width in (("batch", "Batch", 160), ("state", "Stage", 135)):
            self.batches.heading(column, text=title)
            self.batches.column(column, width=width, stretch=column == "batch")
        self.batches.pack(fill="both", expand=True)
        self.batches.bind("<<TreeviewSelect>>", self.select_existing)
        ttk.Button(sidebar, text="Open batch folder", command=self.open_batch).pack(fill="x", pady=(8, 0))
        self.tabs = ttk.Notebook(body)
        body.add(self.tabs, weight=4)
        training_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(training_tab, text="Training & evaluation")
        ttk.Label(training_tab, textvariable=self.baseline, wraplength=720).pack(anchor="w", pady=(0, 8))
        stop_row = ttk.Frame(training_tab)
        stop_row.pack(fill="x", pady=(0, 8))
        self.stop_button = ttk.Button(stop_row, text="Stop training & evaluate", command=self.stop_training, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Label(stop_row, text="Finishes the current epoch and uses the saved model.", wraplength=360).pack(side="left", padx=8)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(7, 3.3), dpi=100, facecolor="#f3f5f9")
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=training_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.draw_history([])
        self.details = tk.Text(training_tab, height=5, wrap="word", relief="flat", padx=10, pady=8,
                               background="white", foreground="#1c2a3a", font=("sans-serif", 10), state="disabled")
        self.details.pack(fill="x", pady=(8, 0))
        board_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(board_tab, text="Leaderboard")
        ttk.Label(board_tab, text="Current task, ranked by F1 · segmentation uses mask precision/recall",
                  wraplength=700).pack(anchor="w", pady=(0, 8))
        columns = ("rank", "batch", "precision", "recall", "f1", "status")
        self.board = ttk.Treeview(board_tab, columns=columns, show="headings", selectmode="browse")
        for column, width in zip(columns, (45, 210, 85, 85, 75, 100)):
            self.board.heading(column, text=column.title())
            self.board.column(column, width=width, stretch=column == "batch")
        self.board.pack(fill="both", expand=True)
        buttons = ttk.Frame(board_tab)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="View evaluation report", command=self.view_report).pack(side="left")
        ttk.Button(buttons, text="Use model for inference", command=self.use_board_model).pack(side="left", padx=8)
        self.inference_tab = ttk.Frame(self.tabs, padding=16)
        self.tabs.add(self.inference_tab, text="Inference")
        ttk.Label(self.inference_tab, text="Run a model on an image folder", style="Heading.TLabel").pack(anchor="w", pady=(0, 12))
        ttk.Label(self.inference_tab, text="Input folder").pack(anchor="w")
        inputs = ttk.Frame(self.inference_tab)
        inputs.pack(fill="x", pady=(4, 12))
        ttk.Entry(inputs, textvariable=self.infer_source).pack(side="left", fill="x", expand=True)
        ttk.Button(inputs, text="Browse…", command=self.choose_inference_folder).pack(side="left", padx=(8, 0))
        ttk.Label(self.inference_tab, text="Model — choose from completed runs or browse for weights").pack(anchor="w")
        models = ttk.Frame(self.inference_tab)
        models.pack(fill="x", pady=(4, 12))
        self.model_combo = ttk.Combobox(models, textvariable=self.infer_model, state="readonly")
        self.model_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(models, text="Browse .pt…", command=self.choose_model).pack(side="left", padx=(8, 0))
        ttk.Label(self.inference_tab, text="Runs on original images. Saves annotated images and labels; "
                  "panel cropping requires annotations and is not applied here.", wraplength=650).pack(anchor="w", pady=(0, 16))
        self.infer_button = ttk.Button(self.inference_tab, text="Run inference", style="Accent.TButton", command=self.infer)
        self.infer_button.pack(anchor="w")
        ttk.Label(self.inference_tab, textvariable=self.infer_status, wraplength=650).pack(anchor="w", pady=16)
        ttk.Button(self.inference_tab, text="Open results folder", command=self.open_results).pack(anchor="w")
        log_tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(log_tab, text="Activity log")
        self.log = tk.Text(log_tab, wrap="word", state="disabled", background="#142231", foreground="#d8e4ee",
                           font=("monospace", 10), relief="flat")
        self.log.pack(fill="both", expand=True)

    def note(self, message):
        self.log.configure(state="normal")
        self.log.insert("end", message.rstrip() + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 1500:
            self.log.delete("1.0", "300.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def select_batch(self):
        folder = filedialog.askdirectory(title="Select a finished incoming image batch")
        if folder:
            self.start_job("import", source=folder)

    def selected_task(self):
        return self.task_choices[self.task_selection.get()]

    def change_task(self, _event=None):
        if self.process is not None:
            return
        self.batch_id = self.history_key = None
        self.model_choices.clear()
        self.infer_model.set("")
        self.batch_text.set("No batch selected")
        self.status.set(f"{self.task_selection.get()}: select a batch folder to begin.")
        self.summary.set("Import images → label in Labelme → mark done → train and evaluate")
        self.progress["value"] = 0
        self.draw_history([])
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.configure(state="disabled")
        self.refresh()

    def start_job(self, action, **fields):
        if self.process is not None:
            return
        folder = self.work / "jobs" / unique_id()
        folder.mkdir(parents=True)
        fields.setdefault("task", self.selected_task())
        self.stop_file = folder / "STOP_TRAINING" if action == "train" else None
        self.stop_requested = False
        if self.stop_file:
            fields["stop_file"] = str(self.stop_file)
        path = folder / "job.json"
        path.write_text(json.dumps({"action": action, "config": self.config, **fields}, indent=2), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH")]))
        try:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "yolo_pipeline.demo_worker", "--job", str(path)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                env=env, start_new_session=os.name != "nt")
        except OSError as error:
            messagebox.showerror("Could not start job", str(error))
            return
        self.process, self.job_action, self.job_error = process, action, None
        self.progress["value"] = 0
        self.status.set({"import": "Copying selected batch…", "train": "Starting pipeline…", "infer": "Starting inference…"}[action])
        self.note(f"Started {action}. Full log: {folder / 'worker.log'}")
        self.set_buttons()

        def read_output():
            with (folder / "worker.log").open("w", encoding="utf-8") as logfile:
                for line in process.stdout:
                    logfile.write(line)
                    logfile.flush()
                    if line.startswith("DEMO_EVENT "):
                        try:
                            self.events.put(json.loads(line[len("DEMO_EVENT "):]))
                            continue
                        except ValueError:
                            pass
                    self.events.put({"kind": "log", "message": line})
            process.stdout.close()
            self.events.put({"kind": "exited", "code": process.wait()})

        threading.Thread(target=read_output, daemon=True).start()

    def open_labelme(self):
        run = self.current_run()
        if not run:
            return
        if not self.labelme or not Path(self.labelme).is_file():
            self.labelme = filedialog.askopenfilename(title="Locate the Labelme executable (usually environment/bin/labelme)")
        if not self.labelme:
            self.status.set("Batch ready. Locate Labelme with Open Labelme, or label the batch externally.")
            return
        try:
            log_dir = self.work / "labelme"
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / f"{run.batch_id}.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen([self.labelme, run.batch_path], stdout=log, stderr=subprocess.STDOUT)
            self.labelme_processes.append((process, run.batch_id))
            self.status.set("Labelme opened. Save each annotation, close Labelme, then mark done.")
        except OSError as error:
            messagebox.showerror("Could not open Labelme", str(error))

    def train(self):
        run = self.current_run()
        if not run:
            return
        if any(p.poll() is None and batch == run.batch_id for p, batch in self.labelme_processes):
            messagebox.showinfo("Finish labeling", "Save annotations and close this batch's Labelme window before training.")
            return
        self.tabs.select(0)
        self.history_key = None
        self.start_job("train", batch_id=run.batch_id)

    def stop_training(self):
        run = self.current_run()
        if self.job_action != "train" or not self.process or not self.stop_file or not run or run.state != State.TRAINING:
            return
        try:
            self.stop_file.touch()
        except OSError as error:
            messagebox.showerror("Could not request stop", str(error))
            return
        self.stop_requested = True
        self.status.set("Stop requested — finishing this epoch, then evaluating the saved model.")
        self.note("Training stop requested. The current epoch and checkpoint save will finish before evaluation.")
        self.set_buttons()

    def choose_inference_folder(self):
        folder = filedialog.askdirectory(title="Choose images for inference")
        if folder:
            self.infer_source.set(folder)

    def choose_model(self):
        path = filedialog.askopenfilename(title="Choose model weights", filetypes=[("YOLO weights", "*.pt")])
        if path:
            label = f"Local: {path}"
            self.model_choices[label] = path
            self.model_combo["values"] = list(self.model_choices)
            self.infer_model.set(label)

    def infer(self):
        model = self.model_choices.get(self.infer_model.get())
        if not model or not self.infer_source.get():
            messagebox.showinfo("Select inputs", "Choose an image folder and model first.")
            return
        output = self.work / "inference" / unique_id()
        self.start_job("infer", source=self.infer_source.get(), model=model, output=str(output))

    def current_run(self):
        return next((run for run in self.runs.values() if run.batch_id == self.batch_id), None)

    def batch_title(self, run):
        if run.id not in self.batch_names:
            try:
                info = (Path(run.batch_path) / "BATCH_INFO.txt").read_text(encoding="utf-8")
                name = next(line.partition("=")[2] for line in info.splitlines() if line.startswith("source_name="))
                self.batch_names[run.id] = f"{name} · #{run.id}"
            except (OSError, StopIteration):
                self.batch_names[run.id] = run.batch_id
        return self.batch_names[run.id]

    def model_title(self, run):
        return f"{self.batch_title(run)} · {run.state.value.title()}"

    def select_existing(self, _event=None):
        selection = self.batches.selection()
        if not selection or self.process is not None:
            return
        run = self.runs.get(selection[0])
        if run:
            self.batch_id, self.history_key = run.batch_id, None
            self.render_run(run)
            self.set_buttons()

    def set_buttons(self):
        busy, run = self.process is not None, self.current_run()
        self.import_button.configure(state="disabled" if busy else "normal")
        self.infer_button.configure(state="disabled" if busy else "normal")
        self.task_combo.configure(state="disabled" if busy else "readonly")
        can_stop = busy and self.job_action == "train" and run and run.state == State.TRAINING and not self.stop_requested
        self.stop_button.configure(state="normal" if can_stop else "disabled")
        self.label_button.configure(state="normal" if not busy and run and run.state == State.LABELING else "disabled")
        trainable = run and run.task in TASKS and run.state not in {State.APPROVED, State.REJECTED}
        self.train_button.configure(state="normal" if not busy and trainable else "disabled",
                                    text="3  Mark done & train" if not run or run.state == State.LABELING else "Resume / retry")

    def refresh(self):
        runs = [run for run in self.store.list_all() if run.task == self.selected_task()]
        self.runs = {str(run.id): run for run in runs}
        for item in self.batches.get_children():
            if item not in self.runs:
                self.batches.delete(item)
        for run in reversed(runs):
            values = (self.batch_title(run), run.state.value.replace("_", " ").title())
            if self.batches.exists(str(run.id)):
                self.batches.item(str(run.id), values=values)
            else:
                self.batches.insert("", "end", iid=str(run.id), values=values)
        rows = leaderboard(self.store, self.selected_task())
        self.model_choices = {key: value for key, value in self.model_choices.items() if key.startswith("Local: ")}
        self.board_rows = {str(row["run"].id): row for row in rows}
        selected = self.board.selection()
        children = self.board.get_children()
        if children:
            self.board.delete(*children)
        for rank, row in enumerate(rows, 1):
            run = row["run"]
            metrics = [f"{row[key]:.3f}" if row[key] is not None and not row["error"] else "—"
                       for key in ("precision", "recall", "f1")]
            self.board.insert("", "end", iid=str(run.id), values=(rank, self.batch_title(run), *metrics,
                              "Report error" if row["error"] else run.state.value.title()))
            if run.model_path and Path(run.model_path).is_file():
                self.model_choices[self.model_title(run)] = run.model_path
        if selected and self.board.exists(selected[0]):
            self.board.selection_set(selected[0])
        self.model_combo["values"] = list(self.model_choices)
        approved = [row for row in rows if row["run"].state == State.APPROVED and row["run"].batch_id != self.batch_id]
        if any(row["error"] for row in approved):
            self.baseline.set("Baseline report unavailable. Restore it before comparing another model.")
        elif approved:
            best = approved[0]
            self.baseline.set(f"Approval target: precision ≥ {max(FLOOR, best['precision']):.3f}, "
                              f"recall ≥ {max(FLOOR, best['recall']):.3f} · best approved model by F1")
        else:
            self.baseline.set("No approved baseline yet · precision ≥ 0.200 and recall ≥ 0.200 required")
        run = self.current_run()
        if run:
            self.render_run(run)
        self.set_buttons()

    def render_run(self, run):
        self.batch_text.set(f"Batch: {run.batch_id}")
        if self.job_action not in {"import", "infer"}:
            self.status.set("Stop requested — finishing this epoch, then evaluating the saved model."
                            if self.stop_requested and self.job_action == "train" and run.state == State.TRAINING
                            else run.state.value.replace("_", " ").title())
        history = training_history(self.config, run)
        key = (run.batch_id, run.training_attempt, len(history), str(history[-1:]))
        if key != self.history_key:
            self.draw_history(history)
            self.history_key = key
        if history and self.job_action not in {"import", "infer"}:
            epoch = int(history[-1]["epoch"])
            epochs = for_task(self.config, run.task or TASK)["training"]["epochs"]
            self.summary.set(f"Training epoch {epoch}/{epochs} · losses update after each completed epoch")
            self.progress["value"] = min(100, epoch / max(1, epochs) * 100)
        else:
            self.summary.set("Import images → label in Labelme → mark done → train and evaluate")
        row = self.board_rows.get(str(run.id))
        text = f"Stage: {run.state.value}\nLabeling folder: {run.batch_path}"
        if run.error:
            text += f"\nError: {run.error}"
        if row:
            if row["error"]:
                text += f"\nReport error: {row['error']}"
            else:
                text += (f"\nPrecision {row['precision']:.4f}  ·  Recall {row['recall']:.4f}  ·  F1 {row['f1']:.4f}"
                         f"\nEvaluation: {run.evaluation_path}")
                if "map50" in row["report"]:
                    text += f"\n{row['report'].get('metric_type', '').title()} mAP50: {row['report']['map50']:.4f}"
                if row["report"].get("evaluation_source") == "batch_validation":
                    text += " · evaluated on this dataset's validation split"
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def draw_history(self, rows):
        self.axes.clear()
        self.axes.set_title("Training and validation loss", loc="left", fontsize=11)
        self.axes.set_xlabel("Epoch")
        self.axes.set_ylabel("Loss")
        self.axes.grid(alpha=0.15)
        keys = sorted({key for row in rows for key in row if "loss" in key})
        for key in keys:
            points = [row for row in rows if key in row]
            self.axes.plot([row["epoch"] for row in points], [row[key] for row in points],
                           label=key, linestyle="--" if key.startswith("val/") else "-")
        if keys:
            self.axes.legend(fontsize=7, ncol=3)
        else:
            self.axes.text(0.5, 0.5, "Loss curves appear after the first epoch", ha="center", transform=self.axes.transAxes,
                           color="#718096")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def poll(self):
        try:
            for _ in range(300):
                try:
                    event = self.events.get_nowait()
                except queue.Empty:
                    break
                kind = event["kind"]
                if kind == "log":
                    self.note(event["message"])
                elif kind == "progress":
                    self.status.set(event["message"])
                    self.progress["value"] = event["current"] / max(1, event["total"]) * 100
                elif kind == "imported":
                    self.batch_id = event["batch_id"]
                    self.refresh()
                    run = self.current_run()
                    self.batches.selection_set(str(run.id))
                    self.batches.see(str(run.id))
                    self.open_labelme()
                elif kind == "state":
                    self.status.set(event["state"].replace("_", " ").title())
                    self.note(f"Stage: {event['state']}")
                elif kind == "inferred":
                    self.last_output = event["output"]
                    self.infer_status.set(f"Finished: {event['processed']} images, {event['errors']} errors.\n{self.last_output}")
                elif kind == "error":
                    self.job_error = event["message"]
                    self.note(event["message"])
                elif kind == "exited":
                    action = self.job_action
                    self.process = self.job_action = None
                    self.refresh()
                    if event["code"] != 0:
                        messagebox.showerror("Job failed", self.job_error or "Worker stopped unexpectedly. Check the activity log.")
                    elif action == "train":
                        self.tabs.select(1)
                    elif action == "infer":
                        self.status.set("Inference complete")
            self.refresh()
            still_open = []
            for process, batch in self.labelme_processes:
                code = process.poll()
                if code is None:
                    still_open.append((process, batch))
                elif code:
                    self.note(f"Labelme exited with code {code}. Log: {self.work / 'labelme' / (batch + '.log')}")
                    messagebox.showerror("Labelme stopped", "Labelme could not run successfully. See the Activity log for its log path.")
            self.labelme_processes = still_open
        except Exception as error:
            self.note(f"Display update failed: {error}")
        self.root.after(1000, self.poll)

    def view_report(self):
        selection = self.board.selection()
        if not selection:
            return
        row = self.board_rows[selection[0]]
        window = tk.Toplevel(self.root)
        window.title(f"Evaluation · {row['run'].batch_id}")
        text = tk.Text(window, wrap="word", width=100, height=32)
        text.pack(fill="both", expand=True)
        text.insert("1.0", row["error"] or json.dumps(row["report"], indent=2))
        text.configure(state="disabled")

    def use_board_model(self):
        selection = self.board.selection()
        if not selection:
            return
        run = self.board_rows[selection[0]]["run"]
        if not run.model_path or not Path(run.model_path).is_file():
            messagebox.showerror("Model unavailable", "The recorded model file is missing. Browse for another model.")
            return
        self.infer_model.set(self.model_title(run))
        self.tabs.select(self.inference_tab)

    def open_path(self, path):
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except OSError as error:
            messagebox.showerror("Cannot open folder", str(error))

    def open_batch(self):
        run = self.current_run()
        if run:
            self.open_path(run.batch_path)

    def open_results(self):
        if self.last_output:
            self.open_path(self.last_output)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("A job is running", "Stop the running job and close? Interrupted training restarts at its stage boundary."):
                return
            if os.name == "nt":
                self.process.terminate()
            else:
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    self.process.kill()
                else:
                    os.killpg(self.process.pid, signal.SIGKILL)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Optional local YOLO demo GUI (no watcher)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--labelme", help="Path to a Labelme executable, including one in another environment")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if not any(task in config.get("tasks", {}) for task in TASKS):
            raise ValueError("Configure at least one supported task: crack_detection, panel_segmentation, or hole_detection.")
        root = tk.Tk()
        DemoApp(root, config, args.labelme or "")
        root.mainloop()
    except (tk.TclError, ImportError, ValueError, OSError) as error:
        parser.exit(1, f"Cannot start the demo: {error}\nUse a desktop session with Tk and matplotlib installed.\n")


if __name__ == "__main__":
    main()
