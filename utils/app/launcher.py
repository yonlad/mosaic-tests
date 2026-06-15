#!/usr/bin/env python3
"""
Blend Review Tool — launcher GUI.

Presents a simple window where the user picks a blend/bucket and
launches one of the two review galleries (S3 images or DynamoDB blends).
The gallery opens in the default browser; the exported JSON manifest
lands in ~/Downloads for the user to email back.
"""

import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import webbrowser

from config import BLENDS, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
from review import run_review
from review_blends import run_review_blends

BLEND_OPTIONS = {
    f"Blend {num}  —  {cfg['bucket']}": num
    for num, cfg in sorted(BLENDS.items())
}


class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Blend Review Tool")
        self.root.resizable(False, False)
        self._msg_queue = queue.Queue()
        self._running = False

        self._build_ui()
        self._check_credentials()
        self._grab_focus()
        self._poll_queue()

    def _grab_focus(self):
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 14, "pady": 6}

        title = tk.Label(
            self.root, text="Blend Review Tool",
            font=("Helvetica Neue", 18, "bold"),
        )
        title.pack(pady=(18, 2))

        subtitle = tk.Label(
            self.root,
            text="Flag bad blends / images for cleanup",
            font=("Helvetica Neue", 12), fg="#666666",
        )
        subtitle.pack(pady=(0, 14))

        # Blend selector — tk.OptionMenu uses native macOS popup (instant clicks)
        selector_frame = tk.Frame(self.root)
        selector_frame.pack(**pad)

        tk.Label(selector_frame, text="Blend:", font=("Helvetica Neue", 12)).pack(
            side=tk.LEFT, padx=(0, 8),
        )

        option_labels = list(BLEND_OPTIONS.keys())
        self.blend_var = tk.StringVar(value=option_labels[0])
        self.blend_dropdown = tk.OptionMenu(
            selector_frame, self.blend_var, *option_labels,
        )
        self.blend_dropdown.config(
            font=("Helvetica Neue", 12), width=38,
        )
        self.blend_dropdown["menu"].config(font=("Helvetica Neue", 12))
        self.blend_dropdown.pack(side=tk.LEFT)

        # Action buttons — ttk.Button has better macOS click handling than tk.Button
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=12)

        style = ttk.Style()
        style.configure(
            "Big.TButton",
            font=("Helvetica Neue", 13, "bold"),
            padding=(24, 14),
        )

        self.btn_images = ttk.Button(
            btn_frame, text="Review S3 Images",
            style="Big.TButton",
            command=self._on_review_images,
        )
        self.btn_images.pack(side=tk.LEFT, padx=8)

        self.btn_blends = ttk.Button(
            btn_frame, text="Review Blends",
            style="Big.TButton",
            command=self._on_review_blends,
        )
        self.btn_blends.pack(side=tk.LEFT, padx=8)

        # Progress bar
        self.progress = ttk.Progressbar(
            self.root, mode="indeterminate", length=480,
        )
        self.progress.pack(pady=(8, 4))

        # Log area
        self.log = scrolledtext.ScrolledText(
            self.root, height=12, width=64,
            font=("Menlo", 11), state=tk.DISABLED,
            bg="#1a1a2e", fg="#e0e0e0",
            insertbackground="#e0e0e0",
        )
        self.log.pack(padx=14, pady=(4, 14))

        reminder = tk.Label(
            self.root,
            text=(
                "After reviewing, click 'Export Deletion Manifest' in the browser.\n"
                "The JSON file downloads to your Downloads folder — email it to Yonatan."
            ),
            font=("Helvetica Neue", 11), fg="#888888", justify=tk.CENTER,
        )
        reminder.pack(pady=(0, 14))

    # ------------------------------------------------------------------
    # Credentials check
    # ------------------------------------------------------------------

    def _check_credentials(self):
        if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
            messagebox.showerror(
                "Missing AWS Credentials",
                "Could not find AWS credentials.\n\n"
                "Make sure the .env file is in the same folder as this app "
                "and contains AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.",
            )
            self.btn_images.config(state=tk.DISABLED)
            self.btn_blends.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Blend number from dropdown
    # ------------------------------------------------------------------

    def _selected_blend(self) -> int:
        return BLEND_OPTIONS[self.blend_var.get()]

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_review_images(self):
        if self._running:
            return
        self._start_task("images")

    def _on_review_blends(self):
        if self._running:
            return
        self._start_task("blends")

    def _start_task(self, task_type: str):
        self._running = True
        self.btn_images.config(state=tk.DISABLED)
        self.btn_blends.config(state=tk.DISABLED)
        self.blend_dropdown.config(state=tk.DISABLED)
        self.progress.start(15)
        self._clear_log()

        blend_num = self._selected_blend()
        t = threading.Thread(
            target=self._worker, args=(task_type, blend_num), daemon=True,
        )
        t.start()

    def _worker(self, task_type: str, blend_num: int):
        try:
            if task_type == "images":
                path = run_review(blend_num, on_progress=self._enqueue_msg)
            else:
                path = run_review_blends(blend_num, on_progress=self._enqueue_msg)

            if path:
                self._enqueue_msg(f"\nOpening gallery in browser...")
                webbrowser.open(path.resolve().as_uri())
                self._enqueue_msg("Done! Flag items in the browser, then click Export.")
            else:
                self._enqueue_msg("No records found — nothing to review.")
        except Exception as e:
            self._enqueue_msg(f"\nERROR: {e}")
            self._enqueue_signal("error", str(e))
        finally:
            self._enqueue_signal("done", "")

    # ------------------------------------------------------------------
    # Thread-safe message passing
    # ------------------------------------------------------------------

    def _enqueue_msg(self, msg: str):
        self._msg_queue.put(("log", msg))

    def _enqueue_signal(self, signal: str, detail: str):
        self._msg_queue.put((signal, detail))

    def _poll_queue(self):
        while True:
            try:
                kind, payload = self._msg_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self._append_log(payload)
            elif kind == "done":
                self._finish_task()
            elif kind == "error":
                messagebox.showerror("Error", payload)

        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------

    def _append_log(self, msg: str):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log.config(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.config(state=tk.DISABLED)

    def _finish_task(self):
        self._running = False
        self.progress.stop()
        self.btn_images.config(state=tk.NORMAL)
        self.btn_blends.config(state=tk.NORMAL)
        self.blend_dropdown.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    root.geometry("540x580")
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
