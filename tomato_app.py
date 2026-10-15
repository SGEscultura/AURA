"""
Tomato Ripeness Checker - desktop application.

A single window with buttons: start the webcam, open a photo, or open a video.
Every tomato in view gets a box, an ID, and a NOT YET / ALMOST / RIPE label.
This file is the entry point that gets packaged into the .exe (see README / build notes).

Run from PyCharm or a terminal:
    python tomato_app.py

It needs tomato_ripeness.py in the same folder (that file holds the detection code).
Logs, snapshots, and error_log.txt are written next to this file (or next to the .exe).
"""

import base64
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

import tomato_ripeness as t

APP_TITLE = "Tomato Ripeness Checker"
MASK_WIDTH = 320

# Where files are saved: next to the .exe when packaged, next to this file otherwise
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
t.LOG_FILE = os.path.join(BASE_DIR, "ripeness_log.csv")
t.SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")


def log_error(text):
    """Writes crashes to error_log.txt, because a windowed .exe has no console."""
    try:
        with open(os.path.join(BASE_DIR, "error_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now().isoformat(timespec='seconds')} ---\n{text}\n")
    except OSError:
        pass


def read_image(path):
    """Reads an image even when the path has non-English characters."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image(path, image):
    ok, buf = cv2.imencode(".jpg", image)
    if ok:
        buf.tofile(path)
    return ok


def to_photo(image):
    """Converts an OpenCV (BGR) image into a Tkinter PhotoImage via PNG data."""
    ok, buf = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])  # level 1 = fast
    if not ok:
        return None
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()).decode("ascii"))


# ------------------------------------------------------------------
# Engine: everything except the window (frame in, drawn frame out)
# ------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.tracker = t.Tracker()

    def reset(self):
        self.tracker.reset()

    def process(self, frame, s_min, v_min, ripe_hue, white_balance, show_hsv, window):
        """Returns (display_image, mask_image, pairs, counts). pairs = tomatoes seen this frame."""
        self.tracker.window = window
        processed = t.gray_world(frame) if white_balance else frame
        results, masks = t.analyze(processed, s_min, v_min, ripe_hue)
        pairs = self.tracker.update(results)

        display = frame.copy()
        counts = Counter()
        for track, res in pairs:
            label = track.label()
            counts[label] += 1
            shares = track.mean_shares()
            score = t.ripeness_score(shares[0], shares[1]) * 100
            x, y, w, h = res["box"]
            color = t.COLORS[label]
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 3)
            tag = f"#{track.id} {label} {score:.0f}%"
            if show_hsv:
                tag += f"  H{res['hue']} S{res['sat']}"
            pos = (x, max(28, y - 10))
            cv2.putText(display, tag, pos, t.FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(display, tag, pos, t.FONT, 0.7, color, 2, cv2.LINE_AA)
        # Tomatoes that vanished a moment ago stay on screen with a thin box
        for track in self.tracker.lost():
            label = track.label()
            counts[label] += 1
            shares = track.mean_shares()
            score = t.ripeness_score(shares[0], shares[1]) * 100
            x, y, w, h = track.box
            color = t.COLORS[label]
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 1)
            tag = f"#{track.id} {label} {score:.0f}% (lost)"
            pos = (x, max(28, y - 10))
            cv2.putText(display, tag, pos, t.FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, tag, pos, t.FONT, 0.6, color, 1, cv2.LINE_AA)
        return display, t.mask_view(masks, results), pairs, counts


# ------------------------------------------------------------------
# Window
# ------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        root.resizable(False, False)

        self.engine = Engine()
        self.cap = None
        self.static_image = None
        self.is_webcam = False
        self.running = False
        self.after_id = None
        self.last_pairs = []
        self.last_display = None
        self.photo_main = None
        self.photo_mask = None
        self.fps = 0.0
        self.last_time = time.time()
        self.note = ""
        self.note_until = 0.0

        self.s_min = tk.IntVar(value=80)
        self.v_min = tk.IntVar(value=60)
        self.ripe_hue = tk.IntVar(value=t.RIPE_MAX_HUE)
        self.window = tk.IntVar(value=t.SMOOTH_FRAMES)
        self.white_balance = tk.BooleanVar(value=True)
        self.show_hsv = tk.BooleanVar(value=False)

        self.build()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_placeholder()

    # ---------- layout ----------
    def build(self):
        pad = {"padx": 6, "pady": 3}
        left = tk.Frame(self.root)
        left.grid(row=0, column=0, padx=8, pady=8)
        right = tk.Frame(self.root)
        right.grid(row=0, column=1, sticky="n", padx=(0, 8), pady=8)

        self.video_label = tk.Label(left)
        self.video_label.pack()
        self.status = tk.Label(left, text="Choose a source to begin.", anchor="w",
                               font=("Segoe UI", 11))
        self.status.pack(fill="x", pady=(6, 0))

        tk.Label(right, text=APP_TITLE, font=("Segoe UI", 14, "bold")).pack(**pad)
        tk.Button(right, text="Start Webcam", width=24, command=self.start_webcam).pack(**pad)
        tk.Button(right, text="Open Image...", width=24, command=self.open_image).pack(**pad)
        tk.Button(right, text="Open Video...", width=24, command=self.open_video).pack(**pad)
        tk.Button(right, text="Stop", width=24, command=self.stop).pack(**pad)

        tk.Checkbutton(right, text="White balance", variable=self.white_balance,
                       command=self.engine.reset).pack(anchor="w", **pad)
        tk.Checkbutton(right, text="Show hue (H) and saturation (S)",
                       variable=self.show_hsv).pack(anchor="w", **pad)

        self.slider(right, "Saturation min", self.s_min, 0, 255)
        self.slider(right, "Brightness min", self.v_min, 0, 255)
        self.slider(right, "Ripe up to hue (orange counts as ripe)", self.ripe_hue, 5, 30)
        self.slider(right, "Label smoothing (frames; lower = faster)", self.window, 2, t.MAX_WINDOW)

        tk.Label(right, text="Log visible tomatoes with TRUE label:").pack(anchor="w", **pad)
        row = tk.Frame(right)
        row.pack(**pad)
        for label in ("NOT YET", "ALMOST", "RIPE"):
            tk.Button(row, text=label, width=8,
                      command=lambda lb=label: self.log_true(lb)).pack(side="left", padx=2)

        tk.Button(right, text="Save Snapshot", width=24, command=self.snapshot).pack(**pad)

        tk.Label(right, text="Mask view").pack(anchor="w", **pad)
        self.mask_label = tk.Label(right)
        self.mask_label.pack(**pad)
        tk.Button(right, text="Quit", width=24, command=self.on_close).pack(**pad)

    def slider(self, parent, text, variable, low, high):
        tk.Label(parent, text=text).pack(anchor="w", padx=6)
        tk.Scale(parent, from_=low, to=high, orient=tk.HORIZONTAL, variable=variable,
                 length=240).pack(padx=6)

    # ---------- drawing ----------
    def show(self, label, image, attr):
        photo = to_photo(image)
        if photo is None:
            return
        label.configure(image=photo)
        setattr(self, attr, photo)  # keep a reference or Tkinter drops the picture

    def show_placeholder(self):
        canvas = np.zeros((480, t.FRAME_WIDTH, 3), np.uint8)
        cv2.putText(canvas, "Start Webcam, Open Image, or Open Video",
                    (40, 240), t.FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self.show(self.video_label, canvas, "photo_main")
        blank = np.zeros((240, MASK_WIDTH, 3), np.uint8)
        self.show(self.mask_label, blank, "photo_mask")

    def flash(self, text):
        self.note = text
        self.note_until = time.time() + 3
        self.status.configure(text=text)

    # ---------- sources ----------
    def start_webcam(self):
        cap, _ = t.open_source(None)
        if cap is None:
            messagebox.showerror(APP_TITLE, "Could not open the webcam.\n"
                                 "Close other apps that use the camera and try again.")
            return
        self.begin(cap, None, True)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose a photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if not path:
            return
        image = None
        try:
            image = read_image(path)
        except OSError:
            pass
        if image is None:
            messagebox.showerror(APP_TITLE, "Could not read that image.")
            return
        self.begin(None, image, False)

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror(APP_TITLE, "Could not open that video.")
            return
        self.begin(cap, None, False)

    def begin(self, cap, image, webcam):
        self.stop_source()
        self.cap = cap
        self.static_image = image
        self.is_webcam = webcam
        self.engine.reset()
        self.running = True
        self.tick()

    def stop_source(self):
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.static_image = None

    def stop(self):
        self.stop_source()
        self.last_pairs = []
        self.show_placeholder()
        self.status.configure(text="Stopped. Choose a source to begin.")

    # ---------- main loop ----------
    def tick(self):
        if not self.running:
            return
        try:
            if self.static_image is not None:
                frame = self.static_image.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    self.stop_source()
                    self.status.configure(text="Finished: no more frames.")
                    return
                if self.is_webcam:
                    frame = cv2.flip(frame, 1)  # mirror view
            frame = t.resize_width(frame, t.FRAME_WIDTH)

            display, mask_image, pairs, counts = self.engine.process(
                frame, self.s_min.get(), self.v_min.get(), self.ripe_hue.get(),
                self.white_balance.get(), self.show_hsv.get(), self.window.get())
            self.last_pairs = pairs
            self.last_display = display

            now = time.time()
            dt = now - self.last_time
            self.last_time = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

            if self.note and time.time() < self.note_until:
                text = self.note
            else:
                text = (f"Tomatoes: {len(pairs)}   Ripe: {counts['RIPE']}   "
                        f"Almost: {counts['ALMOST']}   Not yet: {counts['NOT YET']}   "
                        f"FPS: {self.fps:.0f}")
            self.status.configure(text=text)

            self.show(self.video_label, display, "photo_main")
            mask_h = max(1, int(mask_image.shape[0] * MASK_WIDTH / mask_image.shape[1]))
            small = cv2.resize(mask_image, (MASK_WIDTH, mask_h))
            self.show(self.mask_label, small, "photo_mask")
        except Exception:
            self.stop_source()
            log_error(traceback.format_exc())
            messagebox.showerror(APP_TITLE, "Something went wrong and processing stopped.\n"
                                 "Details were saved to error_log.txt.")
            return

        self.after_id = self.root.after(15 if self.is_webcam else 30, self.tick)

    # ---------- buttons ----------
    def log_true(self, true_label):
        if not self.last_pairs:
            messagebox.showinfo(APP_TITLE, "No tomato in view to log.")
            return
        for track, _ in self.last_pairs:
            t.log_row(true_label, track.label(), track.mean_shares(),
                      self.white_balance.get(), track.hue, track.sat)
        self.flash(f"Logged {len(self.last_pairs)} tomato(es) as {true_label}")

    def snapshot(self):
        if self.last_display is None:
            messagebox.showinfo(APP_TITLE, "Nothing to save yet.")
            return
        os.makedirs(t.SNAPSHOT_DIR, exist_ok=True)
        name = datetime.now().strftime("tomato_%Y%m%d_%H%M%S.jpg")
        save_image(os.path.join(t.SNAPSHOT_DIR, name), self.last_display)
        self.flash(f"Saved snapshots/{name}")

    def on_close(self):
        self.stop_source()
        self.root.destroy()


def main():
    try:
        root = tk.Tk()
        App(root)
        root.mainloop()
    except Exception:
        log_error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()