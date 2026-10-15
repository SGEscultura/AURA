"""
Tomato Ripeness Checker (OpenCV)

Finds every tomato in view and labels each one NOT YET / ALMOST / RIPE
by measuring how much of the fruit is red, yellow-orange, or green.
Each tomato gets an ID (#1, #2, ...) and keeps it from frame to frame.

Run:
    python tomato_ripeness.py              -> live webcam
    python tomato_ripeness.py photo.jpg    -> one still photo
    python tomato_ripeness.py clip.mp4     -> a recorded video (good for backup demos)

Keys (click the video window first):
    q or Esc  quit
    w         turn white balance on/off
    s         save a snapshot into the snapshots folder
    1 / 2 / 3 log every visible tomato with the TRUE label
              (1 = NOT YET, 2 = ALMOST, 3 = RIPE). Only press these when ALL
              tomatoes in view have that same true ripeness.
    h         show/hide each tomato's median hue (H) and saturation (S) on its label
    r         reset the smoothing

Setup that gives the best results: tomatoes spaced apart (not touching), plain
light background (white paper or a white table), good even light, no hands or
shiny glare in view. Tomatoes that touch are merged into ONE blob and counted once.
"""

import csv
import os
import sys
import time
from collections import Counter, deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

# ------------------------------------------------------------------
# Settings you can tune
# ------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640            # every frame is resized to this width
SMOOTH_FRAMES = 8            # label is a majority vote over this many recent frames
MAX_WINDOW = 30              # longest smoothing window the app slider allows
MIN_AREA_FRACTION = 0.005    # ignore blobs smaller than 0.5% of the frame
MAX_TOMATOES = 12            # look at the 12 biggest blobs at most
MATCH_DISTANCE = 140         # pixels: max movement between frames to keep the same ID
MAX_MISSED = 15              # frames a tomato can vanish before its ID is dropped
GRACE_FRAMES = 6             # a tomato that just vanished stays drawn (thin box) this long

# OpenCV hue runs 0-179. Red wraps around, so it needs two ranges.
# Hues from 0 up to RIPE_MAX_HUE count as ripe (red, red-orange, orange), and so do
# hues 170-179. Yellowish hues just above that count as "turning". The "Ripe hue"
# slider changes RIPE_MAX_HUE live. Use the H readout (key h) to see your tomatoes' real hue.
RIPE_MAX_HUE = 20
RED_HIGH = (170, 179)
TURNING_MAX_HUE = 34
GREEN = (35, 85)

# Ripeness score = red share + TURNING_WEIGHT * yellow-orange share
TURNING_WEIGHT = 0.5
RIPE_SCORE = 0.70            # score at or above this -> RIPE
ALMOST_SCORE = 0.35          # score at or above this -> ALMOST, otherwise NOT YET

LOG_FILE = "ripeness_log.csv"
SNAPSHOT_DIR = "snapshots"

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {                   # BGR colors
    "RIPE": (0, 0, 255),
    "ALMOST": (0, 165, 255),
    "NOT YET": (0, 200, 0),
    "NO TOMATO": (160, 160, 160),
}
TRUE_LABEL_KEYS = {ord("1"): "NOT YET", ord("2"): "ALMOST", ord("3"): "RIPE"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# ------------------------------------------------------------------
# Image processing
# ------------------------------------------------------------------
def resize_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    return cv2.resize(img, (width, int(h * width / w)))


def gray_world(img: np.ndarray) -> np.ndarray:
    """Simple white balance: scale each color channel so their averages match."""
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    gain = means.mean() / means
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def hue_mask(hsv, h_low, h_high, s_min, v_min):
    return cv2.inRange(
        hsv, np.array([h_low, s_min, v_min]), np.array([h_high, 255, 255])
    )


def clean(mask):
    """Remove specks, then fill small gaps."""
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)


def analyze(frame: np.ndarray, s_min: int, v_min: int, ripe_max_hue: int = RIPE_MAX_HUE):
    """
    Returns (results, masks).
    results is a list with one dict per tomato (biggest first): contour, box, center,
    the red / turning / green shares, and the median hue and saturation of the fruit.
    It is empty when nothing tomato-sized is found.
    masks is (red, turning, green) for the mask window.
    """
    blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    ripe_max_hue = max(5, min(int(ripe_max_hue), TURNING_MAX_HUE - 1))
    red = cv2.bitwise_or(
        hue_mask(hsv, 0, ripe_max_hue, s_min, v_min),
        hue_mask(hsv, RED_HIGH[0], RED_HIGH[1], s_min, v_min),
    )
    turning = hue_mask(hsv, ripe_max_hue + 1, TURNING_MAX_HUE, s_min, v_min)
    green = hue_mask(hsv, GREEN[0], GREEN[1], s_min, v_min)
    masks = (red, turning, green)

    # The fruit is everything that has a tomato color (red, yellow-orange, or green)
    colored = cv2.bitwise_or(cv2.bitwise_or(red, turning), green)
    fruit = clean(colored)

    contours = cv2.findContours(fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    min_area = MIN_AREA_FRACTION * frame.shape[0] * frame.shape[1]
    big = [c for c in contours if cv2.contourArea(c) >= min_area]
    big.sort(key=cv2.contourArea, reverse=True)

    results = []
    for contour in big[:MAX_TOMATOES]:
        # Count each color only inside this fruit's outline
        region = np.zeros(fruit.shape, np.uint8)
        cv2.drawContours(region, [contour], -1, 255, -1)
        r = cv2.countNonZero(cv2.bitwise_and(red, region))
        t = cv2.countNonZero(cv2.bitwise_and(turning, region))
        g = cv2.countNonZero(cv2.bitwise_and(green, region))
        total = r + t + g
        if total == 0:
            continue
        # Median hue / saturation of this fruit. Hue is shifted by 90 first so red,
        # which wraps around 0/179, does not break the median.
        pixels = hsv[cv2.bitwise_and(colored, region) > 0]
        hue_shifted = (pixels[:, 0].astype(np.int32) + 90) % 180
        median_hue = int(round((float(np.median(hue_shifted)) - 90) % 180))
        median_sat = int(round(float(np.median(pixels[:, 1]))))
        x, y, w, h = cv2.boundingRect(contour)
        results.append({
            "contour": contour,
            "box": (x, y, w, h),
            "center": (x + w / 2.0, y + h / 2.0),
            "red": r / total,
            "turning": t / total,
            "green": g / total,
            "hue": median_hue,
            "sat": median_sat,
        })
    return results, masks


def ripeness_score(red, turning):
    return red + TURNING_WEIGHT * turning


def classify(red, turning):
    score = ripeness_score(red, turning)
    if score >= RIPE_SCORE:
        return "RIPE"
    if score >= ALMOST_SCORE:
        return "ALMOST"
    return "NOT YET"


def mask_view(masks, results):
    red, turning, green = masks
    view = np.zeros((red.shape[0], red.shape[1], 3), np.uint8)
    view[green > 0] = COLORS["NOT YET"]
    view[turning > 0] = COLORS["ALMOST"]
    view[red > 0] = COLORS["RIPE"]
    for res in results:
        cv2.drawContours(view, [res["contour"]], -1, (255, 255, 255), 2)
    return view


# ------------------------------------------------------------------
# Tracking: keep each tomato's ID and smooth its label over time
# ------------------------------------------------------------------
class Track:
    def __init__(self, track_id, center):
        self.id = track_id
        self.center = center
        self.labels = deque(maxlen=MAX_WINDOW)
        self.shares = deque(maxlen=MAX_WINDOW)
        self.window = SMOOTH_FRAMES
        self.missed = 0
        self.hue = 0
        self.sat = 0
        self.box = (0, 0, 0, 0)

    def label(self):
        recent = list(self.labels)[-self.window:]
        return Counter(recent).most_common(1)[0][0]

    def mean_shares(self):
        recent = list(self.shares)[-self.window:]
        return tuple(np.mean(recent, axis=0))


class Tracker:
    """Matches tomatoes between frames by nearest center point."""

    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.window = SMOOTH_FRAMES

    def reset(self):
        self.tracks = []
        self.next_id = 1

    def update(self, results):
        """Returns a list of (track, result) pairs for the tomatoes seen this frame."""
        pairs = []
        unmatched = list(self.tracks)
        for res in sorted(results, key=lambda r: r["center"][0]):  # left to right
            cx, cy = res["center"]
            best, best_dist = None, MATCH_DISTANCE
            for track in unmatched:
                dist = ((track.center[0] - cx) ** 2 + (track.center[1] - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best, best_dist = track, dist
            if best is None:
                best = Track(self.next_id, res["center"])
                self.next_id += 1
                self.tracks.append(best)
            else:
                unmatched.remove(best)
            best.center = res["center"]
            best.missed = 0
            best.window = self.window
            best.box = res["box"]
            best.hue = res["hue"]
            best.sat = res["sat"]
            best.labels.append(classify(res["red"], res["turning"]))
            best.shares.append((res["red"], res["turning"], res["green"]))
            pairs.append((best, res))
        for track in unmatched:
            track.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED]
        return pairs

    def lost(self, grace=GRACE_FRAMES):
        """Tomatoes seen a moment ago but not in this frame (kept on screen briefly)."""
        return [t for t in self.tracks if 1 <= t.missed <= grace and t.labels]


# ------------------------------------------------------------------
# Drawing, logging, input
# ------------------------------------------------------------------
def draw_panel(img: np.ndarray, lines: list) -> None:
    overlay = img.copy()
    height = 14 + 22 * len(lines)
    cv2.rectangle(overlay, (0, 0), (340, height), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (10, 24 + 22 * i), FONT, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)


def log_row(true_label, predicted, shares, white_balance, hue=None, sat=None):
    is_new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "true_label", "predicted",
                             "red_pct", "turning_pct", "green_pct", "white_balance",
                             "hue_med", "sat_med"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"), true_label, predicted,
            f"{shares[0] * 100:.1f}", f"{shares[1] * 100:.1f}", f"{shares[2] * 100:.1f}",
            "on" if white_balance else "off",
            "" if hue is None else hue, "" if sat is None else sat,
        ])


def open_source(source: Optional[str] = None):
    """Returns (capture, static_image). Exactly one of them is not None."""
    if source and source.lower().endswith(IMAGE_EXTENSIONS):
        image = cv2.imread(source)
        if image is None:
            print(f"Could not read image: {source}")
            return None, None
        return None, image

    if source:
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # Windows fallback
    if not cap.isOpened():
        print("Could not open the camera or video. Close other apps that use the webcam.")
        return None, None
    return cap, None


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
def main():
    source = sys.argv[1] if len(sys.argv) > 1 else None
    cap, static_image = open_source(source)
    if cap is None and static_image is None:
        return

    win_main, win_mask = "Tomato Ripeness", "Mask"
    cv2.namedWindow(win_main)
    cv2.namedWindow(win_mask)
    cv2.createTrackbar("S min", win_mask, 80, 255, lambda v: None)
    cv2.createTrackbar("V min", win_mask, 60, 255, lambda v: None)
    cv2.createTrackbar("Ripe hue", win_mask, RIPE_MAX_HUE, 30, lambda v: None)

    tracker = Tracker()
    use_white_balance = True
    show_hsv = False
    fps = 0.0
    last_time = time.time()
    message, message_until = "", 0.0

    while True:
        if static_image is not None:
            frame = static_image.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                print("No more frames (end of video or camera read failed).")
                break
            if source is None:
                frame = cv2.flip(frame, 1)  # mirror view for the webcam
        frame = resize_width(frame, FRAME_WIDTH)

        s_min = cv2.getTrackbarPos("S min", win_mask)
        v_min = cv2.getTrackbarPos("V min", win_mask)
        ripe_hue = cv2.getTrackbarPos("Ripe hue", win_mask)

        processed = gray_world(frame) if use_white_balance else frame
        results, masks = analyze(processed, s_min, v_min, ripe_hue)
        pairs = tracker.update(results)
        lost_tracks = tracker.lost()

        # Draw every tomato on the live frame
        display = frame.copy()
        counts = Counter()
        for track, res in pairs:
            label = track.label()
            counts[label] += 1
            shares = track.mean_shares()
            score = ripeness_score(shares[0], shares[1]) * 100
            x, y, w, h = res["box"]
            color = COLORS[label]
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 3)
            tag = f"#{track.id} {label} {score:.0f}%"
            if show_hsv:
                tag += f"  H{res['hue']} S{res['sat']}"
            pos = (x, max(28, y - 10))
            cv2.putText(display, tag, pos, FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(display, tag, pos, FONT, 0.7, color, 2, cv2.LINE_AA)

        for track in lost_tracks:
            label = track.label()
            counts[label] += 1
            shares = track.mean_shares()
            score = ripeness_score(shares[0], shares[1]) * 100
            x, y, w, h = track.box
            color = COLORS[label]
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 1)
            tag = f"#{track.id} {label} {score:.0f}% (lost)"
            pos = (x, max(28, y - 10))
            cv2.putText(display, tag, pos, FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(display, tag, pos, FONT, 0.6, color, 1, cv2.LINE_AA)

        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        lines = [f"Tomatoes: {sum(counts.values())}",
                 f"Ripe: {counts['RIPE']}   Almost: {counts['ALMOST']}   Not yet: {counts['NOT YET']}",
                 f"White balance: {'ON' if use_white_balance else 'OFF'}",
                 f"FPS: {fps:4.1f}"]
        draw_panel(display, lines)

        if message and time.time() < message_until:
            cv2.putText(display, message, (10, display.shape[0] - 12), FONT, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_main, display)
        cv2.imshow(win_mask, mask_view(masks, results))

        key = cv2.waitKey(30 if static_image is not None else 1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("w"):
            use_white_balance = not use_white_balance
            tracker.reset()
        elif key == ord("h"):
            show_hsv = not show_hsv
        elif key == ord("r"):
            tracker.reset()
        elif key == ord("s"):
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            name = datetime.now().strftime("tomato_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(os.path.join(SNAPSHOT_DIR, name), display)
            message, message_until = f"Saved {name}", time.time() + 2
        elif key in TRUE_LABEL_KEYS:
            if not pairs:
                message, message_until = "No tomato to log", time.time() + 2
            else:
                true_label = TRUE_LABEL_KEYS[key]
                for track, _ in pairs:
                    log_row(true_label, track.label(), track.mean_shares(), use_white_balance,
                            track.hue, track.sat)
                message = f"Logged {len(pairs)} tomato(es) as {true_label}"
                message_until = time.time() + 2

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()