"""
Batch image test for the tomato ripeness checker (handles several tomatoes per photo).

Put your test photos in folders named after the TRUE ripeness:

    test_images/
        not_yet/    photos where EVERY tomato is green
        almost/     photos where EVERY tomato is orange, turning, or half-red
        ripe/       photos where EVERY tomato is fully red
        mixed/      (optional) photos with different ripeness levels together.
                    These are annotated and counted, but NOT scored.

Run:
    python batch_image_test.py                      -> uses the test_images folder
    python batch_image_test.py my_photos            -> uses another folder
    python batch_image_test.py my_photos 90 70      -> also sets S min = 90, V min = 70
    python batch_image_test.py my_photos 80 60 22   -> also sets Ripe hue = 22

Accuracy is counted per TOMATO, not per photo. Tomatoes that touch are merged into
one blob and counted once, so keep them spaced apart.

Outputs:
    image_test_log.csv   -> run:  python evaluate_log.py image_test_log.csv
    image_test_output/   -> every photo with the predicted labels drawn on it
"""

import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2

import tomato_ripeness as t

# None means "show it but do not score it"
FOLDER_TO_LABEL = {"not_yet": "NOT YET", "almost": "ALMOST", "ripe": "RIPE", "mixed": None}
LOG_FILE = "image_test_log.csv"
OUTPUT_DIR = "image_test_output"


def predict(frame, white_balance, s_min, v_min, ripe_hue):
    """Returns a list with one dict per tomato: label, shares, box."""
    processed = t.gray_world(frame) if white_balance else frame
    results, _ = t.analyze(processed, s_min, v_min, ripe_hue)
    detections = []
    for res in sorted(results, key=lambda r: r["center"][0]):  # left to right
        detections.append({
            "label": t.classify(res["red"], res["turning"]),
            "shares": (res["red"], res["turning"], res["green"]),
            "box": res["box"],
            "hue": res["hue"],
            "sat": res["sat"],
        })
    return detections


def annotate(frame, detections, true_label, flag):
    image = frame.copy()
    for i, det in enumerate(detections, start=1):
        x, y, w, h = det["box"]
        color = t.COLORS[det["label"]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        tag = f"#{i} {det['label']}"
        pos = (x, max(50, y - 10))
        cv2.putText(image, tag, pos, t.FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, tag, pos, t.FONT, 0.7, color, 2, cv2.LINE_AA)
    truth = true_label if true_label else "mixed (not scored)"
    header = f"{len(detections)} tomato(es) | true: {truth} | WB {flag.upper()}"
    cv2.putText(image, header, (10, 28), t.FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, header, (10, 28), t.FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_images")
    s_min = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    v_min = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    ripe_hue = int(sys.argv[4]) if len(sys.argv) > 4 else t.RIPE_MAX_HUE

    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        print("Create it with subfolders not_yet, almost, ripe (and optionally mixed).")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    misses = []
    counts = {"on": [0, 0], "off": [0, 0]}  # [correct, total] per tomato
    photos_done = 0

    for sub, true_label in FOLDER_TO_LABEL.items():
        sub_folder = folder / sub
        if not sub_folder.is_dir():
            print(f"Skipping missing folder: {sub_folder}")
            continue

        images = sorted(p for p in sub_folder.iterdir()
                        if p.suffix.lower() in t.IMAGE_EXTENSIONS)
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                print(f"Could not read {path}, skipping")
                continue
            frame = t.resize_width(image, t.FRAME_WIDTH)
            photos_done += 1

            for wb in (True, False):
                flag = "on" if wb else "off"
                detections = predict(frame, wb, s_min, v_min, ripe_hue)

                if wb:  # one summary line per photo is enough
                    found = ", ".join(d["label"] for d in detections) or "NO TOMATO"
                    print(f"{sub}/{path.name}: {len(detections)} tomato(es) -> {found}")

                if true_label is not None:
                    if not detections:
                        detections_for_log = [{"label": "NO TOMATO", "shares": (0.0, 0.0, 0.0), "hue": "", "sat": ""}]
                    else:
                        detections_for_log = detections
                    for det in detections_for_log:
                        shares = det["shares"]
                        rows.append([
                            datetime.now().isoformat(timespec="seconds"), true_label,
                            det["label"], f"{shares[0] * 100:.1f}",
                            f"{shares[1] * 100:.1f}", f"{shares[2] * 100:.1f}", flag,
                            det["hue"], det["sat"],
                        ])
                        counts[flag][1] += 1
                        if det["label"] == true_label:
                            counts[flag][0] += 1
                        else:
                            misses.append((flag, true_label, det["label"], path.name, shares, det["hue"], det["sat"]))

                out_name = f"{sub}_{path.stem}_wb{flag}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, out_name),
                            annotate(frame, detections, true_label, flag))

    if photos_done == 0:
        print("No images were processed. Check your folder names and file types.")
        return

    if rows:
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "true_label", "predicted",
                             "red_pct", "turning_pct", "green_pct", "white_balance",
                             "hue_med", "sat_med"])
            writer.writerows(rows)

    print(f"\nProcessed {photos_done} photos (each tested with white balance on and off).")
    for flag in ("on", "off"):
        correct, total = counts[flag]
        if total:
            print(f"White balance {flag.upper():3s}: {correct}/{total} tomatoes correct "
                  f"({correct / total * 100:.1f}%)")

    if misses:
        print("\nMisses (check these photos in image_test_output):")
        for flag, true_label, label, name, shares, hue, sat in misses:
            print(f"  WB {flag:3s} | true={true_label:7s} predicted={label:9s} | {name} | "
                  f"red={shares[0] * 100:.0f}% yellow={shares[1] * 100:.0f}% green={shares[2] * 100:.0f}% | "
                  f"hue={hue} sat={sat}")

    if rows:
        print(f"\nLog saved to {LOG_FILE}. For confusion tables run:")
        print(f"    python evaluate_log.py {LOG_FILE}")


if __name__ == "__main__":
    main()