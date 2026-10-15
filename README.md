# AURA - Automated Unripe/Ripe Analyzer

**CSC 126 – Computer Vision Midterm Project** (Python + OpenCV)

## Description
Deciding by eye whether tomatoes are ready to pick or sell is slow and inconsistent. AURA finds every tomato in an image, video, or live webcam feed and labels each one **NOT YET**, **ALMOST**, or **RIPE** by measuring how much of the fruit is red, yellow-orange, or green. Every tomato gets a box, an ID (#1, #2, ...), and a ripeness percentage, and keeps its ID from frame to frame.

## Features
- Live webcam, photo, or recorded video input
- Desktop window (`tomato_app.py`) with buttons, live sliders, and a mask view
- Command-line mode (`tomato_ripeness.py`) with keyboard shortcuts
- Batch accuracy test (`batch_image_test.py`) that annotates every photo and writes a CSV log
- White balance (gray-world) option for uneven lighting
- Label smoothing over recent frames so labels do not flicker

## Techniques used (CSC 126 laboratory topics)
| Topic | How it is used |
|---|---|
| **a. Image Operations** | Resizing, Gaussian blur, BGR→HSV conversion, gray-world white balance, morphological opening/closing to clean masks |
| **b. Image Segmentation** | HSV color thresholding (`cv2.inRange`) into red, yellow-orange, and green masks; the fruit region is their combination |
| **d. Image Analysis & Transformation** | Contours and bounding boxes per tomato, red/yellow/green pixel shares, median hue and saturation, ripeness score |
| **e. Motion & Object Tracking** | Nearest-center tracker gives each tomato a persistent ID and a smoothed label across frames |

## How it works
1. Resize the frame and (optionally) apply white balance.
2. Blur and convert to HSV, then build red, turning (yellow-orange), and green masks.
3. Clean the combined mask with morphology and find one contour per tomato.
4. For each tomato, measure the red / turning / green shares.
5. Ripeness score = red share + 0.5 × turning share. Score ≥ 0.70 → **RIPE**, ≥ 0.35 → **ALMOST**, otherwise **NOT YET**.
6. Track tomatoes between frames and vote the label over recent frames.
7. Draw boxes, labels, and the mask view.

## Setup
Requires Python 3.8 or newer.

```bash
git clone https://github.com/SGEscultura/AURA.git
cd AURA
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
Tkinter ships with Python on Windows and macOS. On Linux install it with `sudo apt install python3-tk`.

## Run
```bash
# Desktop app (webcam / image / video buttons)
python tomato_app.py

# Command line
python tomato_ripeness.py              # live webcam
python tomato_ripeness.py photo.jpg    # one photo
python tomato_ripeness.py clip.mp4     # recorded video

# Batch test on labeled photos
python batch_image_test.py                      # uses test_images/
python batch_image_test.py test_images 90 70    # custom S min and V min
```
**Keys (command-line mode):** `q`/Esc quit · `w` white balance · `s` snapshot · `1`/`2`/`3` log true label (NOT YET / ALMOST / RIPE) · `h` show hue and saturation · `r` reset smoothing.

**Best results:** space tomatoes apart (touching tomatoes merge into one blob), use a plain light background and even light, and avoid glare and hands in view.

## Project structure
```
tomato_app.py           # desktop window (entry point)
tomato_ripeness.py      # detection, classification, tracking, command-line mode
batch_image_test.py     # accuracy test over labeled photos
requirements.txt
test_images/            # sample inputs: not_yet/  almost/  ripe/  mixed/
screenshots/            # output screenshots
```

## Optional: build a Windows .exe
```bash
pip install pyinstaller
pyinstaller --onefile --windowed tomato_app.py
```

## Limitations
- Judges ripeness by color only, so lighting, shadows, and glare affect results.
- Tomatoes that touch are merged and counted as one.
- Needs a mostly plain background; red or green objects in view can be mistaken for fruit.
- Thresholds (saturation, brightness, ripe hue) may need tuning for different tomato varieties and cameras.

## Team
Figuron, Sophia Gabrielle E.
Lugsanay, John Mark G.
Magbanua, Ace Nathaniel Y.
