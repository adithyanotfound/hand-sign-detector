"""Draw the report's figures from the project's real results.

Usage: .venv/bin/python -I report/make_figures.py <train_output.txt> <annotated.mp4>
train_output.txt is what src/train.py prints; annotated.mp4 comes from realtime.py --save.
"""
import json
import os
import re
import sys
import zipfile

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures")
IMAGES = os.path.join(ROOT, "data/raw/images/Sign_Language_Dataset")
ALPHABET = os.path.join(ROOT, "data/raw/alphabet_zip/SignAlphaSet.zip")

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#deddd8"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 12, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "savefig.dpi": 300,
})
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
               (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def pipeline():
    steps = ["Webcam\nframe", "MediaPipe\n21 hand\npoints", "Normalise\nto 42\nvalues",
             "Neural net\n42-64-32-6", "Smoothing,\n60% check", "Label and\nconfidence"]
    fig = plt.figure(figsize=(6.5, 0.95))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 6.5), ylim=(0, 0.95))
    ax.axis("off")
    width, gap = 0.93, (6.5 - 6 * 0.93) / 5
    for i, text in enumerate(steps):
        x = i * (width + gap)
        trained = i == 3
        ax.add_patch(FancyBboxPatch((x + 0.03, 0.05), width - 0.06, 0.85, boxstyle="round,pad=0.01,rounding_size=0.06",
                                    fc="#d7e6f8" if trained else "#f3f2ee", ec=BLUE if trained else MUTED, lw=1))
        ax.text(x + width / 2, 0.475, text, ha="center", va="center", linespacing=1.0)
        if i < 5:
            ax.annotate("", xy=(x + width + gap + 0.02, 0.475), xytext=(x + width - 0.02, 0.475),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1, shrinkA=0, shrinkB=0))
    fig.savefig(f"{OUT}/fig1_pipeline.png")
    plt.close(fig)


def crop_hand(image, points, size=400):
    image = image.copy()
    p = points[:, :2]
    line = max(2, int(round((p.max(0) - p.min(0)).max() / 90)))
    for a, b in CONNECTIONS:
        cv2.line(image, tuple(p[a].astype(int)), tuple(p[b].astype(int)), (90, 210, 90), line, cv2.LINE_AA)
    for point in p.astype(int):
        cv2.circle(image, tuple(point), line + 1, (255, 255, 255), -1, cv2.LINE_AA)
    centre, half = (p.min(0) + p.max(0)) / 2, (p.max(0) - p.min(0)).max() / 2 * 1.2
    height, width = image.shape[:2]
    x0, y0 = (centre - half).astype(int)
    x1, y1 = (centre + half).astype(int)
    pad = max(0, -x0, -y0, x1 - width, y1 - height)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = image[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    return cv2.cvtColor(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def signs():
    data = np.load(os.path.join(ROOT, "data/landmarks.npz"))
    choices = [("hello", 213, "Hello"), ("i_love_you", 125, "I Love You"), ("peace", 178, "Peace"),
               ("thank_you", 133, "Thank You"), ("yes", 125, "Yes"), ("F", None, "No sign")]
    archive = zipfile.ZipFile(ALPHABET)
    fig, axes = plt.subplots(1, 6, figsize=(6.5, 1.35))
    for ax, (name, number, title) in zip(axes, choices):
        if number is None:
            rows = np.where((data["names"] == name) & data["detected"])[0]
            i = rows[0]
            raw = archive.read(f"SignAlphaSet/{name}/{name}_{data['numbers'][i]}.jpg")
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        else:
            rows = np.where((data["names"] == name) & (data["sources"] == "gestures") & data["detected"])[0]
            i = rows[np.argmin(np.abs(data["numbers"][rows] - number))]
            image = cv2.imread(f"{IMAGES}/{name}/Image_{data['numbers'][i]}.jpg")
        ax.imshow(crop_hand(image, data["points"][i]))
        ax.set_title(title, fontsize=12, pad=3)
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=0.84, bottom=0, wspace=0.05)
    fig.savefig(f"{OUT}/fig2_signs.png")
    plt.close(fig)


def app_frames(video):
    capture = cv2.VideoCapture(video)
    tiles = []
    for frame in (55, 125, 195, 265, 335, 555):  # the five signs, then the real OK shape: no sign
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            raise SystemExit(f"could not read frame {frame}")
        tiles.append(image)
    gap = 8
    height, width = tiles[0].shape[:2]
    canvas = np.full((2 * height + gap, 3 * width + 2 * gap, 3), 255, np.uint8)
    for i, tile in enumerate(tiles):
        row, column = divmod(i, 3)
        canvas[row * (height + gap):row * (height + gap) + height, column * (width + gap):column * (width + gap) + width] = tile
    cv2.imwrite(f"{OUT}/fig3_app.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


def confusion():
    report = json.load(open(os.path.join(ROOT, "models/report.json")))
    matrix = np.array(report["confusion_matrix"])
    labels = ["Hello", "I Love You", "Peace", "Thank You", "Yes", "No sign"]
    rates = matrix / matrix.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("blue", ["#f7f9fc", "#a9cbf0", BLUE, "#174f94"])
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
    for r in range(6):
        for c in range(6):
            if matrix[r, c]:
                ax.text(c, r, str(matrix[r, c]), ha="center", va="center", color="white" if rates[r, c] > 0.5 else INK)
    ax.set_xticks(range(6), labels, rotation=50, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(6), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, 6), minor=True)
    ax.set_yticks(np.arange(-.5, 6), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    fig.savefig(f"{OUT}/fig4_confusion.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def threshold(train_output):
    rows = re.findall(r"^\s+(0\.\d\d)\s+(0\.\d+) \+/- [\d.]+\s+(0\.\d+) \+/- [\d.]+", open(train_output).read(), re.M)
    t, correct, wrong = (np.array(v, dtype=float) for v in zip(*rows))
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot(t, correct * 100, color=BLUE, lw=2, marker="o", ms=4, label="Signs named correctly")
    ax.plot(t, wrong * 100, color=ORANGE, lw=2, marker="s", ms=4, label="Unseen shapes named as signs")
    ax.axvline(0.60, color=MUTED, lw=1, ls="--")
    i = int(np.argmin(np.abs(t - 0.60)))
    ax.annotate(f"{correct[i] * 100:.1f}%", (0.60, correct[i] * 100), xytext=(-6, -16), textcoords="offset points", ha="right")
    ax.annotate(f"{wrong[i] * 100:.1f}%", (0.60, wrong[i] * 100), xytext=(-6, 7), textcoords="offset points", ha="right")
    ax.set(xlabel="Confidence threshold", ylabel="Images (%)", ylim=(0, 100), xlim=(0.38, 0.97))
    ax.set_xticks([0.4, 0.6, 0.8])
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.2), frameon=False, ncol=1, handlelength=1.6,
              labelspacing=0.15, borderaxespad=0)
    fig.savefig(f"{OUT}/fig5_threshold.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def people(train_output):
    rows = re.findall(r"person ([ABC]): \[(.+?)\]", open(train_output).read())
    counts = {person: [int(v) for v in values.split(",")][:5] for person, values in rows}
    labels = ["Hello", "I Love You", "Peace", "Thank You", "Yes"]
    fig, ax = plt.subplots(figsize=(6.5, 1.9))
    left = np.zeros(5)
    for (person, values), color in zip(counts.items(), (BLUE, ORANGE, AQUA)):
        values = np.array(values)
        ax.barh(labels, values, left=left, color=color, height=0.68, edgecolor="white", linewidth=1, label=f"Person {person}")
        for i, value in enumerate(values):
            if value >= 25:
                ax.text(left[i] + value / 2, i, str(value), ha="center", va="center", color="white" if color == BLUE else INK)
        left += values
    ax.invert_yaxis()
    ax.set_xlabel("Images used")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, borderaxespad=0.1)
    fig.savefig(f"{OUT}/fig6_people.png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    signs()
    app_frames(sys.argv[2])
    confusion()
    threshold(sys.argv[1])
    people(sys.argv[1])
    print(sorted(os.listdir(OUT)))
