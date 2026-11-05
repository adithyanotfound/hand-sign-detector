"""Run MediaPipe over the dataset images once and cache the hand landmarks.

Reads every image of the gesture dataset, plus a fixed sample per letter of the
alphabet dataset. Also stores a colour histogram of each gesture image's torso area,
which train.py uses to approximate which person is in the photo.
"""
import argparse
import os
import random
import re
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from features import NUM_LANDMARKS, to_isotropic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_LETTER = 40


def torso_histogram(image):
    height, width = image.shape[:2]
    torso = image[int(0.8 * height):height, int(0.2 * width):int(0.8 * width)]
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [6, 6, 6], [0, 256] * 3).ravel()
    return np.sqrt(hist / hist.sum()).astype(np.float32)


def gesture_images(folder):
    classes = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
    for name in classes:
        files = [f for f in os.listdir(os.path.join(folder, name)) if f.lower().endswith(".jpg")]
        files.sort(key=lambda f: int(re.findall(r"(\d+)", f)[-1]))
        for filename in files:
            image = cv2.imread(os.path.join(folder, name, filename))
            yield "gestures", name, int(re.findall(r"(\d+)", filename)[-1]), image


def alphabet_images(zip_path):
    rng = random.Random(0)
    with zipfile.ZipFile(zip_path) as archive:
        by_letter = {}
        for entry in archive.namelist():
            match = re.fullmatch(r"SignAlphaSet/([A-Z])/[A-Z]_(\d+)\.jpg", entry)
            if match:
                by_letter.setdefault(match.group(1), []).append((int(match.group(2)), entry))
        for letter in sorted(by_letter):
            for number, entry in rng.sample(sorted(by_letter[letter]), PER_LETTER):
                data = np.frombuffer(archive.read(entry), dtype=np.uint8)
                yield "alphabet", letter, number, cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gestures", default=os.path.join(ROOT, "data/raw/images/Sign_Language_Dataset"))
    parser.add_argument("--alphabet", default=os.path.join(ROOT, "data/raw/alphabet_zip/SignAlphaSet.zip"))
    parser.add_argument("--model", default=os.path.join(ROOT, "models/hand_landmarker.task"))
    parser.add_argument("--out", default=os.path.join(ROOT, "data/landmarks.npz"))
    args = parser.parse_args()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )

    rows = {k: [] for k in ("points", "names", "sources", "numbers", "torso", "detected")}
    start = time.time()
    with HandLandmarker.create_from_options(options) as landmarker:
        for generator in (gesture_images(args.gestures), alphabet_images(args.alphabet)):
            for source, name, number, image in generator:
                if image is None:
                    print(f"  unreadable: {source}/{name}/{number}")
                    continue
                height, width = image.shape[:2]
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                found = bool(result.hand_landmarks)
                if found:
                    raw = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]
                    rows["points"].append(to_isotropic(raw, width, height))
                else:
                    rows["points"].append(np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32))
                rows["names"].append(name)
                rows["sources"].append(source)
                rows["numbers"].append(number)
                rows["torso"].append(torso_histogram(image))
                rows["detected"].append(found)

    arrays = {k: np.array(v) for k, v in rows.items()}
    for source in ("gestures", "alphabet"):
        mask = arrays["sources"] == source
        summary = ", ".join(
            f"{n}:{arrays['detected'][mask & (arrays['names'] == n)].sum()}/{(mask & (arrays['names'] == n)).sum()}"
            for n in np.unique(arrays["names"][mask])
        )
        print(f"{source}: {summary}")
    print(f"{len(arrays['names'])} images in {time.time() - start:.0f}s")
    np.savez_compressed(args.out, **arrays)
    print("saved", args.out)


if __name__ == "__main__":
    main()
