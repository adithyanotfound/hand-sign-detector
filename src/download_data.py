"""Download the datasets and hand-landmark model, verifying checksums. No login needed.

Sign_Language_Dataset (CC BY 4.0)  https://doi.org/10.17632/pwyrdzwzwf.1  ~85 MB
SignAlphaSet          (CC BY 4.0)  https://doi.org/10.17632/8fmvr9m98w.3  ~445 MB
MediaPipe hand landmark model      ~8 MB

Run with `python -I` so that nothing in the working directory can shadow an import
while untrusted archives are being handled.
"""
import hashlib
import os
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

GESTURES_URL = "https://data.mendeley.com/public-api/zip/pwyrdzwzwf/download/1"
GESTURES_INNER_SHA256 = "89f3a1f96f3f7a04c4076942725a198427ae21f18db5bfc85bfd70d4a87736f9"
ALPHABET_URL = (
    "https://data.mendeley.com/public-files/datasets/8fmvr9m98w/files/"
    "a12e2b20-74de-4301-858f-35ab312d77b7/file_downloaded"
)
ALPHABET_SHA256 = "1fb90e83a9d4d3548545e487b5c7b7be8ff6e6929a8751b79c447501d7bd408f"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, path, expected=None):
    if os.path.exists(path) and (expected is None or sha256(path) == expected):
        print("already have", os.path.relpath(path, ROOT))
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("downloading", url)
    request = urllib.request.Request(url, headers={"User-Agent": "sign-translator-setup"})
    with urllib.request.urlopen(request, timeout=120) as response, open(path + ".part", "wb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            out.write(block)
    if expected is not None and sha256(path + ".part") != expected:
        os.remove(path + ".part")
        raise SystemExit(f"Checksum mismatch for {url}; nothing was kept.")
    os.replace(path + ".part", path)


def main():
    fetch(MODEL_URL, os.path.join(ROOT, "models", "hand_landmarker.task"), MODEL_SHA256)
    fetch(ALPHABET_URL, os.path.join(RAW, "alphabet_zip", "SignAlphaSet.zip"), ALPHABET_SHA256)

    # The gesture dataset is a zip inside a zip; only the inner one has a published hash.
    outer = os.path.join(RAW, "zip_outer", "dataset.zip")
    fetch(GESTURES_URL, outer)
    inner_dir = os.path.join(RAW, "zip_inner")
    with zipfile.ZipFile(outer) as archive:
        archive.extractall(inner_dir)
    inner = None
    for folder, _, names in os.walk(inner_dir):
        for name in names:
            if name.endswith(".zip"):
                inner = os.path.join(folder, name)
    if inner is None or sha256(inner) != GESTURES_INNER_SHA256:
        raise SystemExit("Gesture dataset checksum mismatch; not extracting it.")
    with zipfile.ZipFile(inner) as archive:  # extractall strips "..", so it can't escape
        archive.extractall(os.path.join(RAW, "images"))
    print("done")


if __name__ == "__main__":
    main()
