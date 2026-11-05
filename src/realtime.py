"""Real-time sign recognition from a webcam. Press q or Esc to quit, c to clear the text."""
import argparse
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from features import HAND_CONNECTIONS, to_isotropic
from model import SignClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = "Sign Language Translator"
OTHER = "other"
DISPLAY = {
    "hello": "Hello", "i_love_you": "I Love You", "peace": "Peace",
    "thank_you": "Thank You", "yes": "Yes", OTHER: "(no sign)",
}
SMOOTHING = 0.35        # weight of the newest frame in the running average
FRAMES_TO_FORGET = 5    # frames without a hand before the running average resets
HOLD_SECONDS = 0.6      # how long a sign must be held before it is added to the text
PANEL_WIDTH = 250

GREEN, GREY, WHITE, DARK = (90, 210, 90), (150, 150, 150), (245, 245, 245), (32, 32, 32)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hand(frame, pixels, label, confidence, recognised):
    color = GREEN if recognised else GREY
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pixels[a]), tuple(pixels[b]), color, 2, cv2.LINE_AA)
    for point in pixels:
        cv2.circle(frame, tuple(point), 4, WHITE, -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(point), 4, color, 1, cv2.LINE_AA)

    height, width = frame.shape[:2]
    x0, y0 = np.maximum(pixels.min(axis=0) - 20, 0)
    x1, y1 = np.minimum(pixels.max(axis=0) + 20, [width - 1, height - 1])
    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, 2, cv2.LINE_AA)

    text = f"{label} {confidence * 100:.0f}%" if recognised else label
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    tag_x = int(min(x0, width - text_w - 12))
    tag_y = int(max(y0, text_h + 12))
    cv2.rectangle(frame, (tag_x, tag_y - text_h - 12), (tag_x + text_w + 12, tag_y), color, -1)
    cv2.putText(frame, text, (tag_x + 6, tag_y - 6), FONT, 0.8, DARK, 2, cv2.LINE_AA)


def draw_panel(height, classes, probabilities, threshold, fps, latency_ms, hand_visible):
    panel = np.full((height, PANEL_WIDTH, 3), DARK, dtype=np.uint8)
    cv2.putText(panel, "Confidence", (14, 30), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
    top = int(np.argmax(probabilities)) if hand_visible else -1
    bar_width = PANEL_WIDTH - 28
    for i, name in enumerate(classes):
        y = 54 + i * 50
        value = float(probabilities[i]) if hand_visible else 0.0
        cv2.putText(panel, DISPLAY[name], (14, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{value * 100:3.0f}%", (PANEL_WIDTH - 58, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(panel, (14, y + 7), (14 + bar_width, y + 21), (62, 62, 62), -1)
        passes = i == top and name != OTHER and value >= threshold
        cv2.rectangle(panel, (14, y + 7), (14 + int(bar_width * value), y + 21), GREEN if passes else GREY, -1)
        if name != OTHER:
            tick = 14 + int(bar_width * threshold)
            cv2.line(panel, (tick, y + 4), (tick, y + 24), WHITE, 1)
    cv2.putText(panel, f"{fps:.0f} FPS   {latency_ms:.0f} ms/frame", (14, height - 16), FONT, 0.45, GREY, 1, cv2.LINE_AA)
    return panel


def open_source(args):
    if args.source:
        capture = cv2.VideoCapture(args.source)
    else:
        capture = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = capture.read() if capture.isOpened() else (False, None)
    if not ok:
        if args.source:
            raise SystemExit(f"Could not read video: {args.source}")
        raise SystemExit(
            f"Could not read from camera {args.camera}.\n"
            "If macOS just asked for camera access, allow it and run this again.\n"
            "If it didn't ask: System Settings > Privacy & Security > Camera, allow the app\n"
            "you launched this from (Terminal, VS Code...), then restart that app.\n"
            "If another app is using the camera, close it. To try a different one: --camera 1"
        )
    return capture, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index (try 1 if an iPhone camera is picked)")
    parser.add_argument("--threshold", type=float, help="confidence needed to name a sign (default: from training)")
    parser.add_argument("--classifier", default=os.path.join(ROOT, "models/sign_classifier.npz"))
    parser.add_argument("--landmarker", default=os.path.join(ROOT, "models/hand_landmarker.task"))
    parser.add_argument("--source", help=argparse.SUPPRESS)  # video file, for testing without a camera
    parser.add_argument("--no-window", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-frames", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--save", help=argparse.SUPPRESS)  # write the annotated video here
    parser.add_argument("--log", help=argparse.SUPPRESS)  # write per-frame predictions here
    args = parser.parse_args()

    cv2.setNumThreads(1)
    classifier = SignClassifier(args.classifier)
    classes = classifier.classes
    other_id = classes.index(OTHER)
    threshold = args.threshold if args.threshold is not None else classifier.threshold

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.landmarker),
        running_mode=RunningMode.VIDEO,
        num_hands=1,  # with 1, the costly palm detector runs only when no hand is being tracked
    )
    capture, frame = open_source(args)
    source_fps = capture.get(cv2.CAP_PROP_FPS) if args.source else 0

    if not args.no_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    smoothed = np.full(len(classes), 1 / len(classes), dtype=np.float32)
    missing, count, held_id, held_since = FRAMES_TO_FORGET, 0, -1, 0.0
    fps, latency_ms, transcript = 0.0, 0.0, []
    writer, log_rows = None, []
    started = time.perf_counter()
    cpu_started = resource.getrusage(resource.RUSAGE_SELF)
    frame_period = 1.0 / source_fps if source_fps else 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            tick = time.perf_counter()
            frame = cv2.flip(frame, 1)  # selfie view; the classifier was trained on both mirrorings
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = int(count * frame_period * 1000) if frame_period else int((tick - started) * 1000)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
            )

            if result.hand_landmarks:
                raw = np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], dtype=np.float32)
                current = classifier.predict_points(to_isotropic(raw, width, height))
                smoothed = current if missing >= FRAMES_TO_FORGET else SMOOTHING * current + (1 - SMOOTHING) * smoothed
                missing = 0
                pixels = np.round(raw[:, :2] * [width, height]).astype(int)
            else:
                missing += 1

            # The tracker can drop a hand for a frame or two; ride through that without
            # flicker, keeping the previous result, instead of treating it as a new start.
            hand_visible = missing < FRAMES_TO_FORGET
            recognised, top, confidence = False, other_id, 0.0
            if hand_visible:
                top = int(np.argmax(smoothed))
                confidence = float(smoothed[top])
                recognised = top != other_id and confidence >= threshold
                draw_hand(frame, pixels, DISPLAY[classes[top]] if recognised else DISPLAY[OTHER], confidence, recognised)

            # Add a sign to the text once it has been held steadily, and not twice in a row.
            now = time.perf_counter()
            clock = count * frame_period if frame_period else now - started
            current_id = top if recognised else -1
            if current_id != held_id:
                held_id, held_since = current_id, clock
            if held_id >= 0 and clock - held_since >= HOLD_SECONDS:
                word = DISPLAY[classes[held_id]]
                if not transcript or transcript[-1] != word:
                    transcript.append(word)
            if not hand_visible and transcript and transcript[-1] != "":
                transcript.append("")  # lowering the hand lets the same sign be repeated

            elapsed = now - tick
            latency_ms = 0.9 * latency_ms + 100 * elapsed if count else elapsed * 1000
            count += 1

            panel = draw_panel(height, classes, smoothed, threshold, fps, latency_ms, hand_visible)
            canvas = np.hstack([frame, panel])
            words = [w for w in transcript if w]
            caption = "  ".join(words[-5:]) if words else "Show a sign to the camera"
            cv2.rectangle(canvas, (0, height - 42), (width, height), DARK, -1)
            cv2.putText(canvas, caption, (12, height - 13), FONT, 0.7, WHITE, 1, cv2.LINE_AA)

            if args.save:
                if writer is None:
                    size = (canvas.shape[1], canvas.shape[0])
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), source_fps or 30, size)
                writer.write(canvas)
            if args.log:
                log_rows.append((count, hand_visible, classes[top] if recognised else OTHER, confidence, elapsed * 1000))

            if not args.no_window:
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    transcript.clear()
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if args.max_frames and count >= args.max_frames:
                break
            if frame_period:  # play test videos at their real speed so load figures are realistic
                time.sleep(max(0.0, frame_period - (time.perf_counter() - tick)))
            ok, frame = capture.read()
            if not ok:
                break
            fps = count / max(time.perf_counter() - started, 1e-6)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    total = time.perf_counter() - started
    cpu_now = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (cpu_now.ru_utime + cpu_now.ru_stime) - (cpu_started.ru_utime + cpu_started.ru_stime)
    # macOS reports peak memory in bytes.
    print(f"{count} frames in {total:.1f}s = {count / total:.1f} FPS; "
          f"CPU {cpu / total * 100:.0f}% of one core; peak memory {cpu_now.ru_maxrss / 1e6:.0f} MB")
    print("text:", "  ".join(w for w in transcript if w))
    if args.log:
        with open(args.log, "w") as f:
            f.write("frame,hand,label,confidence,ms\n")
            for row in log_rows:
                f.write(",".join(str(v) for v in row) + "\n")


if __name__ == "__main__":
    main()
