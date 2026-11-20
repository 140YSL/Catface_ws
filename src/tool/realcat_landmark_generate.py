#!/usr/bin/env python3
"""Annotate every video frame, running the cat landmark models every N frames.

Requires Python 3.10/3.11, cat-landmark-portable/requirements-ml.txt and
opencv-python. Output is a silent, constant-frame-rate MP4; source audio is
not copied. For variable-frame-rate sources, the reported nominal FPS is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys


PROJECT_SRC = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_SRC / "cat-landmark-portable" / "models"


def load_pipeline():
    """Reuse the existing preprocessing and crop-to-image coordinate contract."""
    source = PROJECT_SRC / "cat-landmark-portable" / "tools" / "extract_landmarks_batch.py"
    print("source:", source)
    spec = importlib.util.spec_from_file_location("cat_landmark_video_pipeline", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inference helpers: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default="/home/ubuntu/Desktop/Catface_ws/cat_video/organized/afraid/1.mp4")
    parser.add_argument("--output", type=Path, default="/home/ubuntu/Desktop/Catface_ws/cat_video/temp/1.mp4")
    parser.add_argument("--stride", type=int, default=1,
                        help="Predict on frames 0, N, 2N, ... (default: 5)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--bbox-model", type=Path, help="Override face-localizer .tflite")
    parser.add_argument("--landmark-model", type=Path, help="Override landmark .tflite")
    parser.add_argument("--crop-margin", type=float, default=0.10)
    parser.add_argument("--hide-labels", action="store_true", help="Hide landmark numbers 0-47")
    parser.add_argument("--show-boxes", action="store_true", help="Draw face and crop boxes")
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if not 0 <= args.crop_margin <= 1:
        parser.error("--crop-margin must be between 0 and 1")
    return args


def predict(frame, localizer, landmarker, pipeline, margin):
    import cv2
    import numpy as np
    from PIL import Image

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor, metadata = pipeline.letterbox_image(image, localizer.input_size)
    bbox_values = localizer.predict(tensor)
    if not np.isfinite(bbox_values).all():
        raise ValueError("Non-finite face box")
    bbox = pipeline.deletterbox_bbox(bbox_values, metadata)
    tensor, crop = pipeline.crop_for_landmarks(image, bbox, margin, landmarker.input_size)
    normalized = landmarker.predict(tensor)
    if normalized.size != 96 or not np.isfinite(normalized).all():
        raise ValueError("Expected 96 finite landmark coordinates")
    points = pipeline.landmarks_to_pixels(np.clip(normalized, 0, 1), crop)
    return bbox, crop, points


def draw(frame, result, pipeline, labels, boxes):
    import cv2

    bbox, crop, points = result
    radius = max(2, min(frame.shape[:2]) // 180)
    if boxes:
        cv2.rectangle(frame, tuple(int(v) for v in bbox[:2]),
                      tuple(int(v) for v in bbox[2:]), (0, 0, 255), 2)
        cv2.rectangle(frame, (round(crop["crop_x1"]), round(crop["crop_y1"])),
                      (round(crop["crop_x2"]), round(crop["crop_y2"])), (0, 255, 255), 1)
    for index, (x, y) in enumerate(points):
        color = tuple(reversed(pipeline.LANDMARK_COLORS[index]))
        center = (round(float(x)), round(float(y)))
        cv2.circle(frame, center, radius + 1, (0, 0, 0), -1)
        cv2.circle(frame, center, radius, color, -1)
        if labels:
            origin = (center[0] + radius + 2, center[1] - radius - 2)
            cv2.putText(frame, str(index), origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, str(index), origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)


def generate(args):
    import cv2

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if source == output:
        raise ValueError("Input and output must be different files")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must use the .mp4 extension")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {output}")
    model_dir = args.model_dir.expanduser().resolve()
    bbox_model = (args.bbox_model or model_dir / "cat_face_localizer.tflite").expanduser().resolve()
    landmark_model = (args.landmark_model or model_dir / "cat_face_landmarks_full.tflite").expanduser().resolve()
    for model in (bbox_model, landmark_model):
        if not model.is_file():
            raise FileNotFoundError(f"Missing model: {model}; run tools/download_models.py --kind inference in cat-landmark-portable")
    pipeline = load_pipeline()
    localizer = pipeline.TFLiteRunner(bbox_model)
    landmarker = pipeline.TFLiteRunner(landmark_model)
    capture = cv2.VideoCapture(str(source))
    writer = None
    frame_index = failures = predictions = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid FPS")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Video has no decodable frames")
        height, width = frame.shape[:2]
        # OpenCV's MP4 encoder silently truncates odd dimensions.
        if width % 2 or height % 2:
            raise ValueError("MP4 encoding requires even video width and height")
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot initialize MP4 writer: {output}")
        print(f"{width}x{height}, {fps:g} FPS; predicting every {args.stride} frames", flush=True)
        print("Output contains video only (no audio).", flush=True)
        last_result = None
        while ok:
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Video dimensions changed at frame {frame_index}")
            if frame_index % args.stride == 0:
                predictions += 1
                try:
                    last_result = predict(frame, localizer, landmarker, pipeline, args.crop_margin)
                except ValueError as error:
                    # Clear stale points after a failed prediction until the next sample.
                    last_result = None
                    failures += 1
                    print(f"Frame {frame_index}: prediction failed: {error}", file=sys.stderr)
            if last_result is not None:
                draw(frame, last_result, pipeline, not args.hide_labels, args.show_boxes)
            writer.write(frame)
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Written {frame_index} frames", flush=True)
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    print(f"Saved {output}\nFrames: {frame_index}; prediction attempts: {predictions}; failures: {failures}")
    return 0


def main():
    args = parse_args()
    try:
        return generate(args)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        print("If interrupted, the output may be partial; use a new output path when retrying.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
