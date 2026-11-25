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


PROJECT_SRC = next(parent for parent in Path(__file__).resolve().parents
                   if (parent / "cat-landmark-portable").is_dir())
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
    parser.add_argument("--input", type=Path, default="/home/ubuntu/Desktop/Catface_ws/cat_video/test/seedance_test2.mp4")
    parser.add_argument("--output", type=Path, default="/home/ubuntu/Desktop/Catface_ws/cat_video/test/seedance_test2_landmark.mp4")
    parser.add_argument("--stride", type=int, default=1,
                        help="Predict on frames 0, N, 2N, ... (default: 1)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--bbox-model", type=Path, help="Override face-localizer .tflite")
    parser.add_argument("--landmark-model", type=Path, help="Override landmark .tflite")
    parser.add_argument("--crop-margin", type=float, default=0.10)
    parser.add_argument("--hide-labels", action="store_true", help="Hide landmark numbers 0-47")
    parser.add_argument("--show-boxes", dest="show_boxes", action="store_true", default=True,
                        help="Draw face box, four corners and crop box (default: enabled)")
    parser.add_argument("--hide-boxes", dest="show_boxes", action="store_false",
                        help="Hide boxes in video; coordinates are still saved")
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
        for name, (x, y) in bbox_corners(bbox).items():
            # Drawing is clipped to the image; stored values retain model precision.
            center = (max(0, min(frame.shape[1]-1, round(float(x)))),
                      max(0, min(frame.shape[0]-1, round(float(y)))))
            cv2.circle(frame, center, radius+2, (0, 0, 255), -1)
            origin = (max(0, min(frame.shape[1]-24, center[0]+5)),
                      max(14, min(frame.shape[0]-4, center[1]-5)))
            cv2.putText(frame, name, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 255), 1, cv2.LINE_AA)
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


def bbox_corners(bbox):
    """Axis-aligned face box corners in original image pixels, clockwise."""
    x1, y1, x2, y2 = bbox
    return {'TL': (x1, y1), 'TR': (x2, y1),
            'BR': (x2, y2), 'BL': (x1, y2)}


def generate(args):
    import cv2
    import numpy as np

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if source == output:
        raise ValueError("Input and output must be different files")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must use the .mp4 extension")
    npy_path = output.with_suffix('.npy')
    if output.exists() or npy_path.exists():
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
    records = []
    fields = [('frame_index', 'i8'), ('timestamp_s', 'f8'),
              ('source_timestamp_s', 'f8'), ('source_timestamp_valid', '?'),
              ('fps', 'f8'), ('image_width', 'i4'), ('image_height', 'i4'),
              ('prediction_attempted', '?'), ('valid', '?'), ('prediction_frame', 'i8')]
    fields += [(f'{prefix}_{axis}_px', 'f8') for prefix in ('bbox', 'crop')
               for axis in ('x1', 'y1', 'x2', 'y2')]
    fields += [(f'bbox_{corner}_{axis}_px', 'f8')
               for corner in ('top_left', 'top_right', 'bottom_right', 'bottom_left')
               for axis in ('x', 'y')]
    fields += [(f'landmark_{i:02d}_{axis}_px', 'f8') for i in range(48) for axis in ('x', 'y')]
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
        prediction_frame = -1
        previous_source_time = -1.0
        while ok:
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Video dimensions changed at frame {frame_index}")
            if frame_index % args.stride == 0:
                predictions += 1
                try:
                    last_result = predict(frame, localizer, landmarker, pipeline, args.crop_margin)
                    prediction_frame = frame_index
                except ValueError as error:
                    # Clear stale points after a failed prediction until the next sample.
                    last_result = None
                    prediction_frame = -1
                    failures += 1
                    print(f"Frame {frame_index}: prediction failed: {error}", file=sys.stderr)
            source_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            source_time_valid = math.isfinite(source_time) and source_time >= 0 and source_time > previous_source_time
            if source_time_valid:
                previous_source_time = source_time
            row = np.zeros((), dtype=fields)
            for name in row.dtype.names:
                if row.dtype[name].kind == 'f':
                    row[name] = np.nan
            for name, value in dict(frame_index=frame_index, timestamp_s=frame_index/fps,
                                    source_timestamp_s=source_time if source_time_valid else np.nan,
                                    source_timestamp_valid=source_time_valid, fps=fps,
                                    image_width=width, image_height=height,
                                    prediction_attempted=frame_index % args.stride == 0,
                                    valid=last_result is not None, prediction_frame=prediction_frame).items():
                row[name] = value
            if last_result is not None:
                bbox, crop, points = last_result
                for axis, value in zip(('x1', 'y1', 'x2', 'y2'), bbox):
                    row[f'bbox_{axis}_px'] = value
                for name, point in zip(('top_left', 'top_right', 'bottom_right', 'bottom_left'),
                                       bbox_corners(bbox).values()):
                    row[f'bbox_{name}_x_px'], row[f'bbox_{name}_y_px'] = point
                for axis in ('x1', 'y1', 'x2', 'y2'):
                    row[f'crop_{axis}_px'] = crop[f'crop_{axis}']
                for i, point in enumerate(points):
                    row[f'landmark_{i:02d}_x_px'], row[f'landmark_{i:02d}_y_px'] = point
                draw(frame, last_result, pipeline, not args.hide_labels, args.show_boxes)
            records.append(row)
            writer.write(frame)
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Written {frame_index} frames", flush=True)
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    np.save(npy_path, np.array(records, dtype=fields), allow_pickle=False)
    print(f"Saved {output}\nLandmarks: {npy_path}\nFrames: {frame_index}; prediction attempts: {predictions}; failures: {failures}")
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
