#!/usr/bin/env python3
"""Measure a manually selected neutral image and export an AU baseline config."""

from pathlib import Path

# 修改此处为人工确认全部 AU=0 的中性脸正面图像（Python 常量）。
NEUTRAL_IMAGE_PATH = Path('/home/ubuntu/Desktop/Catface_ws/cat_video/sim/zeroface.png')
# 与视频提取器保持相同的模型和裁剪参数。
MODEL_DIRECTORY = None  # None: use cat-landmark-portable/models
CROP_MARGIN = 0.10

import argparse
import hashlib
import json

import cv2
import numpy as np

if __package__:
    from .landmark2au_analysis import extract_geometry
    from .realcat_landmark_generate import DEFAULT_MODEL_DIR, load_pipeline, predict
else:
    from landmark2au_analysis import extract_geometry
    from realcat_landmark_generate import DEFAULT_MODEL_DIR, load_pipeline, predict


def sha256(path):
    """Record provenance without depending on the current working directory."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def calculate(image_path, model_dir=None, crop_margin=CROP_MARGIN):
    """Export eight individual neutral-face geometry measurements, without intensity mapping."""
    image_path = Path(image_path).expanduser().resolve()
    output = image_path.with_suffix('.json')
    if not image_path.is_file():
        raise FileNotFoundError(f'Neutral image does not exist: {image_path}; set NEUTRAL_IMAGE_PATH')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing calibration: {output}')
    if not 0 <= crop_margin <= 1:
        raise ValueError('crop_margin must be between 0 and 1')
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR).expanduser().resolve()
    bbox_model = model_dir / 'cat_face_localizer.tflite'
    landmark_model = model_dir / 'cat_face_landmarks_full.tflite'
    for path in (bbox_model, landmark_model):
        if not path.is_file():
            raise FileNotFoundError(f'Model does not exist: {path}')
    # imdecode also supports paths containing non-ASCII characters.
    frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f'Cannot decode image: {image_path}')
    pipeline = load_pipeline()
    bbox, crop, points = predict(
        frame, pipeline.TFLiteRunner(bbox_model),
        pipeline.TFLiteRunner(landmark_model), pipeline, crop_margin)
    # Video NPY coordinates are float64; match that arithmetic exactly.
    points = np.asarray(points, dtype=np.float64)
    geometry = extract_geometry(points[None, :, :])
    baseline = {name: float(values[0]) for name, values in geometry.items()}
    if not all(np.isfinite(value) for value in baseline.values()):
        raise ValueError('Degenerate/non-finite neutral landmarks; check the selected image')
    if any(baseline[f'{side}_eye_open'] <= 1e-6 for side in ('left', 'right')):
        raise ValueError('Neutral reference must have visible, open eyes')
    metadata = {
        'source': str(image_path),
        'reference_type': 'manually_selected_neutral_image',
        'baseline': baseline,
        'normalization': 'extract_geometry in landmark2au_analysis.py: local eye width; interocular scale and roll alignment; local ear base width. No yaw/pitch compensation.',
        'warning': 'Neutral semantics are supplied by the user, not verified by the model. Check predicted landmarks manually.',
        'reference_image': {
            'sha256': sha256(image_path),
            'width': int(frame.shape[1]), 'height': int(frame.shape[0]),
            'bbox_xyxy_px': bbox.tolist(),
            'bbox_corners_px': {
                'top_left': [float(bbox[0]), float(bbox[1])],
                'top_right': [float(bbox[2]), float(bbox[1])],
                'bottom_right': [float(bbox[2]), float(bbox[3])],
                'bottom_left': [float(bbox[0]), float(bbox[3])],
            },
            'crop': crop, 'landmarks_px': points.tolist(),
        },
        'models': {
            'bbox': {'path': str(bbox_model), 'sha256': sha256(bbox_model)},
            'landmark': {'path': str(landmark_model), 'sha256': sha256(landmark_model)},
            'crop_margin': crop_margin,
        },
    }
    # Exclusive create protects manually edited reference files.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Saved neutral reference: {output}')
    print(json.dumps(baseline, ensure_ascii=False, indent=2))
    print('Use: python landmark2au_analysis.py --reference "'+str(output)+'"')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=NEUTRAL_IMAGE_PATH,
                        help='Optional override of NEUTRAL_IMAGE_PATH')
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIRECTORY)
    args = parser.parse_args(argv)
    try:
        calculate(args.image, args.model_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
