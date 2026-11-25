#!/usr/bin/env python3
"""Annotate image folders with face boxes and 48 numbered cat landmarks."""

import argparse
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .realcat_landmark_generate import (
        DEFAULT_MODEL_DIR, PROJECT_SRC, draw, load_pipeline, predict,
    )
else:
    from realcat_landmark_generate import (
        DEFAULT_MODEL_DIR, PROJECT_SRC, draw, load_pipeline, predict,
    )


SIM_DIRECTORY = PROJECT_SRC.parent / 'cat_video' / 'sim'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def generate(source, output, model_dir=DEFAULT_MODEL_DIR, crop_margin=0.10):
    """Preserve every image's relative path, filename, and pixel dimensions."""
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f'Input directory does not exist: {source}')
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Input and output directories must not contain each other')
    if not 0 <= crop_margin <= 1:
        raise ValueError('crop_margin must be between 0 and 1')
    images = sorted(path for path in source.rglob('*')
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f'No images found in {source}')
    for path in images:
        target = output / path.relative_to(source)
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite existing image: {target}')
    model_dir = Path(model_dir).expanduser().resolve()
    bbox_model = model_dir / 'cat_face_localizer.tflite'
    landmark_model = model_dir / 'cat_face_landmarks_full.tflite'
    for model in (bbox_model, landmark_model):
        if not model.is_file():
            raise FileNotFoundError(f'Missing model: {model}')
    pipeline = load_pipeline()
    localizer = pipeline.TFLiteRunner(bbox_model)
    landmarker = pipeline.TFLiteRunner(landmark_model)
    for index, path in enumerate(images, start=1):
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f'Cannot decode image: {path}')
        try:
            result = predict(frame, localizer, landmarker, pipeline, crop_margin)
            # Same red face box, yellow crop box and numbered landmarks as the video tool.
            draw(frame, result, pipeline, labels=True, boxes=True)
        except (ValueError, RuntimeError, cv2.error) as error:
            raise ValueError(f'Cannot annotate {path}: {error}') from error
        target = output / path.relative_to(source)
        success, encoded = cv2.imencode(target.suffix.lower(), frame)
        if not success:
            raise RuntimeError(f'Cannot encode image: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(encoded.tobytes())
        if index % 26 == 0 or index == len(images):
            print(f'Annotated {index}/{len(images)} images: {path.parent.name}', flush=True)
    print(f'Saved {len(images)} annotated images to {output}', flush=True)
    return len(images)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=SIM_DIRECTORY / 'simcat_video_extra')
    parser.add_argument('--output-dir', type=Path,
                        default=SIM_DIRECTORY / 'simcat_video_extra_landmark')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--crop-margin', type=float, default=0.10)
    args = parser.parse_args(argv)
    try:
        generate(args.input, args.output_dir, args.model_dir, args.crop_margin)
    except (OSError, ValueError, RuntimeError, cv2.error) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
