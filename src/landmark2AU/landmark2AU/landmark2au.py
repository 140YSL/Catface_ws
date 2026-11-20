#!/usr/bin/env python3
"""Extract 11 normalized motion proxies and linearly map them to AU 0..5.

Ear pitch is a signed 2D shape proxy, NOT an identifiable 3D rotation.
Defaults are exploratory calibration, not CatFACS intensity ground truth.
Left/right refer to the cat's anatomical side (right is image-left).
"""

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    'left_eye_close', 'right_eye_close',
    'left_ear_outward', 'right_ear_outward',
    'left_ear_forward', 'right_ear_forward',
    'left_ear_backward', 'right_ear_backward',
    'upper_lip_up', 'jaw_down', 'upper_lip_down',
)


def extract_geometry(points):
    """Return roll-aligned, dimensionless geometric measurements per frame."""
    right = (points[:, 4] + points[:, 5]) / 2
    left = (points[:, 8] + points[:, 9]) / 2
    delta = left - right
    scale = np.linalg.norm(delta, axis=1)
    scale[scale < 1e-6] = np.nan
    horizontal = delta / scale[:, None]
    vertical = np.stack((-horizontal[:, 1], horizontal[:, 0]), axis=1)
    centered = points - ((left + right)/2)[:, None, :]
    p = np.stack((np.einsum('nij,nj->ni', centered, horizontal),
                  np.einsum('nij,nj->ni', centered, vertical)), axis=2) / scale[:, None, None]
    g = {}
    for side, outer, inner, upper, lower in (
            ('right', 4, 5, 6, 7), ('left', 8, 9, 10, 11)):
        axis = p[:, inner] - p[:, outer]
        width = np.linalg.norm(axis, axis=1)
        width[width < 1e-6] = np.nan
        normal = np.stack((-axis[:, 1], axis[:, 0]), axis=1) / width[:, None]
        g[f'{side}_eye_open'] = np.abs(np.sum((p[:, lower]-p[:, upper])*normal, axis=1))/width
    for side, a, tip, b, sign in (('right', 22, 24, 26, -1), ('left', 27, 29, 31, 1)):
        base = (p[:, a] + p[:, b])/2
        vector = p[:, tip] - base
        width = np.linalg.norm(p[:, a]-p[:, b], axis=1)
        width[width < 1e-6] = np.nan
        # Outward tilt in the image, relative to the face's vertical axis.
        g[f'{side}_ear_angle'] = np.arctan2(sign*vector[:, 0], -vector[:, 1])
        # Projected ear height/base-width: depth rotations are ambiguous in 2D.
        g[f'{side}_ear_height'] = -vector[:, 1]/width
    nose = (p[:, 12]+p[:, 13])/2
    g['lip_height'] = p[:, 17, 1]-nose[:, 1]
    g['jaw_height'] = p[:, 2, 1]-nose[:, 1]
    return g


def convert(source, output, config=None, neutral_start=0, neutral_end=1):
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'Output already exists: {output}')
    data = np.load(source, allow_pickle=False)
    required = ['frame_index', 'timestamp_s', 'valid', 'prediction_frame']
    required += [f'landmark_{i:02d}_{a}_px' for i in range(48) for a in ('x', 'y')]
    if not data.dtype.names or any(name not in data.dtype.names for name in required):
        raise ValueError('Expected a structured landmark NPY from realcat_landmark_generate.py')
    if data.ndim != 1 or len(data) == 0:
        raise ValueError('Expected a nonempty one-dimensional frame table')
    points = np.stack([data[f'landmark_{i:02d}_{a}_px'] for i in range(48) for a in ('x', 'y')], axis=1).reshape(-1, 48, 2)
    valid = data['valid'] & np.isfinite(points).all(axis=(1, 2))
    points[~valid] = np.nan
    geometry = extract_geometry(points)
    valid &= np.isfinite(np.stack(list(geometry.values()), axis=1)).all(axis=1)
    config = {} if config is None else config
    baseline = dict(config.get('baseline', {}))
    neutral = valid & (data['timestamp_s'] >= neutral_start) & (data['timestamp_s'] < neutral_end)
    # Use independent predictions so stride holds do not bias the reference.
    fresh_neutral = neutral & (data['prediction_frame'] == data['frame_index'])
    neutral = fresh_neutral if fresh_neutral.any() else neutral
    for name, values in geometry.items():
        if name not in baseline:
            if not neutral.any():
                raise ValueError(f'No valid reference frames for {name}; select another neutral interval or provide baseline')
            baseline[name] = float(np.median(values[neutral]))
        if not np.isfinite(baseline[name]):
            raise ValueError(f'Non-finite baseline: {name}')
    metrics = {}
    for side in ('left', 'right'):
        open_ref = baseline[f'{side}_eye_open']
        if open_ref <= 1e-6:
            raise ValueError(f'{side} eye reference must be open, not fully closed')
        metrics[f'{side}_eye_close'] = (open_ref-geometry[f'{side}_eye_open'])/open_ref
        metrics[f'{side}_ear_outward'] = geometry[f'{side}_ear_angle']-baseline[f'{side}_ear_angle']
        pitch = baseline[f'{side}_ear_height']-geometry[f'{side}_ear_height']
        pitch_sign = config.get('ear_pitch_sign', {}).get(side, 1)
        if pitch_sign not in (-1, 1):
            raise ValueError('ear_pitch_sign must be +1 or -1')
        pitch = pitch*pitch_sign
        metrics[f'{side}_ear_forward'] = np.maximum(pitch, 0)
        metrics[f'{side}_ear_backward'] = np.maximum(-pitch, 0)
    lip_delta = geometry['lip_height']-baseline['lip_height']
    metrics['upper_lip_up'] = np.maximum(-lip_delta, 0)
    metrics['upper_lip_down'] = np.maximum(lip_delta, 0)
    metrics['jaw_down'] = geometry['jaw_height']-baseline['jaw_height']
    fields = [('frame_index', 'i8'), ('timestamp_s', 'f8'), ('valid', '?'), ('prediction_frame', 'i8')]
    fields += [(f'{name}_norm', 'f8') for name in METRICS]
    fields += [(f'AU_{name}', 'f8') for name in METRICS]
    result = np.empty(len(data), dtype=fields)
    for name in required[:4]:
        result[name] = data[name]
    result['valid'] = valid
    endpoints = {}
    for name in METRICS:
        # Radians for outward tilt; ratios for all other normalized measures.
        full = 1.0 if 'eye_close' in name else (np.pi/4 if 'outward' in name else 0.25)
        zero, full = config.get('mapping', {}).get(name, [0, full])
        if not np.isfinite([zero, full]).all() or zero == full:
            raise ValueError(f'Invalid linear endpoints for {name}')
        values = metrics[name].copy()
        values[~valid] = np.nan
        result[f'{name}_norm'] = values
        result[f'AU_{name}'] = np.clip(5*(values-zero)/(full-zero), 0, 5)
        endpoints[name] = [float(zero), float(full)]
    metadata = {
        'source': str(source.resolve()), 'metrics': list(METRICS),
        'baseline': baseline, 'mapping': endpoints,
        'ear_pitch_sign': config.get('ear_pitch_sign', {'left': 1, 'right': 1}),
        'reference_interval_s': [neutral_start, neutral_end],
        'normalization': 'eye-corner local width for eyelids; interocular scale and roll alignment for mouth; local ear base width for ear height',
        'warning': 'Exploratory 2D proxies. Ear forward/backward labels are assumed signed height changes, not observable physical 3D angles; validate and calibrate. Reference interval is NOT automatically neutral. Defaults are not calibrated CatFACS intensities.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result, allow_pickle=False)
    output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Saved {output}: {len(result)} frames, {valid.sum()} valid; 11 normalized proxies and 11 AU columns')
    print('WARNING: reference and AU endpoints require calibration; ear pitch direction is a 2D assumption.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Landmark .npy or folder of .npy files')
    parser.add_argument('--output-dir', type=Path, help='Default: INPUT_FOLDER/au')
    parser.add_argument('--config', type=Path, help='JSON baseline, mapping endpoints, ear_pitch_sign; previous output JSON can be reused')
    parser.add_argument('--neutral-start', type=float, default=0)
    parser.add_argument('--neutral-end', type=float, default=1, help='Reference interval end, seconds (default: 1)')
    args = parser.parse_args(argv)
    if not 0 <= args.neutral_start < args.neutral_end:
        parser.error('Require 0 <= neutral-start < neutral-end')
    source = args.input.expanduser().resolve()
    if not source.exists():
        parser.error(f'Input does not exist: {source}')
    files = sorted(source.glob('*.npy')) if source.is_dir() else [source]
    if not files:
        parser.error('No .npy files found')
    output_dir = args.output_dir or ((source if source.is_dir() else source.parent)/'au')
    config = json.loads(args.config.read_text(encoding='utf-8')) if args.config else None
    for path in files:
        convert(path, output_dir/(path.stem+'_au.npy'), config, args.neutral_start, args.neutral_end)


if __name__ == '__main__':
    main()
