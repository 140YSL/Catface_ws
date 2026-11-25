#!/usr/bin/env python3
"""Plot 11 AU estimators against labeled AU intensity using one individual reference.

Each AU directory contains AU_NAME-INTENSITY images. All groups use the same
48 landmarks and four face-box corners from the shared zeroface.json.
No intensity mapping, curve fitting, or per-curve rescaling is performed.
"""

from pathlib import Path
import argparse
import csv
import json

import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __package__:
    from .realcat_landmark_generate import DEFAULT_MODEL_DIR, PROJECT_SRC, load_pipeline, predict
else:
    from realcat_landmark_generate import DEFAULT_MODEL_DIR, PROJECT_SRC, load_pipeline, predict


SIM_DIRECTORY = PROJECT_SRC.parent / 'cat_video' / 'sim'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


# Names identify AU estimators, not calibrated AU intensities.
METRICS = (
    'LAU143_estimate', 'RAU143_estimate',
    'LEAD104_estimate', 'READ104_estimate',
    'LEAD101_estimate', 'READ101_estimate',
    'LEAD106_estimate', 'READ106_estimate',
    'AU109-110_estimate', 'AU26_estimate', 'AU118_estimate',
)
DEFAULT_REFERENCE = PROJECT_SRC.parent / 'cat_video' / 'sim' / 'zeroface.json'


def pack_points(landmarks, bbox):
    """构造原图像素坐标 (52, 2)：0–47 关键点，48–51 脸框 TL/TR/BR/BL。"""
    landmarks = np.asarray(landmarks, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64)
    if landmarks.shape != (48, 2) or bbox.shape != (4,):
        raise ValueError('Expected 48 landmarks and bbox [x1, y1, x2, y2]')
    x1, y1, x2, y2 = bbox
    if not np.isfinite(bbox).all() or x2 <= x1 or y2 <= y1:
        raise ValueError('Invalid face bounding box')
    result = np.concatenate((landmarks, [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
    if not np.isfinite(result).all():
        raise ValueError('Non-finite landmark coordinates')
    return result


def AU26_estimate(points, zeroface_points):
    """点2相对鼻点12、13中点的纵向高度差：当前减中性。

    两份输入均为原图的 (52, 2) 点集，函数内完成全部几何及相对中性计算。
    """
    return 0.0
    measurements = []
    for raw in (points, zeroface_points):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (52, 2) or not np.isfinite(raw).all():
            raise ValueError('Expected 52 finite pixel points: landmarks 0–47, box TL/TR/BR/BL 48–51')
        # 48–51 为脸框角点，保留输入供后续公式验证；当前公式不使用脸框。
        # 每个 AU 独立对当前点和中性点执行相同坐标处理，不缓存中间特征。
        right = (raw[4] + raw[5]) / 2
        left = (raw[8] + raw[9]) / 2
        delta = left - right
        scale = np.linalg.norm(delta)
        if scale < 1e-6:
            raise ValueError('Degenerate interocular distance')
        horizontal = delta / scale
        vertical = np.array([-horizontal[1], horizontal[0]])
        centered = raw - (left + right) / 2
        p = np.stack((centered @ horizontal, centered @ vertical), axis=1) / scale
        nose = (p[12] + p[13]) / 2
        measurements.append(p[2, 1] - nose[1])
    current, neutral = measurements
    return float(current - neutral)


def AU109_110_estimate(points, zeroface_points):
    """上唇点17相对鼻点12、13中点的高度：中性减当前后取正半轴。

    两份输入均为原图的 (52, 2) 点集，函数内完成全部几何及相对中性计算。
    """
    return 0.0
    measurements = []
    for raw in (points, zeroface_points):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (52, 2) or not np.isfinite(raw).all():
            raise ValueError('Expected 52 finite pixel points: landmarks 0–47, box TL/TR/BR/BL 48–51')
        # 48–51 为脸框角点，保留输入供后续公式验证；当前公式不使用脸框。
        # 每个 AU 独立对当前点和中性点执行相同坐标处理，不缓存中间特征。
        right = (raw[4] + raw[5]) / 2
        left = (raw[8] + raw[9]) / 2
        delta = left - right
        scale = np.linalg.norm(delta)
        if scale < 1e-6:
            raise ValueError('Degenerate interocular distance')
        horizontal = delta / scale
        vertical = np.array([-horizontal[1], horizontal[0]])
        centered = raw - (left + right) / 2
        p = np.stack((centered @ horizontal, centered @ vertical), axis=1) / scale
        nose = (p[12] + p[13]) / 2
        measurements.append(p[17, 1] - nose[1])
    current, neutral = measurements
    return float(max(neutral - current, 0))


def AU118_estimate(points, zeroface_points):
    """当前候选为上唇下降：点17相对鼻高度，当前减中性后取正半轴。

    两份输入均为原图的 (52, 2) 点集，函数内完成全部几何及相对中性计算。
    """
    return 0.0
    measurements = []
    for raw in (points, zeroface_points):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (52, 2) or not np.isfinite(raw).all():
            raise ValueError('Expected 52 finite pixel points: landmarks 0–47, box TL/TR/BR/BL 48–51')
        # 48–51 为脸框角点，保留输入供后续公式验证；当前公式不使用脸框。
        # 每个 AU 独立对当前点和中性点执行相同坐标处理，不缓存中间特征。
        right = (raw[4] + raw[5]) / 2
        left = (raw[8] + raw[9]) / 2
        delta = left - right
        scale = np.linalg.norm(delta)
        if scale < 1e-6:
            raise ValueError('Degenerate interocular distance')
        horizontal = delta / scale
        vertical = np.array([-horizontal[1], horizontal[0]])
        centered = raw - (left + right) / 2
        p = np.stack((centered @ horizontal, centered @ vertical), axis=1) / scale
        nose = (p[12] + p[13]) / 2
        measurements.append(p[17, 1] - nose[1])
    current, neutral = measurements
    return float(max(current - neutral, 0))


def AU143_estimate(points, zeroface_points, side):
    """左右眼局部眼睑间距/眼角宽度，返回相对中性眼开度的闭合比例。

    两份输入均为原图的 (52, 2) 点集，函数内完成全部几何及相对中性计算。
    """
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    measurements = []
    for raw in (points, zeroface_points):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (52, 2) or not np.isfinite(raw).all():
            raise ValueError('Expected 52 finite pixel points: landmarks 0–47, box TL/TR/BR/BL 48–51')
        # 48–51 为脸框角点，保留输入供后续公式验证；当前公式不使用脸框。
        # 每个 AU 独立对当前点和中性点执行相同坐标处理，不缓存中间特征。
        right = (raw[4] + raw[5]) / 2
        left = (raw[8] + raw[9]) / 2
        delta = left - right
        scale = np.linalg.norm(delta)
        if scale < 1e-6:
            raise ValueError('Degenerate interocular distance')
        horizontal = delta / scale
        vertical = np.array([-horizontal[1], horizontal[0]])
        centered = raw - (left + right) / 2
        p = np.stack((centered @ horizontal, centered @ vertical), axis=1) / scale
        outer, inner, upper, lower = {
            'right': (4, 5, 6, 7), 'left': (8, 9, 10, 11),
        }[side]
        axis = p[inner] - p[outer]
        width = np.linalg.norm(axis)
        if width < 1e-6:
            raise ValueError('Degenerate eye width')
        normal = np.array([-axis[1], axis[0]]) / width
        measurements.append(abs(np.sum((p[lower] - p[upper]) * normal)) / width)
    current, neutral = measurements
    if neutral <= 1e-6:
        raise ValueError('Neutral reference eye must be open')
    return float((neutral - current) / neutral)


def EAD101_estimate(points, zeroface_points, side, pitch_sign=1):
    """耳尖相对耳根的投影高度/耳根宽度，中性减当前乘方向符号后取正半轴。

    两份输入均为原图的 (52, 2) 点集，函数内完成全部几何及相对中性计算。
    """
    return 0.0
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    if pitch_sign not in (-1, 1):
        raise ValueError('pitch_sign must be +1 or -1')
    measurements = []
    for raw in (points, zeroface_points):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (52, 2) or not np.isfinite(raw).all():
            raise ValueError('Expected 52 finite pixel points: landmarks 0–47, box TL/TR/BR/BL 48–51')
        # 48–51 为脸框角点，保留输入供后续公式验证；当前公式不使用脸框。
        # 每个 AU 独立对当前点和中性点执行相同坐标处理，不缓存中间特征。
        right = (raw[4] + raw[5]) / 2
        left = (raw[8] + raw[9]) / 2
        delta = left - right
        scale = np.linalg.norm(delta)
        if scale < 1e-6:
            raise ValueError('Degenerate interocular distance')
        horizontal = delta / scale
        vertical = np.array([-horizontal[1], horizontal[0]])
        centered = raw - (left + right) / 2
        p = np.stack((centered @ horizontal, centered @ vertical), axis=1) / scale
        a, tip, b = {'right': (22, 24, 26), 'left': (27, 29, 31)}[side]
        base = (p[a] + p[b]) / 2
        vector = p[tip] - base
        width = np.linalg.norm(p[a] - p[b])
        if width < 1e-6:
            raise ValueError('Degenerate ear base width')
        measurements.append(-vector[1] / width)
    current, neutral = measurements
    return float(max((neutral - current) * pitch_sign, 0))


def EAD104_estimate(points, zeroface_points, side):
    """耳根中点到耳尖相对面部竖轴的外倾角（弧度），当前减中性。

    两份输入均为原图的 (52, 2) 点集，函数内完成全部几何及相对中性计算。
    """
    return 0.0
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    measurements = []
    for raw in (points, zeroface_points):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (52, 2) or not np.isfinite(raw).all():
            raise ValueError('Expected 52 finite pixel points: landmarks 0–47, box TL/TR/BR/BL 48–51')
        # 48–51 为脸框角点，保留输入供后续公式验证；当前公式不使用脸框。
        # 每个 AU 独立对当前点和中性点执行相同坐标处理，不缓存中间特征。
        right = (raw[4] + raw[5]) / 2
        left = (raw[8] + raw[9]) / 2
        delta = left - right
        scale = np.linalg.norm(delta)
        if scale < 1e-6:
            raise ValueError('Degenerate interocular distance')
        horizontal = delta / scale
        vertical = np.array([-horizontal[1], horizontal[0]])
        centered = raw - (left + right) / 2
        p = np.stack((centered @ horizontal, centered @ vertical), axis=1) / scale
        a, tip, b, sign = {
            'right': (22, 24, 26, -1), 'left': (27, 29, 31, 1),
        }[side]
        base = (p[a] + p[b]) / 2
        vector = p[tip] - base
        measurements.append(np.arctan2(sign * vector[0], -vector[1]))
    current, neutral = measurements
    return float(current - neutral)


def EAD106_estimate(points, zeroface_points, side, pitch_sign=1):
    """耳尖相对耳根的投影高度/耳根宽度，当前减中性乘方向符号后取正半轴。

    两份输入均为原图的 (52, 2) 点集，函数内完成全部几何及相对中性计算。
    """

    return 0.0
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    if pitch_sign not in (-1, 1):
        raise ValueError('pitch_sign must be +1 or -1')
    measurements = []
    for raw in (points, zeroface_points):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (52, 2) or not np.isfinite(raw).all():
            raise ValueError('Expected 52 finite pixel points: landmarks 0–47, box TL/TR/BR/BL 48–51')
        # 48–51 为脸框角点，保留输入供后续公式验证；当前公式不使用脸框。
        # 每个 AU 独立对当前点和中性点执行相同坐标处理，不缓存中间特征。
        right = (raw[4] + raw[5]) / 2
        left = (raw[8] + raw[9]) / 2
        delta = left - right
        scale = np.linalg.norm(delta)
        if scale < 1e-6:
            raise ValueError('Degenerate interocular distance')
        horizontal = delta / scale
        vertical = np.array([-horizontal[1], horizontal[0]])
        centered = raw - (left + right) / 2
        p = np.stack((centered @ horizontal, centered @ vertical), axis=1) / scale
        a, tip, b = {'right': (22, 24, 26), 'left': (27, 29, 31)}[side]
        base = (p[a] + p[b]) / 2
        vector = p[tip] - base
        width = np.linalg.norm(p[a] - p[b])
        if width < 1e-6:
            raise ValueError('Degenerate ear base width')
        measurements.append(-vector[1] / width)
    current, neutral = measurements
    return float(max((current - neutral) * pitch_sign, 0))


def extract_motion(points, zeroface_points, ear_pitch_sign=None):
    """逐张图片、逐个 AU 调用完整公式，不提取/缓存 geometry 或 baseline。

    points: (N, 52, 2)；zeroface_points: (52, 2)。
    """
    points = np.asarray(points, dtype=np.float64)
    zeroface_points = np.asarray(zeroface_points, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (52, 2):
        raise ValueError('Expected current points shape (N, 52, 2)')
    if zeroface_points.shape != (52, 2):
        raise ValueError('Expected neutral points shape (52, 2)')
    signs = {} if ear_pitch_sign is None else ear_pitch_sign
    estimates = {name: [] for name in METRICS}
    for current_points in points:
        estimates['AU26_estimate'].append(AU26_estimate(current_points, zeroface_points))
        estimates['AU109-110_estimate'].append(AU109_110_estimate(current_points, zeroface_points))
        estimates['AU118_estimate'].append(AU118_estimate(current_points, zeroface_points))
        for side, prefix in (('left', 'L'), ('right', 'R')):
            estimates[f'{prefix}AU143_estimate'].append(
                AU143_estimate(current_points, zeroface_points, side))
            estimates[f'{prefix}EAD101_estimate'].append(
                EAD101_estimate(current_points, zeroface_points, side, signs.get(side, 1)))
            estimates[f'{prefix}EAD104_estimate'].append(
                EAD104_estimate(current_points, zeroface_points, side))
            estimates[f'{prefix}EAD106_estimate'].append(
                EAD106_estimate(current_points, zeroface_points, side, signs.get(side, 1)))
    return {name: np.asarray(values, dtype=np.float64) for name, values in estimates.items()}

def load_reference(path):
    """只读取参考图 48 点和脸框；忽略 JSON 中的 baseline。"""
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    image = config['reference_image']
    zeroface_points = pack_points(image['landmarks_px'], image['bbox_xyxy_px'])
    return path, config, zeroface_points


def read_groups(source):
    groups = {}
    for folder in sorted(source.iterdir()):
        if not folder.is_dir():
            continue
        images = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            prefix, separator, label = path.stem.rpartition('-')
            if not separator or prefix != folder.name:
                raise ValueError(f'Expected {folder.name}-INTENSITY image: {path}')
            intensity = float(label)
            if not np.isfinite(intensity) or intensity < 0:
                raise ValueError(f'Invalid intensity: {path}')
            images.append((intensity, path))
        if not images:
            continue
        images.sort(key=lambda item: (item[0], item[1].name))
        labels = [item[0] for item in images]
        if len(labels) != len(set(labels)):
            raise ValueError(f'{folder}: need unique intensities')
        groups[folder.name] = images
    if not groups:
        raise ValueError(f'No AU image groups in {source}')
    return groups


def recognize_images(groups, cache_path, model_dir, crop_margin):
    """读取或生成 {AU名称: {强度字符串: [[x,y], ... 共52点]}}。"""
    cache_path = Path(cache_path).expanduser().resolve()
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        if not isinstance(cached, dict):
            raise ValueError(f'Expected AU groups in recognition cache: {cache_path}')
        recognized = {}
        for au, images in groups.items():
            recognized[au] = {}
            # JSON 对象键只能是字符串；强度按数值匹配，兼容 0 和 0.0。
            entries = cached.get(au, {})
            if not isinstance(entries, dict):
                raise ValueError(f'Invalid cache group: {au}')
            numeric = {}
            for label, coordinates in entries.items():
                intensity = float(label)
                if not np.isfinite(intensity) or intensity in numeric:
                    raise ValueError(f'Invalid/duplicate cache intensity: {au}/{label}')
                numeric[intensity] = coordinates
            for intensity, path in images:
                if intensity not in numeric:
                    raise ValueError(f'Cache missing {au}/{intensity}: {cache_path}; '
                                     'choose a new --cache path to recognize again')
                points = np.asarray(numeric[intensity], dtype=np.float64)
                if points.shape != (52, 2) or not np.isfinite(points).all():
                    raise ValueError(f'Expected 52 finite cached points: {au}/{intensity}')
                recognized[au][str(intensity)] = points.tolist()
        print(f'Loaded recognition cache (no model inference): {cache_path}', flush=True)
        return recognized

    model_dir = Path(model_dir).expanduser().resolve()
    pipeline = load_pipeline()
    localizer = pipeline.TFLiteRunner(model_dir / 'cat_face_localizer.tflite')
    landmarker = pipeline.TFLiteRunner(model_dir / 'cat_face_landmarks_full.tflite')
    recognized = {}
    for au, images in groups.items():
        recognized[au] = {}
        for intensity, path in images:
            frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f'Cannot decode image: {path}')
            try:
                bbox, _, predicted = predict(frame, localizer, landmarker, pipeline, crop_margin)
                points = pack_points(predicted, bbox)
            except (ValueError, RuntimeError) as error:
                raise ValueError(f'{path}: {error}') from error
            recognized[au][str(intensity)] = points.tolist()
        print(f'{au}: {len(images)} images recognized', flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # 所有图片识别成功才写入，并原子替换，避免中断留下不完整 JSON。
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=cache_path.parent,
                                     prefix=cache_path.name+'.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(recognized, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(cache_path)
    print(f'Saved recognition cache: {cache_path}', flush=True)
    return recognized


def plot_estimator(name, results, output):
    """一幅图固定一个 AU 估计量，各曲线对应不同实际 AU 图片分组。"""
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.get_cmap('tab20').colors
    markers = ('o', 's', '^', 'v', 'D', 'P', 'X', '<', '>', '*', 'h')
    for index, (au, (intensities, metrics, points)) in enumerate(results.items()):
        ax.plot(intensities, metrics[name], label=au,
                color=colors[index % len(colors)], marker=markers[index % len(markers)],
                markersize=3.5, linewidth=1.6, linestyle='--' if index % 2 else '-')
    ax.axhline(0, color='grey', linewidth=0.6, zorder=0)
    ax.set_title(f'{name}: responses across actual AU groups\n'
                 'Shared reference: zeroface.json; no intensity mapping')
    ax.set_xlabel('Actual AU activation intensity (from image filename)')
    unit = 'radians' if 'EAD104' in name else 'ratio'
    ax.set_ylabel(f'{name} ({unit})')
    ax.grid(alpha=0.25)
    ax.legend(title='Actual AU image group', loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(output / f'{name}.png', dpi=180)
    plt.close(fig)


def analyze(source, output, model_dir=DEFAULT_MODEL_DIR, crop_margin=0.10,
            ear_pitch_sign=None, reference=DEFAULT_REFERENCE, overwrite=True, cache=None):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if not 0 <= crop_margin <= 1:
        raise ValueError('crop_margin must be between 0 and 1')
    groups = read_groups(source)
    reference, reference_config, zeroface_points = load_reference(reference)
    # Protect earlier analysis artifacts while allowing an existing empty directory.
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f'Output directory is not empty: {output}')
    model_dir = Path(model_dir).expanduser().resolve()
    cache_path = (source.parent / (source.name + '_landmarks.json')
                  if cache is None else Path(cache).expanduser().resolve())
    if cache_path == reference:
        raise ValueError('Recognition cache and neutral reference must use different paths')
    recognized = recognize_images(groups, cache_path, model_dir, crop_margin)
    signs = {'left': 1, 'right': 1} if ear_pitch_sign is None else ear_pitch_sign
    results, rows, summaries = {}, [], {}
    for au, images in groups.items():
        points = np.asarray([recognized[au][str(intensity)] for intensity, path in images],
                            dtype=np.float64)
        metrics = extract_motion(points, zeroface_points, signs)
        if not np.isfinite(np.stack(list(metrics.values()))).all():
            raise ValueError(f'{au}: non-finite AU estimates')
        intensities = np.array([item[0] for item in images])
        results[au] = (intensities, metrics, points)
        for i, (intensity, path) in enumerate(images):
            rows.append({'au': au, 'intensity': intensity, 'image': str(path),
                         **{name: float(metrics[name][i]) for name in METRICS}})
        summaries[au] = {
            'count': len(images),
            'estimate_range': {name: float(np.ptp(metrics[name])) for name in METRICS},
        }
        print(f'{au}: {len(images)} images processed', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for name in METRICS:
        plot_estimator(name, results, output)
    with (output / 'au_estimates.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['au', 'intensity', 'image', *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    # Store source landmark predictions as well as values, so formulas can be revisited.
    np.savez_compressed(output / 'landmarks.npz',
                        **{au: result[2] for au, result in results.items()})
    metadata = {
        'source': str(source), 'metrics': list(METRICS),
        'units': {name: 'radians' if 'EAD104' in name else 'ratio' for name in METRICS},
        'recognition_cache': str(cache_path),
        'plot_layout': 'One AU estimator per figure; one actual AU folder per curve',
        'reference_config': str(reference),
        'reference_image': reference_config.get('source'),
        'zeroface_points': zeroface_points.tolist(),
        'model_dir': str(model_dir), 'crop_margin': crop_margin, 'ear_pitch_sign': signs,
        'landmark_order': 'Rows follow ascending filename intensity; each image has 52 pixel x,y points: landmarks 0-47, bbox TL/TR/BR/BL 48-51',
        'formula_notes': [
            'Each AU function independently compares current and neutral 52-point sets; baseline is not read or computed by the analysis pipeline.',
            'Original formulas are retained. All geometric preprocessing, differences and half-wave operations are repeated within each AU function.',
            'Ear pitch is a signed 2D height proxy, not an identifiable 3D rotation.',
            'AU118_estimate currently measures vertical lip descent, not directly protrusion.',
            'Left/right are anatomical sides; right is image-left.',
            'Face-box corners are passed to every estimator and saved, but the current formulas use facial landmarks only.',
            'estimate_range is max minus min for reporting only; never used for normalization.',
        ],
        'groups': summaries,
    }
    (output / 'analysis.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Saved {len(METRICS)} plots and {len(rows)} measurements to {output}', flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=SIM_DIRECTORY / 'simcat_video_extra')
    parser.add_argument('--output-dir', type=Path, default=SIM_DIRECTORY / 'landmark2au_analysis')
    parser.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--cache', type=Path, help='Recognition JSON; default: INPUT sibling INPUT_NAME_landmarks.json')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True,
                        help='Replace derived plots and tables (default: enabled; cache is reused)')
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--crop-margin', type=float, default=0.10)
    parser.add_argument('--left-ear-pitch-sign', type=int, choices=(-1, 1), default=1)
    parser.add_argument('--right-ear-pitch-sign', type=int, choices=(-1, 1), default=1)
    args = parser.parse_args(argv)
    try:
        analyze(args.input, args.output_dir, args.model_dir, args.crop_margin,
                {'left': args.left_ear_pitch_sign, 'right': args.right_ear_pitch_sign},
                args.reference, args.overwrite, args.cache)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
