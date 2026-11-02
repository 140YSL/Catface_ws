from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

NPY_PATH = (Path(__file__).resolve().parent / "../au_player/config/au2joint.npy").resolve()
SMOOTH_NPY_PATH = (Path(__file__).resolve().parent / "../au_player/config/au2joint_smooth.npy").resolve()
SMOOTH_META_PATH = (Path(__file__).resolve().parent / "../au_player/config/au2joint_smooth.npz").resolve()

SMOOTH_METHOD = "chaikin"  # 可选: "pchip" / "chaikin" / "gaussian" / "constrained"

AU_NAMES = [
    "LAU143",
    "RAU143",
    "AU109+110",
    "AU118",
    "AU26",
    "LEAD101",
    "LEAD104",
    "LEAD106",
    "READ101",
    "READ104",
    "READ106",
]

AU_DESCRIPTIONS = [
    "左眼闭眼",
    "右眼闭眼",
    "皱鼻+上唇提升",
    "唇部前撅",
    "下颌下垂",
    "左耳向前",
    "左耳旋转",
    "左耳向后",
    "右耳向前",
    "右耳旋转",
    "右耳向后",
]

JOINT_NAMES = [
    "neck_pitch",
    "neck_yaw",
    "Head",
    "Jaw",
    "left_upper_palate",
    "right_upper_palate",
    "left_eye_bottom_skin",
    "left_eye_upper_skin",
    "right_eye_bottom_skin",
    "right_eye_upper_skin",
    "left_ear_rotate",
    "left_ear",
    "right_ear_rotate",
    "right_ear",
]

INTENSITY_LEVELS = np.array([0, 1, 2, 3, 4, 5], dtype=float)

SMOOTH_SAMPLE_COUNT = 1001


def smooth_pchip(timestamps, values, smooth_x):
    return PchipInterpolator(timestamps, values)(smooth_x)


def smooth_chaikin(timestamps, values, smooth_x, iterations=4):
    xs = np.asarray(timestamps, dtype=float)
    ys = np.asarray(values, dtype=float)
    for _ in range(iterations):
        new_xs = [xs[0]]
        new_ys = [ys[0]]
        for i in range(len(xs) - 1):
            new_xs.extend(
                [0.75 * xs[i] + 0.25 * xs[i + 1], 0.25 * xs[i] + 0.75 * xs[i + 1]]
            )
            new_ys.extend(
                [0.75 * ys[i] + 0.25 * ys[i + 1], 0.25 * ys[i] + 0.75 * ys[i + 1]]
            )
        new_xs.append(xs[-1])
        new_ys.append(ys[-1])
        xs, ys = np.array(new_xs), np.array(new_ys)
    return np.interp(smooth_x, xs, ys)


def smooth_gaussian(timestamps, values, smooth_x, sigma=1.5):
    grid = np.linspace(timestamps.min(), timestamps.max(), 1000)
    dense_y = np.interp(grid, timestamps, values)
    smoothed = gaussian_filter1d(dense_y, sigma=sigma, mode="nearest")
    return np.interp(smooth_x, grid, smoothed)


def smooth_constrained(timestamps, values, smooth_x, smooth_weight=10.0):
    grid = np.linspace(timestamps.min(), timestamps.max(), 200)
    init = np.interp(grid, timestamps, values)
    seg = np.clip(np.searchsorted(timestamps, grid) - 1, 0, len(timestamps) - 2)
    seg_a = np.asarray(values, dtype=float)[seg]
    seg_b = np.asarray(values, dtype=float)[seg + 1]
    lower = np.minimum(seg_a, seg_b) - 0.4
    upper = np.maximum(seg_a, seg_b) + 0.4

    def objective(y):
        fidelity = np.sum((y - init) ** 2)
        curvature = np.sum((y[2:] - 2 * y[1:-1] + y[:-2]) ** 2)
        return fidelity + smooth_weight * curvature

    result = minimize(
        objective,
        init,
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": 500},
    )
    return np.interp(smooth_x, grid, result.x)


SMOOTH_FUNCS = {
    "pchip": smooth_pchip,
    "chaikin": smooth_chaikin,
    "gaussian": smooth_gaussian,
    "constrained": smooth_constrained,
}


def load_au2joint(npy_path):
    data = np.load(npy_path, allow_pickle=False)
    if data.shape != (len(AU_NAMES), 5, len(JOINT_NAMES)):
        raise ValueError(f"unexpected au2joint shape: {data.shape}")
    # AU 强度为 0 时所有关节目标角度均为 0，补充该初始采样点
    zeros = np.zeros((data.shape[0], 1, data.shape[2]), dtype=data.dtype)
    return np.concatenate([zeros, data], axis=1)


def smooth_au2joint(data, smooth_x):
    smooth_func = SMOOTH_FUNCS[SMOOTH_METHOD]
    smoothed = np.zeros(
        (data.shape[0], len(smooth_x), data.shape[2]), dtype=np.float64
    )
    for au_idx in range(data.shape[0]):
        for joint_idx in range(data.shape[2]):
            values = data[au_idx, :, joint_idx]
            if not np.any(values):
                continue
            smoothed[au_idx, :, joint_idx] = smooth_func(
                INTENSITY_LEVELS, values, smooth_x
            )
    return smoothed


def save_au2joint_smooth(smoothed, smooth_x):
    np.save(SMOOTH_NPY_PATH, smoothed)
    # 纯 .npy 数组无法携带 key，额外保存名称元数据
    np.savez(
        SMOOTH_META_PATH,
        au_names=np.array(AU_NAMES),
        joint_names=np.array(JOINT_NAMES),
        intensities=smooth_x,
    )
    print(f"saved: {SMOOTH_NPY_PATH} shape={smoothed.shape}")
    print(f"saved: {SMOOTH_META_PATH}")


def plot_au2joint(data, smoothed, smooth_x):
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 11), sharex=True)
    axes = axes.ravel()

    for au_idx, (au_name, au_description) in enumerate(zip(AU_NAMES, AU_DESCRIPTIONS)):
        ax = axes[au_idx]
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            values = data[au_idx, :, joint_idx]
            if not np.any(values):
                continue
            ax.scatter(INTENSITY_LEVELS, values, s=15, alpha=0.6)
            ax.plot(smooth_x, smoothed[au_idx, :, joint_idx], label=joint_name)
        ax.set_xticks(INTENSITY_LEVELS)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{au_name} ({au_description})", fontsize=10)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
        if au_idx // ncols == nrows - 1:
            ax.set_xlabel("AU Intensity")
        if au_idx % ncols == 0:
            ax.set_ylabel("Joint Motion")

    for ax in axes[len(AU_NAMES):]:
        ax.axis("off")

    fig.suptitle(f"AU2Joint Map - {NPY_PATH.name} ({SMOOTH_METHOD})")
    fig.tight_layout()
    return fig


def main():
    data = load_au2joint(NPY_PATH)
    print("data shape:", data.shape)
    smooth_x = np.linspace(INTENSITY_LEVELS.min(), INTENSITY_LEVELS.max(), SMOOTH_SAMPLE_COUNT)
    smoothed = smooth_au2joint(data, smooth_x)
    save_au2joint_smooth(smoothed, smooth_x)
    plot_au2joint(data, smoothed, smooth_x)
    plt.show()


if __name__ == "__main__":
    main()
