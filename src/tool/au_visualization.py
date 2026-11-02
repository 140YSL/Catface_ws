from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

NPZ_PATH = (Path(__file__).resolve().parent / "../../cat_video/3.npz").resolve()

SMOOTH_METHOD = "constrained"  # 可选: "pchip" / "chaikin" / "gaussian" / "constrained"

AU_GROUPS = [
    ["LAU143", "RAU143"],
    ["AU109+110", "AU118", "AU26"],
    ["LEAD101", "LEAD104", "LEAD106", "READ101", "READ104", "READ106"],
]

AU_DESCRIPTIONS = [
    ["左眼闭眼", "右眼闭眼"],
    ["皱鼻+上唇提升", "唇部前撅", "下颌下垂"],
    ["左耳向前", "左耳旋转", "左耳向后", "右耳向前", "右耳旋转", "右耳向后"],   
]




def load_au_data(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        timestamps = np.asarray(data["time"], dtype=np.float64)
        au_values = {
            au_name: np.asarray(data[au_name]) for au_name in sum(AU_GROUPS, [])
        }
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    au_values = {name: values[order] for name, values in au_values.items()}
    return timestamps, au_values


def smooth_pchip(timestamps, values, smooth_x):
    smooth_y = PchipInterpolator(timestamps, values)(smooth_x)
    return np.clip(smooth_y, 0, 5)


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
    return np.clip(np.interp(smooth_x, xs, ys), 0, 5)


def smooth_gaussian(timestamps, values, smooth_x, sigma=1.5):
    grid = np.linspace(timestamps.min(), timestamps.max(), 1000)
    dense_y = np.interp(grid, timestamps, values)
    smoothed = gaussian_filter1d(dense_y, sigma=sigma, mode="nearest")
    return np.clip(np.interp(smooth_x, grid, smoothed), 0, 5)


def smooth_constrained(timestamps, values, smooth_x, smooth_weight=10.0):
    grid = np.linspace(timestamps.min(), timestamps.max(), 200)
    init = np.interp(grid, timestamps, values)
    seg = np.clip(np.searchsorted(timestamps, grid) - 1, 0, len(timestamps) - 2)
    seg_a = np.asarray(values, dtype=float)[seg]
    seg_b = np.asarray(values, dtype=float)[seg + 1]
    lower = np.clip(np.minimum(seg_a, seg_b) - 0.4, 0, 5)
    upper = np.clip(np.maximum(seg_a, seg_b) + 0.4, 0, 5)

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
    return np.clip(np.interp(smooth_x, grid, result.x), 0, 5)


SMOOTH_FUNCS = {
    "pchip": smooth_pchip,
    "chaikin": smooth_chaikin,
    "gaussian": smooth_gaussian,
    "constrained": smooth_constrained,
}


def plot_groups(timestamps, au_values):
    fig, axes = plt.subplots(
        len(AU_GROUPS), 1, figsize=(12, 4 * len(AU_GROUPS)), sharex=True
    )

    for ax, group, description in zip(axes, AU_GROUPS, AU_DESCRIPTIONS):
        for au_name, au_description in zip(group, description):
            values = au_values[au_name]
            ax.scatter(timestamps, values, s=12, alpha=0.6)
            if len(timestamps) >= 4:
                smooth_x = np.linspace(timestamps.min(), timestamps.max(), 500)
                smooth_y = SMOOTH_FUNCS[SMOOTH_METHOD](timestamps, values, smooth_x)
                ax.plot(smooth_x, smooth_y, label=f"{au_description} ")
        ax.set_ylabel("AU Intensity")
        ax.set_ylim(-0.5, 5.5)
        ax.set_yticks(range(6))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax.set_title(" / ".join(group))

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"AU Intensities - {NPZ_PATH.name}")
    fig.tight_layout()
    return fig


def main():
    timestamps, au_values = load_au_data(NPZ_PATH)
    plot_groups(timestamps, au_values)
    plt.show()


if __name__ == "__main__":
    main()
