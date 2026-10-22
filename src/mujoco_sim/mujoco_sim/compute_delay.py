# -*- coding: utf-8 -*-
"""
曲线绘图 + 时间延迟分析脚本
1) 读取 PlotJuggler 导出的时序 CSV, 绘制整个时间范围的曲线并弹窗展示;
2) 在指定时间窗内做延迟计算: 取命令曲线的每个值 v,
   反插值响应曲线到达同一 v 的时刻, 时间差取平均即延迟;
3) 图上用红色竖线标出计算窗口的起止, 并标出参与匹配的响应点.

所有配置在文件开头, 改参数只需编辑 CONFIG 部分.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================ 配置区 (改这里) ============================
CSV_PATH   = "../../../Worklog/1.csv"          # CSV 文件路径

# 要绘制的曲线: (列名关键词, 图例显示名, 颜色)
CURVES = [
    ("/mujoco_ctrl/data[2]",                 "cmd (mujoco_ctrl/data[2])",     "C1"),
    ("/joint_state_blender/Head/position",   "blender (joint_state_blender)", "C0"),
    ("/joint_states/Head/position",          "mujoco (joint_states)",         "C2"),
]

# --- 延迟计算配置 ---
# CMD_KEY       = "/mujoco_ctrl/data[2]"               # 命令曲线
CMD_KEY       = "/joint_states/Head/position"               # Mujoco响应曲线
RESP_KEY      = "/joint_state_blender/Head/position"  # blender响应曲线
CALC_WINDOW   = (5.0, 6.5)      # 计算窗口 (秒, 相对时间), 窗口内响应需近似单调
ENDPOINT_TRIM = 0.05            # 端点余量比例 (命令值超出响应范围 5% 的不参与)

TIME_COL_HINT = "time"      # 时间列名关键词 (第一列优先)
FIG_PATH      = None        # 图片保存路径, None 则自动: <csv名>_curves.png
EPOCH_SHIFT   = True        # 时间戳为 epoch 大数时自动平移到 0 起点
SHOW_WINDOW   = True        # 弹出窗口展示图形并保持打开 (阻塞直到手动关闭)
ALSO_SAVE_PNG = False       # 展示的同时是否保存 PNG
# ======================================================================

plt.rcParams["axes.unicode_minus"] = False


def find_col(df, key):
    key = key.lower().replace(" ", "")
    for c in df.columns:
        if key in str(c).lower().replace(" ", ""):
            return c
    raise KeyError(f"找不到包含 '{key}' 的列, 现有列: {list(df.columns)}")


def interp_on_grid(t_raw, y_raw, grid):
    """把 (t_raw, y_raw) 在自身有效范围内线性插值到 grid 上, 之外置 NaN."""
    m = ~np.isnan(y_raw)
    t_v, y_v = t_raw[m], y_raw[m]
    if len(t_v) < 2:
        raise ValueError("某条曲线有效点不足, 无法插值")
    out = np.full(grid.shape, np.nan)
    inside = (grid >= t_v[0]) & (grid <= t_v[-1])
    out[inside] = np.interp(grid[inside], t_v, y_v)
    return out


def time_at_value(t, y, value):
    """反插值: y(t) 到达 value 的时刻 (窗口内近似单调)."""
    idx = np.argsort(y)
    return float(np.interp(value, y[idx], t[idx]))


def compute_delay(t_all, cmd_raw, resp_raw, t0, t1):
    """返回 delays(数组) 和匹配明细 [(t_cmd, v, t_resp, delay_ms), ...]"""
    grid = np.arange(t_all[0], t_all[-1], 0.001)
    cmd_i  = interp_on_grid(t_all, cmd_raw,  grid)
    resp_i = interp_on_grid(t_all, resp_raw, grid)

    m = (grid >= t0) & (grid <= t1) & ~np.isnan(cmd_i) & ~np.isnan(resp_i)
    t_w, cmd_w, resp_w = grid[m], cmd_i[m], resp_i[m]
    if len(t_w) < 10:
        raise ValueError(f"窗口 [{t0}, {t1}] 内有效采样点太少 ({len(t_w)} 个)")

    d = np.diff(resp_w)
    mono = max(np.mean(d >= -1e-12), np.mean(d <= 1e-12))
    if mono < 0.95:
        print(f"[警告] 响应曲线在窗口内非单调 (一致率 {mono:.0%}), "
              f"结果可能失真, 建议收窄窗口到单一上升/下降沿!")

    delays, rows = [], []
    lo, hi = resp_w.min(), resp_w.max()
    span = hi - lo
    for tc, v in zip(t_w, cmd_w):
        if v < lo + ENDPOINT_TRIM * span or v > hi - ENDPOINT_TRIM * span:
            continue
        tr = time_at_value(t_w, resp_w, v)
        delays.append(tr - tc)
        rows.append((tc, v, tr, (tr - tc) * 1000))
    return np.array(delays), rows


def main():
    df = pd.read_csv(CSV_PATH)

    # --- 时间列 ---
    if TIME_COL_HINT.lower() in str(df.columns[0]).lower():
        t_col = df.columns[0]
    else:
        t_col = find_col(df, TIME_COL_HINT)
    t = pd.to_numeric(df[t_col], errors="coerce").to_numpy()
    if EPOCH_SHIFT and np.nanmedian(t) > 1e8:
        t = t - t[0]
        print("[提示] 检测到绝对时间戳, 已平移到 0 起点")

    # --- 延迟计算 ---
    t0, t1 = CALC_WINDOW
    delays, rows = [], []
    try:
        cmd_raw  = pd.to_numeric(df[find_col(df, CMD_KEY)],  errors="coerce").to_numpy()
        resp_raw = pd.to_numeric(df[find_col(df, RESP_KEY)], errors="coerce").to_numpy()
        delays, rows = compute_delay(t, cmd_raw, resp_raw, t0, t1)
        print(f"\n=== 延迟计算: 窗口 [{t0}, {t1}] s ===")
        print(f"命令: {CMD_KEY}")
        print(f"响应: {RESP_KEY}")
        print(f"参与匹配: {len(delays)} 点")
        print(f"平均延迟 : {np.mean(delays)*1000:8.2f} ms")
        print(f"中位延迟 : {np.median(delays)*1000:8.2f} ms")
        print(f"标准差   : {np.std(delays)*1000:8.2f} ms")
        print(f"最小/最大: {np.min(delays)*1000:.2f} / {np.max(delays)*1000:.2f} ms")
        if ALSO_SAVE_PNG:
            pd.DataFrame(rows, columns=["t_cmd", "cmd_value", "t_resp", "delay_ms"]).to_csv(
                CSV_PATH.rsplit(".", 1)[0] + "_delay_detail.csv", index=False)
    except (KeyError, ValueError) as e:
        print(f"[提示] 延迟计算跳过: {e}")

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = 0
    for key, label, color in CURVES:
        try:
            col = find_col(df, key)
        except KeyError as e:
            print(f"[跳过] {e}")
            continue
        y = pd.to_numeric(df[col], errors="coerce").to_numpy()
        m = ~np.isnan(y)
        if m.sum() < 2:
            print(f"[跳过] '{key}' 有效点不足")
            continue
        ax.plot(t[m], y[m], color=color, linewidth=1.2,
                label=f"{label} ({m.sum()} pts)")
        plotted += 1

    if plotted == 0:
        raise SystemExit("没有任何曲线可绘制, 检查 CURVES 配置")

    # # --- 红色窗口竖线 ---
    # for x, tag in [(t0, "start"), (t1, "end")]:
    #     ax.axvline(x, color="red", linestyle="--", linewidth=1.5)
    #     ax.text(x, ax.get_ylim()[1] if plotted else 0, f" {tag}",
    #             color="red", va="top", fontsize=10)

    # # --- 匹配点散点 ---
    # if rows:
    #     ax.scatter([r[2] for r in rows], [r[1] for r in rows],
    #                s=12, color="red", zorder=5,
    #                label=f"matched points (delay avg {np.mean(delays)*1000:.1f} ms)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    title = f"Head joint curves - {CSV_PATH}"
    if len(delays):
        title += f"  |  delay [{t0}-{t1}s] = {np.mean(delays)*1000:.1f} ms"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()

    if ALSO_SAVE_PNG or not SHOW_WINDOW:
        out = FIG_PATH or (CSV_PATH.rsplit(".", 1)[0] + "_curves.png")
        fig.savefig(out, dpi=150)
        print(f"已保存: {out}")
    if SHOW_WINDOW:
        print("图形窗口已打开, 关闭窗口后程序退出...")
        plt.show()


if __name__ == "__main__":
    main()
