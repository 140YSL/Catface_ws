# Landmark AU 估计量实验

## AU 估计公式验证：直接比较两份点集

```bash
python src/landmark2AU/landmark2AU/landmark2au_analysis.py
```

默认读取 `cat_video/sim/simcat_video_extra/<AU>/<AU>-<强度>.png`，统一使用
`cat_video/sim/zeroface.json` 中 `reference_image.landmarks_px` 和
`reference_image.bbox_xyxy_px` 构建中性脸点集。不读取 JSON 中的 `baseline`，
也不从各分组的强度 0 图片生成参考值。`--reference` 可指定另一份参考文件。

每份点集均为原图像素坐标 `(52, 2)`：

- 0–47：48 个面部关键点。
- 48–51：脸框左上、右上、右下、左下角。

`AU26_estimate(points, zeroface_points)`、`AU109_110_estimate(...)`、
`AU118_estimate(...)` 接收两份完整点集，直接返回最终标量估计值。
`AU143_estimate`、`EAD101_estimate`、`EAD104_estimate`、`EAD106_estimate`
另外接收 `side`（`left` 或 `right`），两个耳前后函数还接收 `pitch_sign`。
各函数独立执行当前点及中性点的坐标处理、特征计算、参考差值和正负半轴处理，
不缓存或共用中间几何特征。脸框角点传入各函数；当前公式仍只使用面部关键点，
未引入脸框归一化。此阶段保留重复计算，以便逐个阅读和验证公式。

`extract_motion(points, zeroface_points, ...)` 接收 `(N, 52, 2)` 当前点集和
`(52, 2)` 中性点集，逐张、逐 AU 调用上述函数，汇总 11 列结果。输出不映射到
0–5，不按曲线极值或范围缩放。Python 标识符不能包含连字符，因此函数名使用
`AU109_110_estimate`，结果列名为 `AU109-110_estimate`。

识别结果默认缓存到 `cat_video/sim/simcat_video_extra_landmarks.json`，
可通过 `--cache /path/results.json` 指定其他路径。缓存存在时直接读取，不加载模型；
不存在时识别全部图片，成功后保存 JSON。结构为 `{AU名称: {强度字符串: 52个[x,y]}}`，
JSON 对象键使用字符串，如 `"0.0"`、`"0.2"`。缓存校验点数、有限值及所需强度是否齐全；
无效/缺失条目会明确报错。更换图片、模型或裁剪参数后需指定新的缓存路径重新识别，
修改估计公式不需要重复识别。

结果保存至 `cat_video/sim/landmark2au_analysis`：

- `<AU名称_estimate>.png`：每幅图固定一个 AU 估计量；11 条曲线分别来自不同实际 AU 文件夹。横轴为各文件名标注强度，纵轴为该估计量，图例标注实际 AU 文件夹名称。
- `au_estimates.csv`：全部数值及图片路径。
- `landmarks.npz`：各组按文件名强度排序的 `(N, 52, 2)` 原图坐标，包含脸框角点。
- `analysis.json`：参考文件路径、中性脸 52 点、单位及各曲线的变化范围；不输出 baseline。

耳外转估计量单位为弧度，其余为无量纲比例。现有公式保留原定义；AU118 当前
仍使用上唇向下位移作为候选特征，耳前后仍使用二维投影高度代理。当前阶段用于
验证公式，不宣称已经恢复准确 AU 强度。左右按猫的解剖侧命名，right 为图左。

可用 `--input`、`--output-dir`、`--model-dir`、`--crop-margin` 更换输入输出及
推理配置；耳前后方向符号通过 `--left-ear-pitch-sign`、`--right-ear-pitch-sign`
指定（默认 +1）。默认重绘同名估计量图片、CSV 和分析元数据，并复用识别缓存；
`--no-overwrite` 可禁止覆盖非空输出目录。

## 中性脸生成脚本的兼容性

`calculate_zeroface.py` 保持原样。为使其现有调用继续工作，分析脚本保留旧接口
`extract_geometry(points)`，仍返回原有 8 项几何测量值，仅供该生成脚本使用。
它不调用新的 AU 估计函数，也不参与上述 AU 分析链路。生成脚本即使仍保存
`baseline`，分析时也只读取中性脸原始关键点和脸框。

## 运行环境与视频关键点

批量标注仿真图片：

```bash
/home/ubuntu/miniconda3/envs/cat_video_process/bin/python src/landmark2AU/landmark2AU/simcat_landmark_show.py
```

递归读取 `cat_video/sim/simcat_video_extra` 的全部图片，将红色脸框、黄色裁剪框及
编号 0–47 的关键点画在原图上，输出至 `cat_video/sim/simcat_video_extra_landmark`。
保留子目录、图片文件名和像素尺寸。可用 `--input`、`--output-dir`、`--model-dir`、
`--crop-margin` 覆盖默认设置；不覆盖已有目标图片。

需要包含 TensorFlow、OpenCV、NumPy、Pillow、Matplotlib 的 Python 3.10/3.11 环境，
无需 ROS。当前工作空间可使用：

```bash
/home/ubuntu/miniconda3/envs/cat_video_process/bin/python src/landmark2AU/landmark2AU/landmark2au_analysis.py
```

视频关键点提取器仍可独立运行：

```bash
python src/landmark2AU/landmark2AU/realcat_landmark_generate.py --input /path/video.mp4 --output /path/video_landmarks.mp4 --stride 1
```

输出标注视频及同名结构化 NPY，包含帧号、时间、有效性、预测来源帧、脸框、裁剪框
及 `landmark_00_x_px` 至 `landmark_47_y_px`。该步骤不计算 AU 强度。
