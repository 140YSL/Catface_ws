# 本tool文件夹用于存放在项目过程中写的工具脚本，包括人工标注界面等等

## realcat_landmark_generate.py

对指定视频每 N 帧预测一次猫脸 48 点，其余帧沿用最近一次预测坐标。
输出 MP4 保留全部可解码帧、原分辨率和原帧率（变帧率输入按标称帧率输出），不保留音频。
默认显示 0–47 编号；预测失败时清除旧点，直到下一次成功预测。适合单只猫的视频。

使用 Python 3.10/3.11，在项目根目录准备环境及模型：

```bash
python3.11 -m venv src/cat-landmark-portable/.venv
src/cat-landmark-portable/.venv/bin/python -m pip install -r src/cat-landmark-portable/requirements-ml.txt 'opencv-python<4.12'
src/cat-landmark-portable/.venv/bin/python src/cat-landmark-portable/tools/download_models.py --kind inference
src/cat-landmark-portable/.venv/bin/python src/tool/realcat_landmark_generate.py --input /path/to/cat.mp4 --output /path/to/cat_landmarks.mp4 --stride 5
```

`--stride 1` 为逐帧推理；`--hide-labels` 隐藏编号；`--show-boxes` 显示脸框与裁剪框。
`--landmark-model /path/to/robot_landmarks.tflite` 可指定微调后的模型。
输出路径必须是尚不存在的 `.mp4` 文件，宽高须为偶数；编码采用 `mp4v`。
中途异常可能留下部分输出，重试时使用新路径。输出为重新编码的视频。

## au_tagger

用于对真实猫视频进行标注其au强度的文件

```bash
cd au_tagger
python3 main.py
```

## au_visualization.py

用于对人工标注得到的曲线进行插值，人工得到的曲线是步长间隔0.2的离散点，经过插值得到120Hz的连续点
并可视化出来

## change_urdf2xml.py

将sw导出的urdf文件自动转化为xml文件

## compute_delay.py

根据记录的blender-joint-state和mujoco-joint-state以及mujoco_ctrl计算得到仿真环境的延迟时间

## import_urdf2blender.py

用于将urdf导入blender中

## joint2au_map_viszualization.py

用于将人工标注的1-5强度的离散的au-joint的对应平滑插值得到连续的对应
并可视化出来

## measure_distance.py

启动后点击屏幕上的两点用于测量这两个点的像素距离，用于作为au强度评判标准的辅助工具
