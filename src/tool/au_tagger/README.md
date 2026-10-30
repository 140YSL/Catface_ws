# 视频 AU 人工标注工具

## 运行

安装依赖后，从工作区根目录运行：

```bash
python3 -m pip install -r src/tool/au_tagger/requirements.txt
python3 src/tool/au_tagger/main.py
```

点击“打开文件夹”选择目录，在左侧文件树中双击视频。支持常见 MP4、AVI、MOV、MKV、WebM 等格式，实际解码能力取决于本机 OpenCV/FFmpeg。

## 操作

- “播放/暂停”按钮或 `Space`：控制播放。
- 进度条：拖动到指定帧，拖动时暂停播放。
- `Left`：后退一帧；`Right`：前进一帧，逐帧操作时自动暂停。
- 11 个 AU 强度滑条按原顺序每行两个排列，均有 0、1、2、3、4、5 六档。
- “插帧”：保存当前帧的 11 项强度及时间戳；同一帧再次插帧会覆盖旧值。
- 跳转到已标注帧时自动恢复强度，未标注帧显示全 0。

## NPZ 格式

标注保存在视频同目录、同文件名的 `.npz` 文件中，例如 `demo.mp4` 对应 `demo.npz`。文件包含：

- `frame_indices`：从 0 开始的帧索引，按升序排列。
- `time`：对应帧的时间，单位为秒。
- `LAU143`、`RAU143`、`AU109+110`、`AU118`、`AU26`、`LEAD101`、`LEAD104`、`LEAD106`、`READ101`、`READ104`、`READ106`：各 AU 在所有标注帧上的强度，均为一维 `uint8` 数组。
- `au_names`：11 个 AU 的键名。
- `au_descriptions`：11 个 AU 的中文名称。

示例读取：

```python
import numpy as np

with np.load("demo.npz", allow_pickle=False) as data:
    print(data["frame_indices"])
    print(data["time"])
    print(data["LAU143"])
    print(data["READ106"])
```

本工具使用 OpenCV 渲染画面，不包含音频播放。
