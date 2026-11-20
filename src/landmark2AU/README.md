# Landmark 到 AU 实验

两个脚本直接使用 catlandmark Python 环境运行，无需启动 ROS。
以下命令工作目录为工作空间根目录。

```bash
conda activate catlandmark
python src/landmark2AU/landmark2AU/realcat_landmark_generate.py --input cat_video/organized/afraid/1.mp4 --output cat_video/temp/1_landmarks_v2.mp4 --stride 1
python src/landmark2AU/landmark2AU/landmark2au.py --input cat_video/temp/1_landmarks_v2.npy
```

视频脚本保留原有默认输入输出，不覆盖已经存在的视频或同名 NPY。
NPY 为一维结构化数组，每帧一行；列名是 `data.dtype.names`，无需 pickle。
视频与 NPY 均在推理成功结束后交付；中断时可能留下部分视频，没有完整 NPY。

```python
import numpy as np
data = np.load('cat_video/temp/1_landmarks_v2.npy', allow_pickle=False)
print(data.dtype.names)
print(data['timestamp_s'])
print(data['bbox_x1_px'])
print(data['landmark_00_x_px'])
```

- `timestamp_s`：输出恒定帧率时间轴，即 frame_index / fps。
- `source_timestamp_s`：OpenCV 返回的源视频相对时间；无效或不递增时为 NaN。
- `source_timestamp_valid`：源时间有效标记；变帧率原始时间戳的精确恢复仍依赖解码器。
- `fps`, `image_width`, `image_height`：视频属性。
- `bbox_{x1,y1,x2,y2}_px`：模型脸框；`crop_*_px`：外扩后实际裁剪框。
- `landmark_00_x_px` 到 `landmark_47_y_px`：原图像素坐标，原点左上，x 向右，y 向下。
- `valid`：坐标计算有效，不代表点位经过人工审核。
- `prediction_attempted`, `prediction_frame`：是否本帧推理、坐标实际来源帧；步长大于 1 时中间帧复用点与框。
- 无效预测的坐标为 NaN，来源帧为 -1。

## AU 输出与标定

`landmark2au.py --input` 支持单个 NPY 或文件夹（不递归）。默认输出至输入旁的
`au/文件名_au.npy` 和同名 JSON，不覆盖已有结果。输入目录应只包含 landmark NPY。

输出包含帧号、时间、有效性、来源帧，以及 11 个 `_norm` 几何量和 11 个 `AU_*` 强度。
左右沿猫的解剖命名：right 为图左侧，left 为图右侧。这些是项目动作名称，不是标准 CatFACS AU 编码。

| 指标 | 计算定义 |
|---|---|
| left/right_eye_close | 眼睑局部垂直间距 / 眼角宽度，随后相对参考睁眼量计算闭合比例 |
| left/right_ear_outward | 耳根中点到耳尖相对面部竖轴的向外角度减参考值，单位弧度 |
| left/right_ear_forward | 耳朵投影高度 / 耳根宽度相对参考的有符号缩短量正半轴 |
| left/right_ear_backward | 上述有符号量负半轴的绝对值 |
| upper_lip_up/down | 上唇点 17 相对鼻参考 12、13 的纵向位移负/正半轴，除以眼中心间距 |
| jaw_down | 下巴点 2 相对鼻参考的纵向位移，除以眼中心间距 |

嘴部和耳朵方向先以眼角中心连线建立坐标系，去除平移、均匀缩放和画面内 roll。
眼睛使用各自眼角建立局部轴。不使用瞳孔点作为眼睑，不使用脸框宽度作为分母。
上下眼睑合计为每只眼一个闭合指标，共两个，不分别恢复上下眼睑独立动作。

**耳朵 forward/backward 仅是初始代理定义。投影缩短无法唯一判断真实前倾或后仰；
两个输出是一个有符号量的正负部分，不是独立恢复的三维自由度。**
`ear_pitch_sign` 可以改为 -1，但翻转符号不能解决三维歧义；需要已知动作样本验证，必要时多特征回归。
所有指标仍受 yaw、pitch、遮挡及预测误差影响，没有自动视角补偿或遮挡置信度。

默认使用前 1 秒有效帧的中位数作为参考，**不宣称它是中性脸**。
通过 `--neutral-start 2 --neutral-end 3` 更换参考时段，必须自行检查该时段动作。
默认线性关系为 `clip(5*(量值-zero)/(full-zero), 0, 5)`：闭眼满幅 1，耳外旋满幅 pi/4，
其余满幅 0.25，零点均为 0。这些只是可运行的起始参数，不是已验证的生理标定。
不按每段视频极值自动拉满到 5。

首次输出 JSON 保存实际参考、映射和警告；修改其 `baseline`（可以分别按器官指定）、
`mapping`、`ear_pitch_sign` 后，通过 `--config 文件.json` 重用固定标定。
`mapping` 的每个值为 `[对应AU0的量值, 对应AU5的量值]`，支持递减映射。
无效帧仍保留，量值和 AU 为 NaN。先核对量值和参考，再将 AU 用于播放。
