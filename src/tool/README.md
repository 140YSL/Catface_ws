# 本tool文件夹用于存放在项目过程中写的工具脚本，包括人工标注界面等等

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