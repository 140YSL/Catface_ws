# 启动说明

```bash
git clone https://github.com/140YSL/Catface_ws.git

cd Catface_ws

colcon build 

ros2 launch mujoco_sim catface_mujoco_blender.launch.py

```

# 控制各个关节

```bash
""" 控制话题消息/mujoco_ctrl,长度14位，
    0: neck_pitch
    1: neck_yaw
    2: Head
    3: Jaw
    4: left_upper_palate
    5: right_upper_palate
    6: left_eye_bottom_skin
    7: left_eye_upper_skin
    8: right_eye_bottom_skin
    9: right_eye_upper_skin
    10: left_ear_rotate
    11: left_ear
    12: right_ear_rotate
    13: right_ear
"""

ros2 topic info /mujoco_ctrl
Type: std_msgs/msg/Float64MultiArray

```