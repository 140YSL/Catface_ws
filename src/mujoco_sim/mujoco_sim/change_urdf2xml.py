import os
import re
import mujoco

urdf_path = ("/home/ysl/Desktop/CatFace/Catface_ws/src/"
             "robot_description/urdf/robotic_cathead_v2.urdf")
xml_out = ("/home/ysl/Desktop/CatFace/Catface_ws/src/"
           "robot_description/urdf/robotic_cathead_v2.xml")

# 1. 读取 URDF，把 package://包名/ 替换为相对路径 ../
#    例如 package://robot_description/meshes/v2/head.stl
#    ->  ../meshes/v2/head.stl
with open(urdf_path, "r", encoding="utf-8") as f:
    urdf_content = f.read()

print(urdf_content)

urdf_content_new, n = re.subn(
    r'package://[^/"\']+/',   # 匹配 package://包名/
    '../',                    # 替换为相对路径（相当于 urdf/ 的上一级）
    urdf_content
)
print(f"替换了 {n} 处 package:// 引用")

# 2. 写到 urdf 所在目录的临时文件（相对路径才能正确解析）
tmp_urdf = os.path.join(os.path.dirname(urdf_path), "_tmp_robotic_cathead_v2.urdf")
with open(tmp_urdf, "w", encoding="utf-8") as f:
    f.write(urdf_content_new)

# 3. 用 MuJoCo 加载并保存为 XML
model = mujoco.MjModel.from_xml_path(tmp_urdf)
mujoco.mj_saveLastXML(xml_out, model)
print(f"已保存: {xml_out}")

# 4. 清理临时文件
os.remove(tmp_urdf)