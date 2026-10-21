#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入 URDF 网格，并依据 URDF 的 link/joint 关系自动创建 Blender Armature。
适用于 Blender 4.0.2。

功能：
1. 解析 URDF links、joints、visual mesh、origin、axis。
2. 将 URDF 中的 STL 网格导入 Blender，并放置到零位姿态。
3. 为每个 URDF link 创建一根 Blender bone。
4. 根据 URDF joint 的 parent/child 关系建立骨骼层级。
5. 将每个 STL 网格作为对应 link 骨骼的刚性子对象，并保持初始世界变换。
6. 将 URDF 关节信息保存为骨骼自定义属性，供后续 ROS 2 / MuJoCo 映射使用。
7. 保存为 catcface.blend。

注意：
- 此脚本创建的是动画骨架，不是物理仿真器；物理仍由 MuJoCo 负责。
- 当前创建的是零位姿态（所有关节变量 q=0）。
- 请先确认 OUTPUT_BLEND 指向你希望保存的文件位置。
- 如果你已有想保留的手工编辑内容，请先备份 .blend 文件。
"""

import bpy
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from mathutils import Matrix, Vector, Euler


# ============================================================
# 路径配置：按当前 CatFace 工作空间设置
# ============================================================

URDF_PATH = Path(
    "/home/ysl/Desktop/CatFace/Catface_ws/src/"
    "robot_description/urdf/robotic_cathead_v2.urdf"
)

PACKAGE_ROOT = Path(
    "/home/ysl/Desktop/CatFace/Catface_ws/src/"
    "robot_description"
)

# 修改为你实际希望保存的 catcface.blend 路径
OUTPUT_BLEND = Path(
    "/home/ysl/Desktop/CatFace/Catface_ws/src/"
    "robot_description/blender/catcface_v2.blend"
)

CLEAR_SCENE = True

# 每根骨骼在视图中的显示长度（单位与 URDF 一致，通常为米）
BONE_DISPLAY_LENGTH = 0.05

# 导入 STL 时，是否将导入对象命名为 link__<link>__visual_<index>
USE_STANDARD_OBJECT_NAMES = True


# ============================================================
# URDF 解析工具
# ============================================================

def parse_xyz(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return Vector(default)
    values = [float(v) for v in text.split()]
    if len(values) != 3:
        raise ValueError(f"预期 xyz 为 3 个数值，实际为：{text}")
    return Vector(values)


def origin_to_matrix(origin_element):
    """将 URDF origin 的 xyz/rpy 转成 4x4 变换矩阵。

    URDF RPY 按 roll、pitch、yaw 给出；对应旋转组合为
    Rz(yaw) @ Ry(pitch) @ Rx(roll)。
    mathutils.Euler((roll, pitch, yaw), 'XYZ') 可生成该组合。
    """
    if origin_element is None:
        xyz = Vector((0.0, 0.0, 0.0))
        rpy = (0.0, 0.0, 0.0)
    else:
        xyz = parse_xyz(origin_element.get("xyz"))
        rpy = tuple(float(v) for v in origin_element.get(
            "rpy", "0 0 0"
        ).split())

    rotation = Euler(rpy, 'XYZ').to_matrix().to_4x4()
    translation = Matrix.Translation(xyz)
    return translation @ rotation


def resolve_mesh_path(filename):
    """解析 package:// 与 package:/ 形式的 mesh URI。"""
    filename = filename.strip()

    if filename.startswith("file://"):
        path = Path(filename[7:])
        if path.exists():
            return path

    if filename.startswith("/"):
        path = Path(filename)
        if path.exists():
            return path

    # 兼容 package://robot_description/meshes/a.STL
    # 以及 package:/robot_description/meshes/a.STL
    match = re.match(r"^package:/+([^/]+)/(.+)$", filename)
    if match:
        package_name, relative_path = match.groups()
        if package_name == PACKAGE_ROOT.name:
            candidate = PACKAGE_ROOT / relative_path
            if candidate.exists():
                return candidate

        # 如果 URI 中的包名不是目录名，仍尝试在 PACKAGE_ROOT 下解析
        candidate = PACKAGE_ROOT / relative_path
        if candidate.exists():
            return candidate

    # 兼容相对路径
    candidate = PACKAGE_ROOT / filename
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"无法解析 URDF mesh 路径：{filename}\n"
        f"尝试的资源根目录：{PACKAGE_ROOT}"
    )


def load_urdf(urdf_path):
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF 文件不存在：{urdf_path}")

    root = ET.parse(str(urdf_path)).getroot()

    links = {}
    for link_element in root.findall("link"):
        name = link_element.get("name")
        visual_items = []

        for visual_index, visual in enumerate(link_element.findall("visual")):
            origin = origin_to_matrix(visual.find("origin"))
            geometry = visual.find("geometry")
            if geometry is None:
                continue

            mesh_element = geometry.find("mesh")
            if mesh_element is None:
                # 本脚本当前针对 URDF 中的 STL mesh visual。
                # box/cylinder/sphere 等几何暂不自动创建。
                continue

            filename = mesh_element.get("filename")
            if not filename:
                continue

            scale = parse_xyz(mesh_element.get("scale"), (1.0, 1.0, 1.0))
            visual_items.append({
                "index": visual_index,
                "origin": origin,
                "filename": filename,
                "scale": scale,
            })

        links[name] = {
            "name": name,
            "visuals": visual_items,
        }

    joints = []
    child_to_joint = {}
    parent_to_joints = {}

    for joint_element in root.findall("joint"):
        name = joint_element.get("name")
        joint_type = joint_element.get("type", "fixed")

        parent_element = joint_element.find("parent")
        child_element = joint_element.find("child")
        if parent_element is None or child_element is None:
            raise ValueError(f"关节 {name} 缺少 parent 或 child")

        parent_link = parent_element.get("link")
        child_link = child_element.get("link")

        origin_element = joint_element.find("origin")
        origin_matrix = origin_to_matrix(origin_element)

        axis_element = joint_element.find("axis")
        axis = parse_xyz(
            axis_element.get("xyz") if axis_element is not None else None,
            (1.0, 0.0, 0.0),
        )
        if axis.length < 1e-9:
            axis = Vector((1.0, 0.0, 0.0))
        else:
            axis.normalize()

        limit_element = joint_element.find("limit")
        limit = {}
        if limit_element is not None:
            for key in ("lower", "upper", "effort", "velocity"):
                value = limit_element.get(key)
                if value is not None:
                    try:
                        limit[key] = float(value)
                    except ValueError:
                        pass

        joint = {
            "name": name,
            "type": joint_type,
            "parent": parent_link,
            "child": child_link,
            "origin": origin_matrix,
            "axis": axis,
            "limit": limit,
        }
        joints.append(joint)
        child_to_joint[child_link] = joint
        parent_to_joints.setdefault(parent_link, []).append(joint)

    missing_links = []
    for joint in joints:
        if joint["parent"] not in links:
            missing_links.append(joint["parent"])
        if joint["child"] not in links:
            missing_links.append(joint["child"])
    if missing_links:
        raise ValueError(f"URDF 关节引用了未定义的 link：{sorted(set(missing_links))}")

    child_links = {joint["child"] for joint in joints}
    root_links = [name for name in links if name not in child_links]

    return links, joints, child_to_joint, parent_to_joints, root_links


# ============================================================
# 计算 URDF 零位时每个 link 的世界变换
# ============================================================

def calculate_link_transforms(links, joints, child_to_joint, parent_to_joints, root_links):
    transforms = {}
    visiting = set()

    def visit(link_name, transform):
        if link_name in visiting:
            raise ValueError(f"URDF 关节结构存在环路，涉及 link：{link_name}")
        if link_name in transforms:
            return

        visiting.add(link_name)
        transforms[link_name] = transform.copy()

        for joint in parent_to_joints.get(link_name, []):
            child_transform = transform @ joint["origin"]
            visit(joint["child"], child_transform)

        visiting.remove(link_name)

    identity = Matrix.Identity(4)
    for root_link in root_links:
        visit(root_link, identity)

    # 检查是否有无法从根连杆到达的 link
    unvisited = sorted(set(links) - set(transforms))
    if unvisited:
        raise ValueError(
            "以下 link 无法从 URDF 根连杆到达，请检查关节结构："
            + ", ".join(unvisited)
        )

    return transforms


# ============================================================
# 场景与网格导入
# ============================================================

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 清理未使用的数据块，避免反复执行脚本后堆积
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.armatures,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def import_link_meshes(links, link_transforms):
    """导入 visual STL，先记录其正确的世界矩阵，稍后再绑定骨骼。"""
    imported_by_link = {name: [] for name in links}

    for link_name, link_data in links.items():
        link_world = link_transforms[link_name]

        for visual in link_data["visuals"]:
            mesh_path = resolve_mesh_path(visual["filename"])
            if not mesh_path.exists():
                raise FileNotFoundError(f"STL 文件不存在：{mesh_path}")

            before = set(bpy.context.scene.objects)

            # Blender 4.0.2 的 STL 导入接口
            bpy.ops.import_mesh.stl(filepath=str(mesh_path))

            created = [obj for obj in bpy.context.scene.objects if obj not in before]
            mesh_objects = [obj for obj in created if obj.type == 'MESH']
            if not mesh_objects:
                raise RuntimeError(f"导入 STL 后没有生成 Mesh 对象：{mesh_path}")

            # 一个 STL 通常生成一个对象；若导入器生成多个对象，均保留。
            visual_world = (
                link_world
                @ visual["origin"]
                @ Matrix.Diagonal((
                    visual["scale"].x,
                    visual["scale"].y,
                    visual["scale"].z,
                    1.0,
                ))
            )

            for sub_index, obj in enumerate(mesh_objects):
                if USE_STANDARD_OBJECT_NAMES:
                    suffix = f"__part_{sub_index}" if len(mesh_objects) > 1 else ""
                    obj.name = (
                        f"link__{link_name}__visual_{visual['index']}{suffix}"
                    )

                obj.matrix_world = visual_world.copy()
                obj["urdf_link_name"] = link_name
                obj["urdf_mesh_path"] = str(mesh_path)
                obj["urdf_visual_index"] = int(visual["index"])

                imported_by_link[link_name].append(obj)

            print(f"[成功] {link_name}: {mesh_path}")

    return imported_by_link


# ============================================================
# 自动创建 Armature
# ============================================================

def create_armature(links, joints, link_transforms, child_to_joint):
    armature_data = bpy.data.armatures.new("CatFace_ArmatureData")
    armature_obj = bpy.data.objects.new("CatFace_Armature", armature_data)
    bpy.context.collection.objects.link(armature_obj)

    # 让 Armature 的坐标系与 URDF 计算出的世界坐标一致。
    armature_obj.matrix_world = Matrix.Identity(4)
    armature_obj.show_in_front = True
    armature_data.display_type = 'STICK'

    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = {}
    bone_name_by_link = {
        link_name: f"link__{link_name}"
        for link_name in links
    }

    # 先创建全部骨骼。每根骨骼的 head 位于对应 link 的零位原点。
    for link_name in links:
        link_world = link_transforms[link_name]
        head = link_world.to_translation()

        joint = child_to_joint.get(link_name)
        if joint is not None:
            # URDF axis 在 joint/child link 局部坐标中表达。
            axis_local = joint["axis"]
            axis_world = link_world.to_3x3() @ axis_local
            if axis_world.length < 1e-9:
                axis_world = Vector((0.0, 1.0, 0.0))
            else:
                axis_world.normalize()
        else:
            # 根连杆没有输入关节轴，给它一个可见的默认骨骼方向。
            axis_world = Vector((0.0, 1.0, 0.0))

        tail = head + axis_world * BONE_DISPLAY_LENGTH
        if (tail - head).length < 1e-6:
            tail = head + Vector((0.0, BONE_DISPLAY_LENGTH, 0.0))

        bone = armature_data.edit_bones.new(bone_name_by_link[link_name])
        bone.head = head
        bone.tail = tail
        bone.use_connect = False
        edit_bones[link_name] = bone

    # 按 URDF joint parent/child 关系设置骨骼层级。
    for joint in joints:
        parent_link = joint["parent"]
        child_link = joint["child"]
        edit_bones[child_link].parent = edit_bones[parent_link]
        edit_bones[child_link].use_connect = False

    bpy.ops.object.mode_set(mode='OBJECT')

    # 设置骨骼姿态模式及 URDF 元数据。
    for link_name, bone_name in bone_name_by_link.items():
        data_bone = armature_data.bones[bone_name]
        data_bone["urdf_link_name"] = link_name

        joint = child_to_joint.get(link_name)
        if joint:
            data_bone["urdf_joint_name"] = joint["name"]
            data_bone["urdf_joint_type"] = joint["type"]
            data_bone["urdf_joint_axis"] = [
                float(joint["axis"].x),
                float(joint["axis"].y),
                float(joint["axis"].z),
            ]
            data_bone["urdf_joint_limit"] = joint["limit"]
        else:
            data_bone["urdf_joint_name"] = ""
            data_bone["urdf_joint_type"] = "root"
            data_bone["urdf_joint_axis"] = [0.0, 0.0, 0.0]

        pose_bone = armature_obj.pose.bones.get(bone_name)
        if pose_bone:
            pose_bone.rotation_mode = 'XYZ'

    # 保存便于后续映射使用的对象级信息。
    armature_obj["urdf_path"] = str(URDF_PATH)
    armature_obj["urdf_link_count"] = len(links)
    armature_obj["urdf_joint_count"] = len(joints)

    return armature_obj, bone_name_by_link


# ============================================================
# 将刚性 STL 网格绑定到对应 link 骨骼
# ============================================================

def parent_meshes_to_bones(imported_by_link, armature_obj, bone_name_by_link):
    """将网格作为 bone-parent 对象，并保持绑定前的世界矩阵。

    这是刚性零件绑定，不是软组织权重绑定。
    """
    bound_count = 0

    for link_name, objects in imported_by_link.items():
        bone_name = bone_name_by_link[link_name]

        for obj in objects:
            original_world = obj.matrix_world.copy()

            obj.parent = armature_obj
            obj.parent_type = 'BONE'
            obj.parent_bone = bone_name

            # 保持网格在绑定前的世界变换，避免初始姿态跳变。
            obj.matrix_world = original_world

            obj["urdf_link_name"] = link_name
            obj["armature_name"] = armature_obj.name
            obj["armature_bone_name"] = bone_name
            bound_count += 1

    return bound_count


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 64)
    print("开始导入 URDF 并创建 Blender Armature")
    print(f"URDF: {URDF_PATH}")
    print(f"资源包: {PACKAGE_ROOT}")
    print(f"输出工程: {OUTPUT_BLEND}")
    print("=" * 64)

    links, joints, child_to_joint, parent_to_joints, root_links = load_urdf(
        URDF_PATH
    )

    print(f"连杆数量: {len(links)}")
    print(f"关节数量: {len(joints)}")
    print(f"根连杆: {root_links}")

    link_transforms = calculate_link_transforms(
        links, joints, child_to_joint, parent_to_joints, root_links
    )

    if CLEAR_SCENE:
        clear_scene()

    imported_by_link = import_link_meshes(links, link_transforms)

    armature_obj, bone_name_by_link = create_armature(
        links, joints, link_transforms, child_to_joint
    )

    bound_count = parent_meshes_to_bones(
        imported_by_link, armature_obj, bone_name_by_link
    )

    # 选择 Armature，便于打开文件后查看骨架。
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # 确保输出目录存在。
    OUTPUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_BLEND))

    print("=" * 64)
    print("URDF 导入与骨架生成完成")
    print(f"URDF links: {len(links)}")
    print(f"URDF joints: {len(joints)}")
    print(f"Armature: {armature_obj.name}")
    print(f"Blender bones: {len(armature_obj.data.bones)}")
    print(f"绑定到骨骼的网格对象数: {bound_count}")
    print(f"已保存: {OUTPUT_BLEND}")
    print("=" * 64)
    print(
        "提示：打开保存后的 .blend，选中 CatFace_Armature，"
        "进入 Pose Mode 测试骨骼。请重点验证初始姿态和单关节运动。"
    )


if __name__ == "__main__":
    main()