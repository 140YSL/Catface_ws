# -*- coding: utf-8 -*-
"""
Blender-side UDP receiver for the CatFace Armature. (FIXED VERSION)

Run inside the GUI Blender process, for example:
  blender /path/to/catface.blend --python /path/to/blender_joint_receiver.py

修复要点（相对旧版）：
  1. 不再使用 pose_bone.matrix 赋值（其换算依赖父骨骼已求值姿态，
     未经 update 时会引入一帧滞后反馈，表现为父关节运动时末端抖动）。
     改为直接计算并赋值 matrix_basis（纯局部量，与依赖图状态无关）。
  2. FK 与所有矩阵运算全部在 numpy float64 中进行，
     仅在最终赋值时接触 mathutils(float32)，避免精度放大。
  3. 每帧不再调用 view_layer.update()；GUI 重绘时依赖图自动刷新蒙皮。
  4. 增加死区(DEADBAND)：小于阈值的关节变化直接忽略，切断上游数值噪声。

Expected bone names: link__<URDF link name>
The Armature object is expected to be named CatFace_Armature.
"""

import bpy
import json
import math
import socket
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import numpy as np
from mathutils import Vector, Euler, Matrix

URDF_PATH = Path(bpy.path.abspath("//../urdf/robotic_cathead_v2.urdf"))

ARMATURE_OBJECT_NAME = "CatFace_Armature"
UDP_HOST = "127.0.0.1"
UDP_PORT = 5005
TIMER_INTERVAL = 1.0 / 100.0
DEADBAND = 1e-4          # rad，小于该值的变化视为噪声直接丢弃
BONE_PREFIX = "link__"

# ----------------------------- state -----------------------------
_udp_socket = None
_joint_positions = {}        # joint_name -> float (current, deadband-filtered)
_link_data = {}
_joint_data = []
_root_links = []
_parent_to_joints = {}
_child_to_joint = {}
_armature = None

_topo_order = []             # link names, parents before children (Blender bone tree)
_parent_link = {}            # link -> parent link name (from Blender bone hierarchy)
_origin_np = {}              # joint name -> 4x4 float64 origin
_axis_np = {}                # joint name -> 3 float64 axis
_rest_rel_inv = {}           # link -> inv(parent.matrix_local^-1 @ bone.matrix_local)
_pose_offset = {}            # link -> inv(T0_link) @ rest_bone_matrix
_last_error = None

node = None

class BlenderRelay(Node):
    def __init__(self):
        super().__init__('blender_state_relay')
        self.pub = self.create_publisher(JointState, '/joint_state_blender', 10)

# ----------------------------- helpers -----------------------------
def parse_xyz(text, default=(0.0, 0.0, 0.0)):
    if not text:
        return Vector(default)
    return Vector(tuple(float(x) for x in text.split()))


def origin_matrix(element):
    """URDF origin -> 4x4 (mathutils, parse-time only, precision irrelevant)."""
    if element is None:
        xyz = Vector((0.0, 0.0, 0.0))
        rpy = (0.0, 0.0, 0.0)
    else:
        xyz = parse_xyz(element.get("xyz"))
        rpy = tuple(float(x) for x in element.get("rpy", "0 0 0").split())
    # URDF RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return Matrix.Translation(xyz) @ Euler(rpy, 'XYZ').to_matrix().to_4x4()


def set_rendered_viewport():
    """把渲染引擎设为 EEVEE，并把所有 3D 视口切到 Rendered 着色模式。"""
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'RENDERED'
    print("[CatFace] Viewport switched to RENDERED mode.")


def parse_urdf():
    root = ET.parse(str(URDF_PATH)).getroot()
    links = {elem.get("name"): {} for elem in root.findall("link")}
    joints = []
    child_to_joint = {}
    parent_to_joints = {}

    for elem in root.findall("joint"):
        parent_elem = elem.find("parent")
        child_elem = elem.find("child")
        if parent_elem is None or child_elem is None:
            raise RuntimeError(f"Joint {elem.get('name')} has no parent/child.")
        parent = parent_elem.get("link")
        child = child_elem.get("link")
        axis_elem = elem.find("axis")
        axis = parse_xyz(
            axis_elem.get("xyz") if axis_elem is not None else None,
            (1.0, 0.0, 0.0),
        )
        if axis.length < 1e-9:
            axis = Vector((1.0, 0.0, 0.0))
        axis.normalize()

        limit_elem = elem.find("limit")
        limit = {}
        if limit_elem is not None:
            for key in ("lower", "upper"):
                value = limit_elem.get(key)
                if value is not None:
                    limit[key] = float(value)

        item = {
            "name": elem.get("name"),
            "type": elem.get("type", "fixed"),
            "parent": parent,
            "child": child,
            "origin": origin_matrix(elem.find("origin")),
            "axis": axis,
            "limit": limit,
        }
        joints.append(item)
        child_to_joint[child] = item
        parent_to_joints.setdefault(parent, []).append(item)

    child_links = {j["child"] for j in joints}
    roots = [name for name in links if name not in child_links]
    return links, joints, child_to_joint, parent_to_joints, roots


# ----------------------- numpy FK (float64) -----------------------
def rot_axis_angle(axis, q):
    """Rodrigues: 4x4 rotation of q rad about unit axis."""
    x, y, z = axis
    c = math.cos(q)
    s = math.sin(q)
    C = 1.0 - c
    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    return M


def motion_matrix_np(joint_name, q):
    kind = _joint_type[joint_name]
    if kind in ("revolute", "continuous"):
        return rot_axis_angle(_axis_np[joint_name], float(q))
    if kind == "prismatic":
        M = np.eye(4, dtype=np.float64)
        M[:3, 3] = _axis_np[joint_name] * float(q)
        return M
    return np.eye(4, dtype=np.float64)   # fixed / unknown


def compute_link_transforms(positions):
    """URDF FK in float64. Returns {link_name: 4x4 armature-space transform}."""
    T = {}
    visiting = set()

    def visit(link_name, M):
        if link_name in visiting:
            raise RuntimeError(f"Cycle in URDF link graph at {link_name}")
        if link_name in T:
            return
        visiting.add(link_name)
        T[link_name] = M
        for joint in _parent_to_joints.get(link_name, []):
            q = positions.get(joint["name"], 0.0)
            visit(joint["child"], M @ _origin_np[joint["name"]]
                  @ motion_matrix_np(joint["name"], q))
        visiting.remove(link_name)

    for root_name in _root_links:
        visit(root_name, np.eye(4, dtype=np.float64))

    missing = set(_link_data) - set(T)
    if missing:
        raise RuntimeError(
            "URDF links not reachable from roots: " + ", ".join(sorted(missing)))
    return T


# ----------------------------- setup -----------------------------
def setup():
    global _udp_socket, _link_data, _joint_data, _root_links
    global _parent_to_joints, _child_to_joint, _armature
    global _topo_order, _parent_link, _origin_np, _axis_np
    global _rest_rel_inv, _pose_offset, _joint_type

    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF not found: {URDF_PATH}")

    _link_data, _joint_data, _child_to_joint, _parent_to_joints, _root_links = \
        parse_urdf()

    _armature = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if _armature is None or _armature.type != "ARMATURE":
        raise RuntimeError(
            f"Armature '{ARMATURE_OBJECT_NAME}' not found. "
            "Open the .blend file generated by the armature script."
        )

    # --- validate bones & build Blender-hierarchy mappings ---
    bone_names = set()
    for link_name in _link_data:
        bone_name = f"{BONE_PREFIX}{link_name}"
        if _armature.data.bones.get(bone_name) is None:
            raise RuntimeError(
                f"Bone '{bone_name}' not found in {_armature.name}; "
                "check your generated bone names."
            )
        bone_names.add(bone_name)

    for link_name in _link_data:
        bone = _armature.data.bones[f"{BONE_PREFIX}{link_name}"]
        if bone.parent is not None and bone.parent.name in bone_names:
            _parent_link[link_name] = bone.parent.name[len(BONE_PREFIX):]

    # topological order over the Blender bone tree (parents first)
    visited = set()
    def walk(bone):
        if bone.name in visited:
            return
        visited.add(bone.name)
        _topo_order.append(bone.name[len(BONE_PREFIX):])
        for child in bone.children:
            walk(child)

    for bone in _armature.data.bones:
        if bone.parent is None:
            walk(bone)
    for link_name in _link_data:   # fallback: anything missed
        if link_name not in _topo_order:
            _topo_order.append(link_name)

    # --- numpy conversions of static data ---
    _origin_np = {
        j["name"]: np.array(j["origin"], dtype=np.float64).reshape(4, 4)
        for j in _joint_data
    }
    _axis_np = {
        j["name"]: np.array(j["axis"], dtype=np.float64)
        for j in _joint_data
    }
    _joint_type = {j["name"]: j["type"] for j in _joint_data}

    # --- rest matrices & basis conversion terms (float64) ---
    rest = {}
    for link_name in _link_data:
        bone = _armature.data.bones[f"{BONE_PREFIX}{link_name}"]
        rest[link_name] = np.array(bone.matrix_local, dtype=np.float64).reshape(4, 4)

    for link_name in _link_data:
        parent = _parent_link.get(link_name)
        if parent is not None:
            rest_rel = np.linalg.inv(rest[parent]) @ rest[link_name]
        else:
            rest_rel = rest[link_name]
        _rest_rel_inv[link_name] = np.linalg.inv(rest_rel)

    # --- link frame -> bone rest frame offset (as in the old script) ---
    T0 = compute_link_transforms({})
    for link_name in _link_data:
        _pose_offset[link_name] = (
            np.linalg.inv(T0[link_name]) @ rest[link_name]
        )

    # --- UDP ---
    _udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _udp_socket.bind((UDP_HOST, UDP_PORT))
    _udp_socket.setblocking(False)

    print(
        f"[CatFace] Receiver ready on {UDP_HOST}:{UDP_PORT}; "
        f"{len(_link_data)} links, {len(_joint_data)} joints, "
        f"deadband {DEADBAND:g} rad."
    )


# ----------------------------- apply -----------------------------
def apply_joint_positions():
    """
    Direct matrix_basis assignment.

    For each bone (parents first):
        D        = T_link @ pose_offset          # desired armature-space matrix
        local    = inv(D_parent) @ D             # parent-relative target
        basis    = inv(rest_rel) @ local         # pure local, no depsgraph reads
    No view_layer.update() needed; skin refreshes on redraw.
    """
    global node
    T = compute_link_transforms(_joint_positions)
    eye = np.eye(4, dtype=np.float64)

    for link_name in _topo_order:
        D = T[link_name] @ _pose_offset[link_name]
        parent = _parent_link.get(link_name)
        Dp = (T[parent] @ _pose_offset[parent]) if parent is not None else eye
        basis = _rest_rel_inv[link_name] @ (np.linalg.inv(Dp) @ D)
        pose_bone = _armature.pose.bones[f"{BONE_PREFIX}{link_name}"]
        pose_bone.matrix_basis = Matrix(basis.tolist())
    
    msg = JointState()
    # 用 Blender 记录的时刻作为 header.stamp（墙钟）
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "blender"

    msg.name = list(_joint_positions.keys())
    msg.position = list(_joint_positions.values())
    node.pub.publish(msg)

# ----------------------------- timer -----------------------------
def timer_callback():
    global _last_error
    if _udp_socket is None:
        return None

    changed = False
    while True:
        try:
            data, _address = _udp_socket.recvfrom(65535)
        except BlockingIOError:
            break
        except OSError as exc:
            print(f"[CatFace] UDP receive error: {exc}")
            break

        try:
            packet = json.loads(data.decode("utf-8"))
        except Exception as exc:
            print(f"[CatFace] Invalid packet: {exc}")
            continue

        if packet.get("type") != "joint_state":
            continue

        positions = packet.get("positions", {})
        for name, value in positions.items():
            v = float(value)
            old = _joint_positions.get(name)
            if old is not None and abs(v - old) < DEADBAND:
                continue          # noise -> keep old value, do NOT trigger apply
            _joint_positions[name] = v
            changed = True

    if changed:
        try:
            apply_joint_positions()
            _last_error = None
        except Exception as exc:
            if str(exc) != _last_error:
                print(f"[CatFace] Pose update failed: {exc}")
                _last_error = str(exc)

    return TIMER_INTERVAL


def stop_receiver():
    global _udp_socket
    if _udp_socket is not None:
        try:
            _udp_socket.close()
        except Exception:
            pass
        _udp_socket = None
    print("[CatFace] UDP receiver stopped.")


# ----------------------------- main -----------------------------
if __name__ == "__main__":
    rclpy.init()
    node = BlenderRelay()
    try:
        setup()
        set_rendered_viewport()
        if bpy.app.timers.is_registered(timer_callback):
            bpy.app.timers.unregister(timer_callback)
        bpy.app.timers.register(timer_callback, first_interval=TIMER_INTERVAL)
    except Exception as exc:
        print(f"[CatFace] Startup failed: {exc}")
        raise
