#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 -> Blender joint-state bridge.

Subscribes to sensor_msgs/msg/JointState (default: /joint_states) and forwards
joint names/positions to the Blender receiver over UDP as JSON.

This node does not run MuJoCo. Your MuJoCo process/node must publish the
JointState topic for the bridge to receive data.
"""
import json
import socket
from typing import Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class MujocoBlenderBridge(Node):
    def __init__(self):
        super().__init__("mujoco_blender_bridge")

        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("udp_host", "127.0.0.1")
        self.declare_parameter("udp_port", 5005)
        self.declare_parameter("send_rate_hz", 60.0)

        topic = str(self.get_parameter("joint_states_topic").value)
        self.udp_host = str(self.get_parameter("udp_host").value)
        self.udp_port = int(self.get_parameter("udp_port").value)
        self.send_rate_hz = max(
            1.0, float(self.get_parameter("send_rate_hz").value)
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.latest_positions: Dict[str, float] = {}
        self.last_sent = {}
        self.sub = self.create_subscription(
            JointState, topic, self.joint_state_callback, 10
        )
        self.timer = self.create_timer(1.0 / self.send_rate_hz, self.send_latest)

        self.get_logger().info(
            f"Subscribing {topic}; forwarding UDP to "
            f"{self.udp_host}:{self.udp_port} at up to {self.send_rate_hz:g} Hz"
        )

    def joint_state_callback(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            self.get_logger().warning(
                "Ignoring JointState: name and position lengths differ."
            )
            return
        self.latest_positions = {
            name: float(position)
            for name, position in zip(msg.name, msg.position)
        }

    def send_latest(self):
        if not self.latest_positions:
            return
        packet = {
            "type": "joint_state",
            "positions": self.latest_positions,
        }
        try:
            data = json.dumps(packet, separators=(",", ":")).encode("utf-8")
            self.sock.sendto(data, (self.udp_host, self.udp_port))
            self.last_sent = dict(self.latest_positions)
        except OSError as exc:
            self.get_logger().warning(f"UDP send failed: {exc}")

    def destroy_node(self):
        try:
            self.sock.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoBlenderBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
