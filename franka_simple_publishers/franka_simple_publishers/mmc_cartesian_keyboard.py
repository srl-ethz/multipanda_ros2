#!/usr/bin/env python3

import argparse
import math
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from franka_msgs.msg import FrankaState
from multi_mode_control_msgs.msg import (
    Controller,
)
from multi_mode_control_msgs.srv import SetControllers


def quaternion_from_rotation_matrix(rotation):
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2][1] - rotation[1][2]) / scale
        qy = (rotation[0][2] - rotation[2][0]) / scale
        qz = (rotation[1][0] - rotation[0][1]) / scale
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        qw = (rotation[2][1] - rotation[1][2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0][1] + rotation[1][0]) / scale
        qz = (rotation[0][2] + rotation[2][0]) / scale
    elif rotation[1][1] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        qw = (rotation[0][2] - rotation[2][0]) / scale
        qx = (rotation[0][1] + rotation[1][0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1][2] + rotation[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        qw = (rotation[1][0] - rotation[0][1]) / scale
        qx = (rotation[0][2] + rotation[2][0]) / scale
        qy = (rotation[1][2] + rotation[2][1]) / scale
        qz = 0.25 * scale

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (qw / norm, qx / norm, qy / norm, qz / norm)


class MMCCartesianKeyboard(Node):
    def __init__(
        self,
        mode,
        arm_id,
        left_arm_id,
        right_arm_id,
        step,
        controller_name,
        controller_service,
        state_topic,
        left_state_topic,
        right_state_topic,
        auto_switch,
    ):
        super().__init__("mmc_cartesian_keyboard")

        self.mode = mode
        self.is_dual_mode = self.mode == "dual"
        self.arm_id = arm_id
        self.left_arm_id = left_arm_id
        self.right_arm_id = right_arm_id
        self.step = step
        self.controller_name = controller_name
        self.controller_service = controller_service
        self.state_topic = state_topic
        self.left_state_topic = left_state_topic
        self.right_state_topic = right_state_topic
        self.auto_switch = auto_switch

        if self.is_dual_mode:
            self.resource = f"{self.left_arm_id}&{self.right_arm_id}"
            self.left_goal_topic = f"/{self.left_arm_id}/arm/end_effector_pose_cmd"
            self.right_goal_topic = f"/{self.right_arm_id}/arm/end_effector_pose_cmd"
            self.left_goal_pub = self.create_publisher(
                PoseStamped, self.left_goal_topic, 10
            )
            self.right_goal_pub = self.create_publisher(
                PoseStamped, self.right_goal_topic, 10
            )
            self.left_state_sub = self.create_subscription(
                FrankaState, self.left_state_topic, self._left_state_callback, 10
            )
            self.right_state_sub = self.create_subscription(
                FrankaState, self.right_state_topic, self._right_state_callback, 10
            )
        else:
            self.resource = self.arm_id
            self.goal_topic = f"/{self.arm_id}/arm/end_effector_pose_cmd"
            self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic, 10)
            self.state_sub = self.create_subscription(
                FrankaState, self.state_topic, self._state_callback, 10
            )

        self.set_controller_client = self.create_client(SetControllers, self.controller_service)
        self.fallback_controller_service = "/set_controllers"
        self.fallback_set_controller_client = None
        if self.controller_service != self.fallback_controller_service:
            self.fallback_set_controller_client = self.create_client(
                SetControllers, self.fallback_controller_service
            )

        self.have_state = False
        self.have_left_state = False
        self.have_right_state = False
        self.desired_initialized = False

        self.current_position = [0.0, 0.0, 0.0]
        self.current_orientation = (1.0, 0.0, 0.0, 0.0)
        self.desired_position = [0.0, 0.0, 0.0]
        self.desired_orientation = (1.0, 0.0, 0.0, 0.0)
        self.latest_q = [0.0] * 7

        self.left_current_position = [0.0, 0.0, 0.0]
        self.left_current_orientation = (1.0, 0.0, 0.0, 0.0)
        self.left_desired_position = [0.0, 0.0, 0.0]
        self.left_desired_orientation = (1.0, 0.0, 0.0, 0.0)
        self.left_latest_q = [0.0] * 7

        self.right_current_position = [0.0, 0.0, 0.0]
        self.right_current_orientation = (1.0, 0.0, 0.0, 0.0)
        self.right_desired_position = [0.0, 0.0, 0.0]
        self.right_desired_orientation = (1.0, 0.0, 0.0, 0.0)
        self.right_latest_q = [0.0] * 7

        self.max_position_offset = 0.09
        self.translation_bindings = {
            "w": (self.step, 0.0, 0.0),
            "s": (-self.step, 0.0, 0.0),
            "a": (0.0, self.step, 0.0),
            "d": (0.0, -self.step, 0.0),
            "r": (0.0, 0.0, self.step),
            "f": (0.0, 0.0, -self.step),
        }

        if self.is_dual_mode:
            self.get_logger().info(
                "Keyboard teleop ready. "
                "Mode: dual | Goal topics: "
                f"{self.left_goal_topic}, {self.right_goal_topic} | "
                f"State topics: {self.left_state_topic}, {self.right_state_topic}"
            )
        else:
            self.get_logger().info(
                "Keyboard teleop ready. "
                f"Mode: single | Goal topic: {self.goal_topic} | "
                f"State topic: {self.state_topic}",
            )
        self.print_help()

        if self.auto_switch:
            self._switch_timer = self.create_timer(1.0, self._switch_once)

    def print_help(self):
        print(
            "\nKeyboard controls:\n"
            "  w/s: +x/-x\n"
            "  a/d: +y/-y\n"
            "  r/f: +z/-z\n"
            "  (in dual mode, translations are applied to both arms)\n"
            "  c: switch MMC to cartesian impedance controller\n"
            "  space: reset desired pose to current pose\n"
            "  h: print help\n"
            "  q: quit\n"
        )

    def _switch_once(self):
        if self.switch_to_cartesian_controller():
            self._switch_timer.cancel()

    @staticmethod
    def _pose_from_state(msg):
        o_t_ee = msg.o_t_ee
        position = [o_t_ee[12], o_t_ee[13], o_t_ee[14]]
        rotation = [
            [o_t_ee[0], o_t_ee[4], o_t_ee[8]],
            [o_t_ee[1], o_t_ee[5], o_t_ee[9]],
            [o_t_ee[2], o_t_ee[6], o_t_ee[10]],
        ]
        orientation = quaternion_from_rotation_matrix(rotation)
        q = list(msg.q)
        return position, orientation, q

    def _state_callback(self, msg):
        self.current_position, self.current_orientation, self.latest_q = self._pose_from_state(msg)
        self.have_state = True
        if not self.is_dual_mode and not self.desired_initialized:
            self._initialize_desired_from_current()

    def _left_state_callback(self, msg):
        (
            self.left_current_position,
            self.left_current_orientation,
            self.left_latest_q,
        ) = self._pose_from_state(msg)
        self.have_left_state = True
        self._try_initialize_dual_desired()

    def _right_state_callback(self, msg):
        (
            self.right_current_position,
            self.right_current_orientation,
            self.right_latest_q,
        ) = self._pose_from_state(msg)
        self.have_right_state = True
        self._try_initialize_dual_desired()

    def _try_initialize_dual_desired(self):
        if not self.is_dual_mode:
            return
        if self.desired_initialized:
            return
        if self.have_left_state and self.have_right_state:
            self._initialize_desired_from_current()

    def _initialize_desired_from_current(self):
        if self.is_dual_mode:
            self.left_desired_position = self.left_current_position.copy()
            self.left_desired_orientation = self.left_current_orientation
            self.right_desired_position = self.right_current_position.copy()
            self.right_desired_orientation = self.right_current_orientation
            self.have_state = True
            self.desired_initialized = True
            self.get_logger().info(
                "Initialized desired poses from current robot poses (dual mode)."
            )
            self.publish_goal()
            return

        if not self.have_state:
            return
        self.desired_position = self.current_position.copy()
        self.desired_orientation = self.current_orientation
        self.desired_initialized = True
        self.get_logger().info("Initialized desired pose from current robot pose.")
        self.publish_goal()

    def switch_to_cartesian_controller(self):
        active_client = self.set_controller_client
        active_service = self.controller_service
        if not active_client.wait_for_service(timeout_sec=0.2):
            if self.fallback_set_controller_client is not None and self.fallback_set_controller_client.wait_for_service(
                timeout_sec=0.2
            ):
                active_client = self.fallback_set_controller_client
                active_service = self.fallback_controller_service
            else:
                self.get_logger().warn(
                    f"Controller services {self.controller_service} and {self.fallback_controller_service} are not available yet."
                )
                return False

        request = SetControllers.Request()
        controller = Controller()
        controller.name = self.controller_name
        controller.resources = [self.resource]
        request.controllers = [controller]

        future = active_client.call_async(request)
        future.add_done_callback(self._on_switch_done)
        self.get_logger().info(f"Sent controller switch request via {active_service}.")
        return True

    def _on_switch_done(self, future):
        if future.exception() is not None:
            self.get_logger().error(
                f"Failed to switch controller: {str(future.exception())}"
            )
            return
        self.get_logger().info(
            f"Requested controller switch to {self.controller_name} for resource {self.resource}."
        )

    def handle_key(self, key):
        if key == "h":
            self.print_help()
            return
        if key == "c":
            self.switch_to_cartesian_controller()
            return
        if key == " ":
            if self.have_state:
                self._initialize_desired_from_current()
            return

        if key not in self.translation_bindings:
            return

        if not self.have_state:
            if self.is_dual_mode:
                self.get_logger().warn(
                    "No FrankaState received yet for both arms. "
                    "Start the two robot_state broadcasters first."
                )
            else:
                self.get_logger().warn(
                    "No FrankaState received yet. Start franka_robot_state_broadcaster first."
                )
            return

        delta = self.translation_bindings[key]
        if self.is_dual_mode:
            for i in range(3):
                self.left_desired_position[i] += delta[i]
                self.right_desired_position[i] += delta[i]
        else:
            for i in range(3):
                self.desired_position[i] += delta[i]

        self._enforce_safety_window()
        self.publish_goal()

    def _clip_desired_position(self, desired_position, current_position):
        diff = [
            desired_position[0] - current_position[0],
            desired_position[1] - current_position[1],
            desired_position[2] - current_position[2],
        ]
        norm = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
        if norm <= self.max_position_offset or norm == 0.0:
            return desired_position
        scale = self.max_position_offset / norm
        return [
            current_position[0] + diff[0] * scale,
            current_position[1] + diff[1] * scale,
            current_position[2] + diff[2] * scale,
        ]

    def _enforce_safety_window(self):
        if self.is_dual_mode:
            self.left_desired_position = self._clip_desired_position(
                self.left_desired_position, self.left_current_position
            )
            self.right_desired_position = self._clip_desired_position(
                self.right_desired_position, self.right_current_position
            )
            return

        self.desired_position = self._clip_desired_position(
            self.desired_position, self.current_position
        )

    def _create_pose_stamped(self, position, orientation, stamp):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        pose_msg.pose.orientation.w = orientation[0]
        pose_msg.pose.orientation.x = orientation[1]
        pose_msg.pose.orientation.y = orientation[2]
        pose_msg.pose.orientation.z = orientation[3]
        return pose_msg

    def publish_goal(self):
        stamp = self.get_clock().now().to_msg()
        if self.is_dual_mode:
            left_pose_msg = self._create_pose_stamped(
                self.left_desired_position, self.left_desired_orientation, stamp
            )
            right_pose_msg = self._create_pose_stamped(
                self.right_desired_position, self.right_desired_orientation, stamp
            )
            self.left_goal_pub.publish(left_pose_msg)
            self.right_goal_pub.publish(right_pose_msg)
            return

        pose_msg = self._create_pose_stamped(
            self.desired_position, self.desired_orientation, stamp
        )
        self.goal_pub.publish(pose_msg)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Keyboard teleop for MMC Cartesian impedance controllers (single or dual arm), publishing PoseStamped end-effector commands."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "dual"],
        default="single",
        help="Use single-arm or dual-arm PoseStamped end-effector command publishing.",
    )
    parser.add_argument("--arm-id", default="panda")
    parser.add_argument("--left-arm-id", default="left")
    parser.add_argument("--right-arm-id", default="right")
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--controller-name",
        default=None,
        help="Controller name. Defaults to panda_cartesian_impedance_controller (single) "
        "or dual_cartesian_impedance_controller (dual).",
    )
    parser.add_argument(
        "--controller-service", default="/multi_mode_controller/set_controllers"
    )
    parser.add_argument(
        "--state-topic", default="/franka_robot_state_broadcaster/robot_state"
    )
    parser.add_argument(
        "--left-state-topic", default="/franka_left_robot_state_broadcaster/robot_state"
    )
    parser.add_argument(
        "--right-state-topic", default="/franka_right_robot_state_broadcaster/robot_state"
    )
    parser.add_argument("--no-auto-switch", action="store_true")
    return parser.parse_args(args)


def main(args=None):
    ros_args = args if args is not None else sys.argv
    rclpy.init(args=ros_args)
    parsed = parse_args(rclpy.utilities.remove_ros_args(args=ros_args)[1:])

    controller_name = parsed.controller_name
    if controller_name is None:
        if parsed.mode == "dual":
            controller_name = "dual_cartesian_impedance_controller"
        else:
            controller_name = "panda_cartesian_impedance_controller"

    node = MMCCartesianKeyboard(
        mode=parsed.mode,
        arm_id=parsed.arm_id,
        left_arm_id=parsed.left_arm_id,
        right_arm_id=parsed.right_arm_id,
        step=parsed.step,
        controller_name=controller_name,
        controller_service=parsed.controller_service,
        state_topic=parsed.state_topic,
        left_state_topic=parsed.left_state_topic,
        right_state_topic=parsed.right_state_topic,
        auto_switch=not parsed.no_auto_switch,
    )

    if not sys.stdin.isatty():
        node.get_logger().error("stdin is not a TTY. Run this node from a terminal.")
        node.destroy_node()
        rclpy.shutdown()
        return

    terminal_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key in ("q", "\x03"):
                break
            node.handle_key(key)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, terminal_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
