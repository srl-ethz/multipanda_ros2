#!/usr/bin/env python3

import argparse
import math
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node

from franka_msgs.msg import FrankaState
from multi_mode_control_msgs.msg import CartesianImpedanceGoal, Controller
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
        arm_id,
        step,
        controller_name,
        controller_service,
        state_topic,
        auto_switch,
    ):
        super().__init__("mmc_cartesian_keyboard")

        self.arm_id = arm_id
        self.step = step
        self.controller_name = controller_name
        self.controller_service = controller_service
        self.state_topic = state_topic
        self.auto_switch = auto_switch

        self.goal_topic = f"/{self.arm_id}/{self.controller_name}/desired_pose"
        self.goal_pub = self.create_publisher(CartesianImpedanceGoal, self.goal_topic, 10)
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
        self.current_position = [0.0, 0.0, 0.0]
        self.current_orientation = (1.0, 0.0, 0.0, 0.0)
        self.desired_position = [0.0, 0.0, 0.0]
        self.desired_orientation = (1.0, 0.0, 0.0, 0.0)
        self.latest_q = [0.0] * 7

        self.max_position_offset = 0.09
        self.translation_bindings = {
            "w": (self.step, 0.0, 0.0),
            "s": (-self.step, 0.0, 0.0),
            "a": (0.0, self.step, 0.0),
            "d": (0.0, -self.step, 0.0),
            "r": (0.0, 0.0, self.step),
            "f": (0.0, 0.0, -self.step),
        }

        self.get_logger().info(
            f"Keyboard teleop ready. Goal topic: {self.goal_topic} | State topic: {self.state_topic}",
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
            "  c: switch MMC to cartesian impedance controller\n"
            "  space: reset desired pose to current pose\n"
            "  h: print help\n"
            "  q: quit\n"
        )

    def _switch_once(self):
        if self.switch_to_cartesian_controller():
            self._switch_timer.cancel()

    def _state_callback(self, msg):
        # FrankaState::o_t_ee is a 4x4 transform in column-major order.
        o_t_ee = msg.o_t_ee
        self.current_position = [o_t_ee[12], o_t_ee[13], o_t_ee[14]]
        rotation = [
            [o_t_ee[0], o_t_ee[4], o_t_ee[8]],
            [o_t_ee[1], o_t_ee[5], o_t_ee[9]],
            [o_t_ee[2], o_t_ee[6], o_t_ee[10]],
        ]
        self.current_orientation = quaternion_from_rotation_matrix(rotation)
        self.latest_q = list(msg.q)

        if not self.have_state:
            self.have_state = True
            self.desired_position = self.current_position.copy()
            self.desired_orientation = self.current_orientation
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
        controller.resources = [self.arm_id]
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
            f"Requested controller switch to {self.controller_name} for resource {self.arm_id}."
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
                self.desired_position = self.current_position.copy()
                self.desired_orientation = self.current_orientation
                self.publish_goal()
            return

        if key not in self.translation_bindings:
            return

        if not self.have_state:
            self.get_logger().warn(
                "No FrankaState received yet. Start franka_robot_state_broadcaster first."
            )
            return

        delta = self.translation_bindings[key]
        for i in range(3):
            self.desired_position[i] += delta[i]

        self._enforce_safety_window()
        self.publish_goal()

    def _enforce_safety_window(self):
        diff = [
            self.desired_position[0] - self.current_position[0],
            self.desired_position[1] - self.current_position[1],
            self.desired_position[2] - self.current_position[2],
        ]
        norm = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
        if norm <= self.max_position_offset or norm == 0.0:
            return
        scale = self.max_position_offset / norm
        self.desired_position = [
            self.current_position[0] + diff[0] * scale,
            self.current_position[1] + diff[1] * scale,
            self.current_position[2] + diff[2] * scale,
        ]

    def publish_goal(self):
        goal = CartesianImpedanceGoal()
        goal.pose.position.x = self.desired_position[0]
        goal.pose.position.y = self.desired_position[1]
        goal.pose.position.z = self.desired_position[2]
        goal.pose.orientation.w = self.desired_orientation[0]
        goal.pose.orientation.x = self.desired_orientation[1]
        goal.pose.orientation.y = self.desired_orientation[2]
        goal.pose.orientation.z = self.desired_orientation[3]
        goal.q_n = self.latest_q
        self.goal_pub.publish(goal)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Keyboard teleop for single-arm MMC cartesian impedance controller."
    )
    parser.add_argument("--arm-id", default="panda")
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--controller-name", default="panda_cartesian_impedance_controller"
    )
    parser.add_argument(
        "--controller-service", default="/multi_mode_controller/set_controllers"
    )
    parser.add_argument(
        "--state-topic", default="/franka_robot_state_broadcaster/robot_state"
    )
    parser.add_argument("--no-auto-switch", action="store_true")
    return parser.parse_args(args)


def main(args=None):
    ros_args = args if args is not None else sys.argv
    rclpy.init(args=ros_args)
    parsed = parse_args(rclpy.utilities.remove_ros_args(args=ros_args)[1:])

    node = MMCCartesianKeyboard(
        arm_id=parsed.arm_id,
        step=parsed.step,
        controller_name=parsed.controller_name,
        controller_service=parsed.controller_service,
        state_topic=parsed.state_topic,
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
