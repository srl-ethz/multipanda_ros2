#!/usr/bin/env python3

import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node


class CartesianPosePublisher(Node):
    def __init__(self):
        super().__init__("cartesian_pose_publisher")

        self.declare_parameter("left_arm_id", "mj_left")
        self.declare_parameter("right_arm_id", "mj_right")
        self.declare_parameter("hardware_layout_node", "/hardware_layout")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("hardware_layout_wait_sec", 5.0)
        self.declare_parameter("frame_id", "")

        self.left_arm_id = str(self.get_parameter("left_arm_id").value)
        self.right_arm_id = str(self.get_parameter("right_arm_id").value)
        self.arm_ids = [self.left_arm_id, self.right_arm_id]
        self.hardware_layout_node = str(
            self.get_parameter("hardware_layout_node").value
        )
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.hardware_layout_wait_sec = float(
            self.get_parameter("hardware_layout_wait_sec").value
        )

        if not self.left_arm_id:
            raise RuntimeError("Parameter 'left_arm_id' must be non-empty.")
        if not self.right_arm_id:
            raise RuntimeError("Parameter 'right_arm_id' must be non-empty.")
        if self.publish_rate_hz <= 0.0:
            raise RuntimeError("Parameter 'publish_rate_hz' must be > 0.")
        if self.hardware_layout_wait_sec < 0.0:
            raise RuntimeError("Parameter 'hardware_layout_wait_sec' must be >= 0.")

        self.left_goal_topic = f"/{self.left_arm_id}/arm/end_effector_pose_cmd"
        self.right_goal_topic = f"/{self.right_arm_id}/arm/end_effector_pose_cmd"
        self.left_goal_pub = self.create_publisher(PoseStamped, self.left_goal_topic, 10)
        self.right_goal_pub = self.create_publisher(
            PoseStamped, self.right_goal_topic, 10
        )

        hardware_layout_node = self.hardware_layout_node.rstrip("/")
        if not hardware_layout_node:
            hardware_layout_node = "/hardware_layout"
        self.hardware_layout_get_parameters_service = (
            f"{hardware_layout_node}/get_parameters"
        )
        self.get_parameters_client = self.create_client(
            GetParameters, self.hardware_layout_get_parameters_service
        )
        if not self.get_parameters_client.wait_for_service(
            timeout_sec=self.hardware_layout_wait_sec
        ):
            raise RuntimeError(
                f"Parameter service '{self.hardware_layout_get_parameters_service}' is not available "
                f"after {self.hardware_layout_wait_sec:.1f} s."
            )

        self.default_poses = self._load_default_poses()
        self.frame_id = self._resolve_frame_id()
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_all)

        self.get_logger().info(
            "Publishing default world poses as PoseStamped to "
            f"{self.left_goal_topic} and {self.right_goal_topic} "
            f"at {self.publish_rate_hz:.1f} Hz."
        )

    def _get_parameters_or_raise(self, names):
        request = GetParameters.Request()
        request.names = names
        future = self.get_parameters_client.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=self.hardware_layout_wait_sec
        )
        if not future.done() or future.result() is None:
            raise RuntimeError(
                f"Timed out while reading parameters from '{self.hardware_layout_node}'."
            )
        return future.result().values

    def _read_double_array_parameter(self, name, expected_len):
        values = self._get_parameters_or_raise([name])
        value = values[0]
        if value.type == ParameterType.PARAMETER_NOT_SET:
            raise RuntimeError(
                f"Missing parameter '{name}' on node '{self.hardware_layout_node}'."
            )
        if value.type != ParameterType.PARAMETER_DOUBLE_ARRAY:
            raise RuntimeError(
                f"Parameter '{name}' must be a double array, got type={value.type}."
            )
        data = list(value.double_array_value)
        if len(data) != expected_len:
            raise RuntimeError(
                f"Parameter '{name}' must have {expected_len} elements, got {len(data)}."
            )
        return data

    def _load_default_poses(self):
        poses = {}
        for arm_id in self.arm_ids:
            translation_name = f"default_pose_world.{arm_id}.translation"
            rotation_name = f"default_pose_world.{arm_id}.rotation_xyzw"
            translation = self._read_double_array_parameter(translation_name, 3)
            rotation_xyzw = self._read_double_array_parameter(rotation_name, 4)
            poses[arm_id] = {
                "translation": translation,
                "rotation_xyzw": rotation_xyzw,
            }
            self.get_logger().info(
                "Loaded default pose for %s: translation=%s rotation_xyzw=%s"
                % (arm_id, translation, rotation_xyzw)
            )
        return poses

    def _resolve_frame_id(self):
        explicit_frame_id = str(self.get_parameter("frame_id").value)
        if explicit_frame_id:
            return explicit_frame_id

        values = self._get_parameters_or_raise(["world_frame_id"])
        value = values[0]
        if value.type == ParameterType.PARAMETER_STRING:
            return value.string_value
        if value.type == ParameterType.PARAMETER_NOT_SET:
            self.get_logger().warn(
                "Parameter 'world_frame_id' is not set on '%s'. Using 'world'."
                % self.hardware_layout_node
            )
            return "world"
        self.get_logger().warn(
            "Parameter 'world_frame_id' has unexpected type=%d on '%s'. Using 'world'."
            % (value.type, self.hardware_layout_node)
        )
        return "world"

    def _build_pose_msg(self, arm_id):
        pose_cfg = self.default_poses[arm_id]
        translation = pose_cfg["translation"]
        rotation_xyzw = pose_cfg["rotation_xyzw"]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = translation[0]
        msg.pose.position.y = translation[1]
        msg.pose.position.z = translation[2]
        msg.pose.orientation.x = rotation_xyzw[0]
        msg.pose.orientation.y = rotation_xyzw[1]
        msg.pose.orientation.z = rotation_xyzw[2]
        msg.pose.orientation.w = rotation_xyzw[3]
        return msg

    def _publish_all(self):
        self.left_goal_pub.publish(self._build_pose_msg(self.left_arm_id))
        self.right_goal_pub.publish(self._build_pose_msg(self.right_arm_id))


def main(args=None):
    ros_args = args if args is not None else sys.argv
    rclpy.init(args=ros_args)
    node = None
    try:
        node = CartesianPosePublisher()
        rclpy.spin(node)
    except Exception as exc:
        print(f"[cartesian_pose_publisher] {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
