#include "franka_robot_state_broadcaster/franka_robot_state_broadcaster.hpp"

#include <stddef.h>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/qos_event.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rcpputils/split.hpp"
#include "rcutils/logging_macros.h"
#include "std_msgs/msg/header.hpp"

namespace {

std::array<double, 16> identityTransform() {
  std::array<double, 16> transform{};
  transform[0] = 1.0;
  transform[5] = 1.0;
  transform[10] = 1.0;
  transform[15] = 1.0;
  return transform;
}

std::array<double, 16> multiplyColumnMajorTransforms(
    const std::array<double, 16>& lhs,
    const std::array<double, 16>& rhs) {
  std::array<double, 16> result{};
  for (size_t col = 0; col < 4; ++col) {
    for (size_t row = 0; row < 4; ++row) {
      double sum = 0.0;
      for (size_t k = 0; k < 4; ++k) {
        sum += lhs[k * 4 + row] * rhs[col * 4 + k];
      }
      result[col * 4 + row] = sum;
    }
  }
  return result;
}

std::array<double, 3> rotateVectorWithTransform(
    const std::array<double, 16>& transform,
    const std::array<double, 3>& vector) {
  std::array<double, 3> rotated{};
  for (size_t row = 0; row < 3; ++row) {
    rotated[row] = transform[0 * 4 + row] * vector[0] +
                   transform[1 * 4 + row] * vector[1] +
                   transform[2 * 4 + row] * vector[2];
  }
  return rotated;
}

void transformPose(
    const std::array<double, 16>& world_t_base,
    std::array<double, 16>& base_t_pose) {
  base_t_pose = multiplyColumnMajorTransforms(world_t_base, base_t_pose);
}

void transformSpatialVector(
    const std::array<double, 16>& world_t_base,
    std::array<double, 6>& base_vector) {
  std::array<double, 3> linear{{base_vector[0], base_vector[1], base_vector[2]}};
  std::array<double, 3> angular{{base_vector[3], base_vector[4], base_vector[5]}};
  const auto world_linear = rotateVectorWithTransform(world_t_base, linear);
  const auto world_angular = rotateVectorWithTransform(world_t_base, angular);
  for (size_t i = 0; i < 3; ++i) {
    base_vector[i] = world_linear[i];
    base_vector[i + 3] = world_angular[i];
  }
}

std::array<double, 16> createTransformFromTranslationQuaternion(
    const std::array<double, 3>& translation,
    const std::array<double, 4>& quaternion_xyzw) {
  std::array<double, 16> transform = identityTransform();

  double qx = quaternion_xyzw[0];
  double qy = quaternion_xyzw[1];
  double qz = quaternion_xyzw[2];
  double qw = quaternion_xyzw[3];

  const double norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  qx /= norm;
  qy /= norm;
  qz /= norm;
  qw /= norm;

  const double r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
  const double r01 = 2.0 * (qx * qy - qz * qw);
  const double r02 = 2.0 * (qx * qz + qy * qw);
  const double r10 = 2.0 * (qx * qy + qz * qw);
  const double r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
  const double r12 = 2.0 * (qy * qz - qx * qw);
  const double r20 = 2.0 * (qx * qz - qy * qw);
  const double r21 = 2.0 * (qy * qz + qx * qw);
  const double r22 = 1.0 - 2.0 * (qx * qx + qy * qy);

  transform[0] = r00;
  transform[1] = r10;
  transform[2] = r20;
  transform[4] = r01;
  transform[5] = r11;
  transform[6] = r21;
  transform[8] = r02;
  transform[9] = r12;
  transform[10] = r22;
  transform[12] = translation[0];
  transform[13] = translation[1];
  transform[14] = translation[2];

  return transform;
}

}  // namespace

namespace franka_robot_state_broadcaster {

controller_interface::CallbackReturn FrankaRobotStateBroadcaster::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
    auto_declare<int>("frequency", 30);
    auto_declare<std::vector<double>>("world_to_franka_base.translation",
                                      {0.0, 0.0, 0.0});
    auto_declare<std::vector<double>>("world_to_franka_base.rotation_xyzw",
                                      {0.0, 0.0, 0.0, 1.0});
    auto_declare<std::string>("world_frame_id", "world");

  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }

  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FrankaRobotStateBroadcaster::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id = get_node()->get_parameter("arm_id").as_string();
  frequency = get_node()->get_parameter("frequency").as_int();
  world_frame_id_ = get_node()->get_parameter("world_frame_id").as_string();
  last_pub_ = get_node()->now();
  franka_robot_state = std::make_unique<franka_semantic_components::FrankaRobotState>(
      franka_semantic_components::FrankaRobotState(arm_id + "/" + state_interface_name, arm_id));

  const auto translation_param =
      get_node()->get_parameter("world_to_franka_base.translation").as_double_array();
  const auto rotation_param =
      get_node()->get_parameter("world_to_franka_base.rotation_xyzw").as_double_array();

  if (translation_param.size() != 3) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "%s: world_to_franka_base.translation must have 3 values [x, y, z].",
                 arm_id.c_str());
    return CallbackReturn::ERROR;
  }
  if (rotation_param.size() != 4) {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "%s: world_to_franka_base.rotation_xyzw must have 4 values [x, y, z, w].",
        arm_id.c_str());
    return CallbackReturn::ERROR;
  }
  const double quat_norm = std::sqrt(
      rotation_param[0] * rotation_param[0] +
      rotation_param[1] * rotation_param[1] +
      rotation_param[2] * rotation_param[2] +
      rotation_param[3] * rotation_param[3]);
  if (quat_norm < 1e-9) {
    RCLCPP_ERROR(get_node()->get_logger(),
                 "%s: world_to_franka_base.rotation_xyzw has zero norm.",
                 arm_id.c_str());
    return CallbackReturn::ERROR;
  }
  const std::array<double, 3> translation{
      {translation_param[0], translation_param[1], translation_param[2]}};
  const std::array<double, 4> rotation_xyzw{
      {rotation_param[0], rotation_param[1], rotation_param[2], rotation_param[3]}};
  world_t_base_ =
      createTransformFromTranslationQuaternion(translation, rotation_xyzw);

  try {
    franka_state_publisher = get_node()->create_publisher<franka_msgs::msg::FrankaState>(
        "~/" + state_interface_name, rclcpp::SystemDefaultsQoS());
    realtime_franka_state_publisher =
        std::make_shared<realtime_tools::RealtimePublisher<franka_msgs::msg::FrankaState>>(
            franka_state_publisher);
    ;
  } catch (const std::exception& e) {
    fprintf(stderr,
            "Exception thrown during publisher creation at configure stage with message : %s \n",
            e.what());
    return CallbackReturn::ERROR;
  }
  RCLCPP_INFO(get_node()->get_logger(), "%s franka state broadcaster configuration successful", arm_id.c_str());
  return CallbackReturn::SUCCESS;
}


controller_interface::CallbackReturn FrankaRobotStateBroadcaster::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_state->assign_loaned_state_interfaces(state_interfaces_);
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FrankaRobotStateBroadcaster::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_state->release_interfaces();
  return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
FrankaRobotStateBroadcaster::command_interface_configuration() const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::NONE};
}

controller_interface::InterfaceConfiguration
FrankaRobotStateBroadcaster::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  state_interfaces_config.names = franka_robot_state->get_state_interface_names();
  return state_interfaces_config;
}

controller_interface::return_type FrankaRobotStateBroadcaster::update(
    const rclcpp::Time& time,
    const rclcpp::Duration& /*period*/) {
  if(time.nanoseconds() - last_pub_.nanoseconds() < 1'000'000'000 / frequency){
    return controller_interface::return_type::OK;
  }
  if (realtime_franka_state_publisher && realtime_franka_state_publisher->trylock()) {
    realtime_franka_state_publisher->msg_.header.stamp = time;
    realtime_franka_state_publisher->msg_.header.frame_id = world_frame_id_;

    if (!franka_robot_state->get_values_as_message(realtime_franka_state_publisher->msg_)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "Failed to get franka state via franka state interface.");
      realtime_franka_state_publisher->unlock();
      return controller_interface::return_type::ERROR;
    }

    transformPose(world_t_base_, realtime_franka_state_publisher->msg_.o_t_ee);
    transformPose(world_t_base_, realtime_franka_state_publisher->msg_.o_t_ee_d);
    transformPose(world_t_base_, realtime_franka_state_publisher->msg_.o_t_ee_c);
    transformSpatialVector(world_t_base_, realtime_franka_state_publisher->msg_.o_dp_ee_d);
    transformSpatialVector(world_t_base_, realtime_franka_state_publisher->msg_.o_dp_ee_c);
    transformSpatialVector(world_t_base_, realtime_franka_state_publisher->msg_.o_ddp_ee_c);

    realtime_franka_state_publisher->unlockAndPublish();
    last_pub_ = get_node()->now();
    return controller_interface::return_type::OK;

  } else {
    return controller_interface::return_type::ERROR;
  }
}

} // namespace franka_robot_state_broadcaster

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_robot_state_broadcaster::FrankaRobotStateBroadcaster,
                       controller_interface::ControllerInterface)
