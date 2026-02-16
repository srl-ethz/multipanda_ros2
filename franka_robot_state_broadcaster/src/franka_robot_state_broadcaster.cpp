#include "franka_robot_state_broadcaster/franka_robot_state_broadcaster.hpp"

#include <stddef.h>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/parameter_client.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/qos_event.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rcpputils/split.hpp"
#include "rcutils/logging_macros.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
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

std::array<double, 4> quaternionFromColumnMajorRotation(
    const std::array<double, 16>& transform) {
  const double r00 = transform[0];
  const double r01 = transform[4];
  const double r02 = transform[8];
  const double r10 = transform[1];
  const double r11 = transform[5];
  const double r12 = transform[9];
  const double r20 = transform[2];
  const double r21 = transform[6];
  const double r22 = transform[10];

  double qw = 1.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;

  const double trace = r00 + r11 + r22;
  if (trace > 0.0) {
    const double scale = std::sqrt(trace + 1.0) * 2.0;
    qw = 0.25 * scale;
    qx = (r21 - r12) / scale;
    qy = (r02 - r20) / scale;
    qz = (r10 - r01) / scale;
  } else if (r00 > r11 && r00 > r22) {
    const double scale = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
    qw = (r21 - r12) / scale;
    qx = 0.25 * scale;
    qy = (r01 + r10) / scale;
    qz = (r02 + r20) / scale;
  } else if (r11 > r22) {
    const double scale = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
    qw = (r02 - r20) / scale;
    qx = (r01 + r10) / scale;
    qy = 0.25 * scale;
    qz = (r12 + r21) / scale;
  } else {
    const double scale = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
    qw = (r10 - r01) / scale;
    qx = (r02 + r20) / scale;
    qy = (r12 + r21) / scale;
    qz = 0.25 * scale;
  }

  const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  if (norm < 1e-9) {
    return {{1.0, 0.0, 0.0, 0.0}};
  }
  return {{qw / norm, qx / norm, qy / norm, qz / norm}};
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

bool loadDoubleArrayParameterFromHardwareLayout(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
    const std::shared_ptr<rclcpp::SyncParametersClient>& parameter_client,
    const std::string& parameter_name,
    bool& has_parameter,
    std::vector<double>& values) {
  has_parameter = false;
  values.clear();
  try {
    if (!parameter_client) {
      return true;
    }
    const auto parameters = parameter_client->get_parameters({parameter_name});
    if (parameters.empty()) {
      return true;
    }
    const auto& parameter = parameters.front();
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
      return true;
    }
    if (parameter.get_type() !=
        rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
      RCLCPP_ERROR(node->get_logger(),
                   "Parameter '/hardware_layout.%s' must be a double array.",
                   parameter_name.c_str());
      return false;
    }
    has_parameter = true;
    values = parameter.as_double_array();
    return true;
  } catch (const std::exception& e) {
    RCLCPP_WARN(node->get_logger(),
                "Failed to read '/hardware_layout.%s': %s",
                parameter_name.c_str(), e.what());
    return true;
  }
}

bool loadStringParameterFromHardwareLayout(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
    const std::shared_ptr<rclcpp::SyncParametersClient>& parameter_client,
    const std::string& parameter_name,
    bool& has_parameter,
    std::string& value) {
  has_parameter = false;
  try {
    if (!parameter_client) {
      return true;
    }
    const auto parameters = parameter_client->get_parameters({parameter_name});
    if (parameters.empty()) {
      return true;
    }
    const auto& parameter = parameters.front();
    if (parameter.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
      return true;
    }
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
      RCLCPP_ERROR(node->get_logger(),
                   "Parameter '/hardware_layout.%s' must be a string.",
                   parameter_name.c_str());
      return false;
    }
    has_parameter = true;
    value = parameter.as_string();
    return true;
  } catch (const std::exception& e) {
    RCLCPP_WARN(node->get_logger(),
                "Failed to read '/hardware_layout.%s': %s",
                parameter_name.c_str(), e.what());
    return true;
  }
}

}  // namespace

namespace franka_robot_state_broadcaster {

controller_interface::CallbackReturn FrankaRobotStateBroadcaster::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
    auto_declare<int>("frequency", 30);
    auto_declare<int>("hardware_layout_wait_ms", 5000);
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
  int hardware_layout_wait_ms =
      get_node()->get_parameter("hardware_layout_wait_ms").as_int();
  if (hardware_layout_wait_ms < 0) {
    RCLCPP_WARN(
        get_node()->get_logger(),
        "%s: hardware_layout_wait_ms (%d) cannot be negative. Clamping to 0 ms.",
        arm_id.c_str(), hardware_layout_wait_ms);
    hardware_layout_wait_ms = 0;
  }
  world_frame_id_ = get_node()->get_parameter("world_frame_id").as_string();
  last_pub_ = get_node()->now();
  franka_robot_state = std::make_unique<franka_semantic_components::FrankaRobotState>(
      franka_semantic_components::FrankaRobotState(arm_id + "/" + state_interface_name, arm_id));

  const std::string global_translation_param =
      "world_to_franka_base." + arm_id + ".translation";
  const std::string global_rotation_param =
      "world_to_franka_base." + arm_id + ".rotation_xyzw";

  std::vector<double> translation_param;
  std::vector<double> rotation_param;
  bool has_global_translation = false;
  bool has_global_rotation = false;
  bool has_global_world_frame = false;

  rclcpp::NodeOptions hardware_layout_client_node_options;
  hardware_layout_client_node_options.context(
      get_node()->get_node_base_interface()->get_context());
  hardware_layout_client_node_options.use_global_arguments(false);
  hardware_layout_client_node_options.start_parameter_services(false);
  hardware_layout_client_node_options.start_parameter_event_publisher(false);
  const auto hardware_layout_client_node =
      std::make_shared<rclcpp::Node>(
          arm_id + "_hardware_layout_client",
          hardware_layout_client_node_options);
  auto hardware_layout_client =
      std::make_shared<rclcpp::SyncParametersClient>(
          hardware_layout_client_node, "/hardware_layout");
  const auto hardware_layout_wait_timeout =
      std::chrono::milliseconds(hardware_layout_wait_ms);
  if (!hardware_layout_client->wait_for_service(hardware_layout_wait_timeout)) {
    RCLCPP_WARN(
        get_node()->get_logger(),
        "%s: '/hardware_layout/get_parameters' not available after %d ms. "
        "Global hardware layout overrides will be skipped.",
        arm_id.c_str(), hardware_layout_wait_ms);
    hardware_layout_client.reset();
  }

  if (!loadDoubleArrayParameterFromHardwareLayout(
          get_node(), hardware_layout_client, global_translation_param, has_global_translation,
          translation_param) ||
      !loadDoubleArrayParameterFromHardwareLayout(
          get_node(), hardware_layout_client, global_rotation_param, has_global_rotation,
          rotation_param) ||
      !loadStringParameterFromHardwareLayout(
          get_node(), hardware_layout_client, "world_frame_id", has_global_world_frame, world_frame_id_)) {
    return CallbackReturn::ERROR;
  }

  if (!has_global_translation && !has_global_rotation) {
    if (hardware_layout_client) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "%s: '/hardware_layout.%s' and '/hardware_layout.%s' were not found. "
          "Falling back to local controller parameters. Check arm_id and key names in hardware_layout.yaml.",
          arm_id.c_str(), global_translation_param.c_str(),
          global_rotation_param.c_str());
    }
    translation_param =
        get_node()->get_parameter("world_to_franka_base.translation")
            .as_double_array();
    rotation_param =
        get_node()->get_parameter("world_to_franka_base.rotation_xyzw")
            .as_double_array();
  } else if (!has_global_translation || !has_global_rotation) {
    RCLCPP_ERROR(
        get_node()->get_logger(),
        "%s: both '/hardware_layout.%s' and '/hardware_layout.%s' must be set.",
        arm_id.c_str(), global_translation_param.c_str(),
        global_rotation_param.c_str());
    return CallbackReturn::ERROR;
  }

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

  if (has_global_translation && has_global_rotation) {
    if (has_global_world_frame) {
      RCLCPP_INFO(
          get_node()->get_logger(),
          "%s: loaded world_to_franka_base and world_frame_id from /hardware_layout.",
          arm_id.c_str());
    } else {
      RCLCPP_INFO(
          get_node()->get_logger(),
          "%s: loaded world_to_franka_base from /hardware_layout and kept local world_frame_id.",
          arm_id.c_str());
    }
  } else {
    RCLCPP_INFO(
        get_node()->get_logger(),
        "%s: /hardware_layout parameters not found, using local controller parameters.",
        arm_id.c_str());
  }

  const bool using_global_transform = has_global_translation && has_global_rotation;
  RCLCPP_INFO(
      get_node()->get_logger(),
      "%s: world_t_base source=%s world_frame_id='%s' translation=[%.6f, %.6f, %.6f] "
      "rotation_xyzw=[%.6f, %.6f, %.6f, %.6f]",
      arm_id.c_str(), using_global_transform ? "/hardware_layout" : "local",
      world_frame_id_.c_str(),
      translation_param[0], translation_param[1], translation_param[2],
      rotation_param[0], rotation_param[1], rotation_param[2], rotation_param[3]);

  try {
    franka_state_publisher = get_node()->create_publisher<franka_msgs::msg::FrankaState>(
        "~/" + state_interface_name, rclcpp::SystemDefaultsQoS());
    realtime_franka_state_publisher =
        std::make_shared<realtime_tools::RealtimePublisher<franka_msgs::msg::FrankaState>>(
            franka_state_publisher);
    const std::string end_effector_pose_topic =
        "/" + arm_id + "/arm/end_effector_pose";
    end_effector_pose_publisher =
        get_node()->create_publisher<geometry_msgs::msg::PoseStamped>(
            end_effector_pose_topic, rclcpp::SystemDefaultsQoS());
    realtime_end_effector_pose_publisher =
        std::make_shared<
            realtime_tools::RealtimePublisher<geometry_msgs::msg::PoseStamped>>(
            end_effector_pose_publisher);
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

    if (realtime_end_effector_pose_publisher &&
        realtime_end_effector_pose_publisher->trylock()) {
      auto& pose_msg = realtime_end_effector_pose_publisher->msg_;
      pose_msg.header.stamp = time;
      pose_msg.header.frame_id = world_frame_id_;
      pose_msg.pose.position.x = realtime_franka_state_publisher->msg_.o_t_ee[12];
      pose_msg.pose.position.y = realtime_franka_state_publisher->msg_.o_t_ee[13];
      pose_msg.pose.position.z = realtime_franka_state_publisher->msg_.o_t_ee[14];
      const auto quat_wxyz =
          quaternionFromColumnMajorRotation(realtime_franka_state_publisher->msg_.o_t_ee);
      pose_msg.pose.orientation.w = quat_wxyz[0];
      pose_msg.pose.orientation.x = quat_wxyz[1];
      pose_msg.pose.orientation.y = quat_wxyz[2];
      pose_msg.pose.orientation.z = quat_wxyz[3];
      realtime_end_effector_pose_publisher->unlockAndPublish();
    }

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
