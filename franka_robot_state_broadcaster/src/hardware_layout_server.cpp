#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

namespace {

constexpr char kWorldFrameIdParam[] = "world_frame_id";
constexpr char kWorldToFrankaBasePrefix[] = "world_to_franka_base.";
constexpr char kTranslationSuffix[] = ".translation";
constexpr char kRotationSuffix[] = ".rotation_xyzw";
constexpr char kFrankaBaseFrameSuffix[] = "_link0";
constexpr double kQuaternionNormEpsilon = 1e-9;

bool hasSuffix(const std::string& value, const std::string& suffix) {
  if (value.size() < suffix.size()) {
    return false;
  }
  return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> discoverArmIds(const rclcpp::Node::SharedPtr& node) {
  std::unordered_set<std::string> arm_ids;
  const auto parameters = node->list_parameters({"world_to_franka_base"}, 10);

  for (const auto& name : parameters.names) {
    if (name.compare(0, std::strlen(kWorldToFrankaBasePrefix), kWorldToFrankaBasePrefix) != 0) {
      continue;
    }
    if (!hasSuffix(name, kTranslationSuffix) && !hasSuffix(name, kRotationSuffix)) {
      continue;
    }

    const auto first_dot_after_prefix =
        name.find('.', std::strlen(kWorldToFrankaBasePrefix));
    if (first_dot_after_prefix == std::string::npos) {
      continue;
    }
    const auto arm_id = name.substr(
        std::strlen(kWorldToFrankaBasePrefix),
        first_dot_after_prefix - std::strlen(kWorldToFrankaBasePrefix));
    if (!arm_id.empty()) {
      arm_ids.insert(arm_id);
    }
  }

  std::vector<std::string> sorted_arm_ids(arm_ids.begin(), arm_ids.end());
  std::sort(sorted_arm_ids.begin(), sorted_arm_ids.end());
  return sorted_arm_ids;
}

bool loadTransformForArm(
    const rclcpp::Node::SharedPtr& node,
    const std::string& world_frame_id,
    const std::string& arm_id,
    geometry_msgs::msg::TransformStamped& transform) {
  const std::string translation_param =
      std::string(kWorldToFrankaBasePrefix) + arm_id + kTranslationSuffix;
  const std::string rotation_param =
      std::string(kWorldToFrankaBasePrefix) + arm_id + kRotationSuffix;

  rclcpp::Parameter translation_value;
  rclcpp::Parameter rotation_value;
  const bool has_translation = node->get_parameter(translation_param, translation_value) &&
                               translation_value.get_type() !=
                                   rclcpp::ParameterType::PARAMETER_NOT_SET;
  const bool has_rotation = node->get_parameter(rotation_param, rotation_value) &&
                            rotation_value.get_type() !=
                                rclcpp::ParameterType::PARAMETER_NOT_SET;

  if (!has_translation || !has_rotation) {
    RCLCPP_WARN(
        node->get_logger(),
        "Skipping '%s': expected both '%s' and '%s'.",
        arm_id.c_str(), translation_param.c_str(), rotation_param.c_str());
    return false;
  }

  if (translation_value.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
    RCLCPP_WARN(
        node->get_logger(),
        "Skipping '%s': '%s' must be a double array [x, y, z].",
        arm_id.c_str(), translation_param.c_str());
    return false;
  }
  if (rotation_value.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
    RCLCPP_WARN(
        node->get_logger(),
        "Skipping '%s': '%s' must be a double array [x, y, z, w].",
        arm_id.c_str(), rotation_param.c_str());
    return false;
  }

  const auto translation = translation_value.as_double_array();
  const auto rotation_xyzw = rotation_value.as_double_array();
  if (translation.size() != 3) {
    RCLCPP_WARN(
        node->get_logger(),
        "Skipping '%s': '%s' must contain exactly 3 values.",
        arm_id.c_str(), translation_param.c_str());
    return false;
  }
  if (rotation_xyzw.size() != 4) {
    RCLCPP_WARN(
        node->get_logger(),
        "Skipping '%s': '%s' must contain exactly 4 values.",
        arm_id.c_str(), rotation_param.c_str());
    return false;
  }

  const double norm = std::sqrt(
      rotation_xyzw[0] * rotation_xyzw[0] +
      rotation_xyzw[1] * rotation_xyzw[1] +
      rotation_xyzw[2] * rotation_xyzw[2] +
      rotation_xyzw[3] * rotation_xyzw[3]);
  if (norm < kQuaternionNormEpsilon) {
    RCLCPP_WARN(
        node->get_logger(),
        "Skipping '%s': '%s' quaternion norm is zero.",
        arm_id.c_str(), rotation_param.c_str());
    return false;
  }

  transform.header.stamp = node->get_clock()->now();
  transform.header.frame_id = world_frame_id;
  transform.child_frame_id = arm_id + kFrankaBaseFrameSuffix;
  transform.transform.translation.x = translation[0];
  transform.transform.translation.y = translation[1];
  transform.transform.translation.z = translation[2];
  transform.transform.rotation.x = rotation_xyzw[0] / norm;
  transform.transform.rotation.y = rotation_xyzw[1] / norm;
  transform.transform.rotation.z = rotation_xyzw[2] / norm;
  transform.transform.rotation.w = rotation_xyzw[3] / norm;

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<rclcpp::Node>("hardware_layout", options);
  auto static_tf_broadcaster =
      std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);

  std::string world_frame_id = "world";
  node->get_parameter(kWorldFrameIdParam, world_frame_id);

  const auto arm_ids = discoverArmIds(node);
  if (arm_ids.empty()) {
    RCLCPP_WARN(
        node->get_logger(),
        "No '%s<arm_id>.(translation|rotation_xyzw)' parameters found. "
        "No world-to-franka-base TFs will be published.",
        kWorldToFrankaBasePrefix);
  } else {
    std::vector<geometry_msgs::msg::TransformStamped> transforms;
    transforms.reserve(arm_ids.size());
    for (const auto& arm_id : arm_ids) {
      geometry_msgs::msg::TransformStamped transform;
      if (!loadTransformForArm(node, world_frame_id, arm_id, transform)) {
        continue;
      }
      RCLCPP_INFO(
          node->get_logger(),
          "Publishing TF '%s' -> '%s' from /hardware_layout.",
          transform.header.frame_id.c_str(), transform.child_frame_id.c_str());
      transforms.push_back(transform);
    }
    if (transforms.empty()) {
      RCLCPP_WARN(
          node->get_logger(),
          "No valid world-to-franka-base transforms were found.");
    } else {
      static_tf_broadcaster->sendTransform(transforms);
      RCLCPP_INFO(
          node->get_logger(),
          "Published %zu static world-to-franka-base TF transform(s).",
          transforms.size());
    }
  }

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
