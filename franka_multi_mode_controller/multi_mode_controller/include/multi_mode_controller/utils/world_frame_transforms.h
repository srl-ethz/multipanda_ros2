#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <rclcpp/parameter_client.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <multi_mode_controller/utils/conversions.h>

namespace panda_controllers {

inline bool loadDoubleArrayParameter(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
    const std::string& parameter_name,
    bool& has_parameter,
    std::vector<double>& values) {
  has_parameter = node->has_parameter(parameter_name);
  if (!has_parameter) {
    return true;
  }
  const auto parameter = node->get_parameter(parameter_name);
  if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
    RCLCPP_ERROR(node->get_logger(),
                 "Parameter '%s' must be a double array.",
                 parameter_name.c_str());
    return false;
  }
  values = parameter.as_double_array();
  return true;
}

inline bool loadDoubleArrayParameterFromHardwareLayout(
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
    if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
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

inline bool loadWorldToFrankaBaseTransform(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
    const std::string& controller_name,
    const std::string& arm_id,
    const std::shared_ptr<rclcpp::SyncParametersClient>& hardware_layout_client,
    Eigen::Affine3d& world_to_franka_base) {
  const std::string translation_parameter =
      "world_to_franka_base." + arm_id + ".translation";
  const std::string rotation_parameter =
      "world_to_franka_base." + arm_id + ".rotation_xyzw";

  std::vector<double> translation;
  std::vector<double> rotation_xyzw;
  bool has_translation = false;
  bool has_rotation = false;

  if (!loadDoubleArrayParameterFromHardwareLayout(
          node, hardware_layout_client, translation_parameter, has_translation,
          translation) ||
      !loadDoubleArrayParameterFromHardwareLayout(
          node, hardware_layout_client, rotation_parameter, has_rotation,
          rotation_xyzw)) {
    return false;
  }
  const bool using_global_transform = has_translation && has_rotation;

  if (!has_translation && !has_rotation) {
    if (hardware_layout_client) {
      RCLCPP_WARN(
          node->get_logger(),
          "%s: '/hardware_layout.%s' and '/hardware_layout.%s' were not found for arm '%s'. "
          "Falling back to local controller parameters.",
          controller_name.c_str(), translation_parameter.c_str(),
          rotation_parameter.c_str(), arm_id.c_str());
    }
    if (!loadDoubleArrayParameter(node, translation_parameter, has_translation,
                                  translation) ||
        !loadDoubleArrayParameter(node, rotation_parameter, has_rotation,
                                  rotation_xyzw)) {
      return false;
    }
  } else if (!has_translation || !has_rotation) {
    RCLCPP_ERROR(
        node->get_logger(),
        "%s: both '/hardware_layout.%s' and '/hardware_layout.%s' must be provided.",
        controller_name.c_str(), translation_parameter.c_str(),
        rotation_parameter.c_str());
    return false;
  }

  if (!has_translation && !has_rotation) {
    world_to_franka_base.setIdentity();
    RCLCPP_INFO(
        node->get_logger(),
        "%s: world_to_franka_base for '%s' is not configured. Using identity transform.",
        controller_name.c_str(), arm_id.c_str());
    return true;
  }

  if (!has_translation || !has_rotation) {
    RCLCPP_ERROR(
        node->get_logger(),
        "%s: both '%s' and '%s' must be provided when configuring world_to_franka_base for '%s'.",
        controller_name.c_str(), translation_parameter.c_str(),
        rotation_parameter.c_str(), arm_id.c_str());
    return false;
  }

  if (translation.size() != 3) {
    RCLCPP_ERROR(node->get_logger(),
                 "%s: '%s' must contain exactly 3 elements [x, y, z].",
                 controller_name.c_str(), translation_parameter.c_str());
    return false;
  }
  if (rotation_xyzw.size() != 4) {
    RCLCPP_ERROR(
        node->get_logger(),
        "%s: '%s' must contain exactly 4 elements [x, y, z, w].",
        controller_name.c_str(), rotation_parameter.c_str());
    return false;
  }

  Eigen::Quaterniond rotation_wxyz(rotation_xyzw[3], rotation_xyzw[0],
                                   rotation_xyzw[1], rotation_xyzw[2]);
  if (rotation_wxyz.norm() < 1e-9) {
    RCLCPP_ERROR(
        node->get_logger(),
        "%s: '%s' contains an invalid zero-norm quaternion.",
        controller_name.c_str(), rotation_parameter.c_str());
    return false;
  }
  rotation_wxyz.normalize();

  world_to_franka_base.setIdentity();
  world_to_franka_base.translation() =
      Eigen::Vector3d(translation[0], translation[1], translation[2]);
  world_to_franka_base.linear() = rotation_wxyz.toRotationMatrix();
  RCLCPP_INFO(
      node->get_logger(),
      "%s: world_to_franka_base for '%s' source=%s translation=[%.6f, %.6f, %.6f] "
      "rotation_xyzw=[%.6f, %.6f, %.6f, %.6f]",
      controller_name.c_str(), arm_id.c_str(),
      using_global_transform ? "/hardware_layout" : "local",
      translation[0], translation[1], translation[2], rotation_xyzw[0],
      rotation_xyzw[1], rotation_xyzw[2], rotation_xyzw[3]);
  return true;
}

inline bool loadWorldToFrankaBaseTransformsForResource(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
    const std::string& controller_name,
    const std::string& resource,
    const std::size_t expected_arm_count,
    std::vector<std::string>& arm_ids,
    std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>&
        world_to_franka_base) {
  int hardware_layout_wait_ms = 5000;
  if (node->has_parameter("hardware_layout_wait_ms")) {
    hardware_layout_wait_ms =
        node->get_parameter("hardware_layout_wait_ms").as_int();
  }
  if (hardware_layout_wait_ms < 0) {
    RCLCPP_WARN(
        node->get_logger(),
        "%s: hardware_layout_wait_ms (%d) cannot be negative. Clamping to 0 ms.",
        controller_name.c_str(), hardware_layout_wait_ms);
    hardware_layout_wait_ms = 0;
  }

  rclcpp::NodeOptions hardware_layout_client_node_options;
  hardware_layout_client_node_options.context(
      node->get_node_base_interface()->get_context());
  hardware_layout_client_node_options.use_global_arguments(false);
  hardware_layout_client_node_options.start_parameter_services(false);
  hardware_layout_client_node_options.start_parameter_event_publisher(false);
  const auto hardware_layout_client_node =
      std::make_shared<rclcpp::Node>(
          std::string(node->get_name()) + "_hardware_layout_client",
          hardware_layout_client_node_options);
  auto hardware_layout_client =
      std::make_shared<rclcpp::SyncParametersClient>(
          hardware_layout_client_node, "/hardware_layout");
  if (!hardware_layout_client->wait_for_service(
          std::chrono::milliseconds(hardware_layout_wait_ms))) {
    RCLCPP_WARN(
        node->get_logger(),
        "%s: '/hardware_layout/get_parameters' not available after %d ms. "
        "Global hardware layout overrides will be skipped.",
        controller_name.c_str(), hardware_layout_wait_ms);
    hardware_layout_client.reset();
  }

  arm_ids = resourceToVector(resource);
  if (arm_ids.size() != expected_arm_count) {
    RCLCPP_ERROR(
        node->get_logger(),
        "%s: resource '%s' resolves to %zu arm(s), expected %zu.",
        controller_name.c_str(), resource.c_str(), arm_ids.size(),
        expected_arm_count);
    return false;
  }

  world_to_franka_base.clear();
  world_to_franka_base.reserve(arm_ids.size());
  for (const auto& arm_id : arm_ids) {
    Eigen::Affine3d transform;
    if (!loadWorldToFrankaBaseTransform(node, controller_name, arm_id,
                                        hardware_layout_client, transform)) {
      return false;
    }
    world_to_franka_base.push_back(transform);
  }
  return true;
}

inline bool transformWorldPoseToFrankaBase(
    const Eigen::Vector3d& world_position,
    const Eigen::Quaterniond& world_orientation,
    const Eigen::Affine3d& world_to_franka_base,
    Eigen::Vector3d& franka_position,
    Eigen::Quaterniond& franka_orientation) {
  if (world_orientation.norm() < 1e-9) {
    return false;
  }

  Eigen::Quaterniond world_orientation_normalized = world_orientation;
  world_orientation_normalized.normalize();

  Eigen::Affine3d world_to_ee = Eigen::Affine3d::Identity();
  world_to_ee.translation() = world_position;
  world_to_ee.linear() = world_orientation_normalized.toRotationMatrix();

  const Eigen::Affine3d franka_base_to_ee =
      world_to_franka_base.inverse() * world_to_ee;
  franka_position = franka_base_to_ee.translation();
  franka_orientation = Eigen::Quaterniond(franka_base_to_ee.linear());
  franka_orientation.normalize();
  return true;
}

}  // namespace panda_controllers
