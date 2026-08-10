#include <multi_mode_controller/controllers/panda_mpc_controller.h>

#include <memory>
#include <utility>

#include <multi_mode_controller/utils/controller_factory.h>
#include <multi_mode_controller/utils/world_frame_transforms.h>

using namespace panda_controllers;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Pose = PandaMpcControllerPose;
using Params = PandaMpcControllerParams;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using PoseArray = geometry_msgs::msg::PoseArray;
using Trigger = std_srvs::srv::Trigger;
using Controller = PandaMpcController;
using NodeBase = PandaControllerBase<PandaMpcControllerParams,
                                     PandaMpcControllerPose>;

static auto registration =
    ControllerFactory::registerClass<Controller>("panda_mpc_controller");

bool Controller::initImpl(const std::vector<RobotData*>& /*robot_data*/,
                          rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
                          std::string name,
                          std::string resource) {
  if (!loadWorldToFrankaBaseTransformsForResource(
          node, name, resource, 1, arm_ids_, world_to_franka_base_)) {
    return false;
  }
  // Optional spacing between consecutive commanded waypoints (s).
  const std::string command_dt_param = name + ".command_dt";
  try {
    if (node->has_parameter(command_dt_param)) {
      setCommandDt(node->get_parameter(command_dt_param).as_double());
    }
  } catch (const std::exception& e) {
    RCLCPP_WARN(node->get_logger(), "%s: failed to read '%s': %s", name.c_str(),
                command_dt_param.c_str(), e.what());
  }
  // Optional nominal posture for the QP's redundancy regularization
  // (defaults to the Panda ready pose).
  const std::string q_nominal_param = name + ".q_nominal";
  try {
    if (node->has_parameter(q_nominal_param)) {
      const auto q_nominal = node->get_parameter(q_nominal_param).as_double_array();
      if (q_nominal.size() == 7) {
        setNominalPosture(Eigen::Map<const Vector7d>(q_nominal.data()));
      } else {
        RCLCPP_WARN(node->get_logger(), "%s: '%s' needs 7 values, got %zu.",
                    name.c_str(), q_nominal_param.c_str(), q_nominal.size());
      }
    }
  } catch (const std::exception& e) {
    RCLCPP_WARN(node->get_logger(), "%s: failed to read '%s': %s", name.c_str(),
                q_nominal_param.c_str(), e.what());
  }
  // Optional reference-model overrides for the single-pose command path
  // (see MpcReferenceModel). Defaults are fitted to the measured
  // panda_cartesian_impedance_controller response.
  MpcReferenceModel ref_model;
  const std::string prefix = name + ".reference_model.";
  const auto read_double = [&](const char* key, double& target) {
    const std::string p = prefix + key;
    try {
      if (node->has_parameter(p)) {
        target = node->get_parameter(p).as_double();
      }
    } catch (const std::exception& e) {
      RCLCPP_WARN(node->get_logger(), "%s: failed to read '%s': %s",
                  name.c_str(), p.c_str(), e.what());
    }
  };
  try {
    const std::string p = prefix + "enabled";
    if (node->has_parameter(p)) {
      ref_model.enabled = node->get_parameter(p).as_bool();
    }
  } catch (const std::exception& e) {
    RCLCPP_WARN(node->get_logger(), "%s: failed to read '%senabled': %s",
                name.c_str(), prefix.c_str(), e.what());
  }
  read_double("filter_tau", ref_model.filter_tau);
  read_double("omega_n", ref_model.omega_n);
  read_double("zeta", ref_model.zeta);
  read_double("max_velocity", ref_model.max_velocity);
  read_double("max_accel", ref_model.max_accel);
  read_double("omega_n_rot", ref_model.omega_n_rot);
  read_double("zeta_rot", ref_model.zeta_rot);
  read_double("max_velocity_rot", ref_model.max_velocity_rot);
  read_double("max_accel_rot", ref_model.max_accel_rot);
  setReferenceModel(ref_model);
  RCLCPP_INFO(node->get_logger(),
              "%s: reference model %s (wn %.2f, zeta %.2f, tau %.3f s)",
              name.c_str(), ref_model.enabled ? "ON" : "OFF",
              ref_model.omega_n, ref_model.zeta, ref_model.filter_tau);
  return true;
}

void Controller::startROSComImpl() {
  const std::string ns = "/" + arm_ids_.at(0) + "/arm/";
  waypoint_sequence_sub_ = NodeBase::node_->create_subscription<PoseArray>(
      ns + "mpc_end_effector_pose_cmd", 10,
      [this](const PoseArray& msg) { this->waypointSequenceCallback(msg); });
  end_effector_pose_cmd_sub_ = NodeBase::node_->create_subscription<PoseStamped>(
      ns + "end_effector_pose_cmd", 10,
      [this](const PoseStamped& msg) { this->endEffectorPoseCmdCallback(msg); });
  clear_waypoints_srv_ = NodeBase::node_->create_service<Trigger>(
      ns + "mpc_clear_waypoints",
      [this](const std::shared_ptr<Trigger::Request> req,
             const std::shared_ptr<Trigger::Response> res) {
        this->clearWaypointsCallback(req, res);
      });
}

void Controller::stopROSComImpl() {
  waypoint_sequence_sub_.reset();
  end_effector_pose_cmd_sub_.reset();
  clear_waypoints_srv_.reset();
  clearWaypoints();
}

bool Controller::transformWorldPose(const geometry_msgs::msg::Pose& pose,
                                    Eigen::Vector3d& base_position,
                                    Eigen::Quaterniond& base_orientation) {
  const Eigen::Vector3d world_position(pose.position.x, pose.position.y,
                                       pose.position.z);
  const Eigen::Quaterniond world_orientation(
      pose.orientation.w, pose.orientation.x, pose.orientation.y,
      pose.orientation.z);
  if (!transformWorldPoseToFrankaBase(world_position, world_orientation,
                                      world_to_franka_base_.at(0),
                                      base_position, base_orientation)) {
    RCLCPP_WARN_THROTTLE(
        NodeBase::node_->get_logger(), *NodeBase::node_->get_clock(), 1000,
        "panda_mpc_controller: skipping pose with invalid orientation.");
    return false;
  }
  return true;
}

void Controller::waypointSequenceCallback(const PoseArray& msg) {
  std::vector<std::pair<Eigen::Vector3d, Eigen::Quaterniond>> base_poses;
  base_poses.reserve(msg.poses.size());
  for (const auto& pose : msg.poses) {
    Eigen::Vector3d base_position;
    Eigen::Quaterniond base_orientation;
    if (transformWorldPose(pose, base_position, base_orientation)) {
      base_poses.emplace_back(base_position, base_orientation);
    }
  }
  if (!base_poses.empty()) {
    appendWaypoints(base_poses);
  }
}

void Controller::endEffectorPoseCmdCallback(const PoseStamped& msg) {
  // Single pose: replace the trajectory rather than extend it.
  Eigen::Vector3d base_position;
  Eigen::Quaterniond base_orientation;
  if (transformWorldPose(msg.pose, base_position, base_orientation)) {
    setImmediateTarget(base_position, base_orientation);
  }
}

void Controller::clearWaypointsCallback(
    const std::shared_ptr<Trigger::Request> /*req*/,
    const std::shared_ptr<Trigger::Response> res) {
  clearWaypoints();
  res->success = true;
  res->message = "panda_mpc_controller: cleared unexecuted waypoints.";
}

bool Controller::desiredPoseCallbackImpl(Pose& p_d, const Pose& /*p*/,
                                         const MpcDesPoseMsg& msg) {
  Eigen::Vector3d base_position;
  Eigen::Quaterniond base_orientation;
  if (!transformWorldPose(msg.pose, base_position, base_orientation)) {
    return false;
  }
  appendWaypoints({{base_position, base_orientation}});
  p_d = getCurrentPose();
  p_d.position = base_position;
  p_d.orientation = base_orientation;
  return true;
}

bool Controller::setParametersCallbackImpl(
    Params& p_d, const Params& p,
    const MpcServiceParameter::Request::SharedPtr& req,
    const MpcServiceParameter::Response::SharedPtr& res) {
  p_d = p;
  p_d.kp = Eigen::Map<const Vector7d>(req->kp.data());
  p_d.kd = Eigen::Map<const Vector7d>(req->kd.data());
  p_d.task_weight = Eigen::Map<const Vector6d>(req->task_weight.data());
  p_d.input_weight = req->input_weight;
  p_d.posture_weight = req->posture_weight;
  p_d.velocity_weight = req->velocity_weight;
  p_d.accel_weight = req->accel_weight;
  res->success = true;
  return true;
}
