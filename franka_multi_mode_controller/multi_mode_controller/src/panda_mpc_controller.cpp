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
  return true;
}

void Controller::startROSComImpl() {
  const std::string ns = "/" + arm_ids_.at(0) + "/arm/";
  waypoint_sequence_sub_ = NodeBase::node_->create_subscription<PoseArray>(
      ns + "mpc_end_effector_pose_cmd", 10,
      [this](const PoseArray& msg) { this->waypointSequenceCallback(msg); });
  clear_waypoints_srv_ = NodeBase::node_->create_service<Trigger>(
      ns + "mpc_clear_waypoints",
      [this](const std::shared_ptr<Trigger::Request> req,
             const std::shared_ptr<Trigger::Response> res) {
        this->clearWaypointsCallback(req, res);
      });
}

void Controller::stopROSComImpl() {
  waypoint_sequence_sub_.reset();
  clear_waypoints_srv_.reset();
  clearWaypoints();
}

void Controller::waypointSequenceCallback(const PoseArray& msg) {
  std::vector<std::pair<Eigen::Vector3d, Eigen::Quaterniond>> base_poses;
  base_poses.reserve(msg.poses.size());
  for (const auto& pose : msg.poses) {
    const Eigen::Vector3d world_position(pose.position.x, pose.position.y,
                                         pose.position.z);
    const Eigen::Quaterniond world_orientation(
        pose.orientation.w, pose.orientation.x, pose.orientation.y,
        pose.orientation.z);
    Eigen::Vector3d base_position;
    Eigen::Quaterniond base_orientation;
    if (!transformWorldPoseToFrankaBase(world_position, world_orientation,
                                        world_to_franka_base_.at(0),
                                        base_position, base_orientation)) {
      RCLCPP_WARN_THROTTLE(
          NodeBase::node_->get_logger(), *NodeBase::node_->get_clock(), 1000,
          "panda_mpc_controller: skipping waypoint with invalid orientation.");
      continue;
    }
    base_poses.emplace_back(base_position, base_orientation);
  }
  if (!base_poses.empty()) {
    appendWaypoints(base_poses);
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
  const Eigen::Vector3d world_position(msg.pose.position.x, msg.pose.position.y,
                                       msg.pose.position.z);
  const Eigen::Quaterniond world_orientation(
      msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y,
      msg.pose.orientation.z);
  Eigen::Vector3d base_position;
  Eigen::Quaterniond base_orientation;
  if (!transformWorldPoseToFrankaBase(world_position, world_orientation,
                                      world_to_franka_base_.at(0),
                                      base_position, base_orientation)) {
    RCLCPP_WARN_THROTTLE(
        NodeBase::node_->get_logger(), *NodeBase::node_->get_clock(), 1000,
        "panda_mpc_controller: discarding desired pose with invalid orientation.");
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
  res->success = true;
  return true;
}
