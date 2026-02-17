#include <multi_mode_controller/controllers/dual_cartesian_impedance_controller.h>

#include <algorithm>
#include <chrono>
#include <limits>

#include <multi_mode_controller/utils/controller_factory.h>
#include <multi_mode_controller/utils/world_frame_transforms.h>

using namespace panda_controllers;
using Eigen::Vector3d;
using Eigen::Quaterniond;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Pose = DualCartesianImpedanceControllerPose;
using Parameters = DualCartesianImpedanceControllerParams;
using GoalMsg = multi_mode_control_msgs::msg::DualCartesianImpedanceGoal;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using ConfigRequest = multi_mode_control_msgs::srv::SetCartesianImpedance::Request;
using ConfigResponse = multi_mode_control_msgs::srv::SetCartesianImpedance::Response;
using Controller = DualCartesianImpedanceController;

namespace {
constexpr double kMaxPositionStep = 0.1;
constexpr double kMaxOrientationStep = 0.15;
constexpr auto kWaypointUpdatePeriod = std::chrono::milliseconds(10);

geometry_msgs::msg::Pose toWorldPose(
    const PandaCartesianImpedanceControllerPose& pose,
    const Eigen::Affine3d& world_to_franka_base) {
  Eigen::Quaterniond franka_orientation = pose.orientation;
  if (franka_orientation.norm() < 1e-9) {
    franka_orientation = Eigen::Quaterniond::Identity();
  } else {
    franka_orientation.normalize();
  }

  Eigen::Affine3d franka_base_to_ee = Eigen::Affine3d::Identity();
  franka_base_to_ee.translation() = pose.position;
  franka_base_to_ee.linear() = franka_orientation.toRotationMatrix();

  const Eigen::Affine3d world_to_ee = world_to_franka_base * franka_base_to_ee;
  Eigen::Quaterniond world_orientation(world_to_ee.linear());
  world_orientation.normalize();

  geometry_msgs::msg::Pose world_pose;
  world_pose.position.x = world_to_ee.translation().x();
  world_pose.position.y = world_to_ee.translation().y();
  world_pose.position.z = world_to_ee.translation().z();
  world_pose.orientation.w = world_orientation.w();
  world_pose.orientation.x = world_orientation.x();
  world_pose.orientation.y = world_orientation.y();
  world_pose.orientation.z = world_orientation.z();
  return world_pose;
}

Quaterniond normalizedQuaternion(const Quaterniond& q) {
  if (q.norm() < std::numeric_limits<double>::epsilon()) {
    return Quaterniond::Identity();
  }
  Quaterniond normalized = q;
  normalized.normalize();
  return normalized;
}

PandaCartesianImpedanceControllerPose computeWaypointPose(
    const PandaCartesianImpedanceControllerPose& current_raw,
    const PandaCartesianImpedanceControllerPose& goal,
    const PandaCartesianImpedanceControllerPose& offset,
    bool& reached_goal) {
  PandaCartesianImpedanceControllerPose waypoint = goal;
  const Vector3d current_position = current_raw.position - offset.position;
  const Vector3d delta_position = goal.position - current_position;
  const double position_distance = delta_position.norm();

  const Quaterniond current_orientation = normalizedQuaternion(current_raw.orientation);
  const Quaterniond goal_orientation = normalizedQuaternion(goal.orientation);
  const double orientation_distance =
      current_orientation.angularDistance(goal_orientation);

  double step_scale = 1.0;
  if (position_distance > std::numeric_limits<double>::epsilon()) {
    step_scale = std::min(step_scale, kMaxPositionStep / position_distance);
  }
  if (orientation_distance > std::numeric_limits<double>::epsilon()) {
    step_scale = std::min(step_scale, kMaxOrientationStep / orientation_distance);
  }
  step_scale = std::clamp(step_scale, 0.0, 1.0);

  if (step_scale >= 1.0) {
    waypoint.position = goal.position;
    waypoint.orientation = goal_orientation;
    reached_goal = true;
  } else {
    waypoint.position = current_position + step_scale * delta_position;
    waypoint.orientation = current_orientation.slerp(step_scale, goal_orientation);
    waypoint.orientation.normalize();
    reached_goal = false;
  }
  waypoint.q_n = goal.q_n;
  return waypoint;
}
}  // namespace


static auto registration = ControllerFactory::registerClass<Controller>(
    "dual_cartesian_impedance_controller");

bool Controller::initImpl(const std::vector<RobotData*>& /*robot_data*/,
                          rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
                          std::string name,
                          std::string resource) {
  return loadWorldToFrankaBaseTransformsForResource(
      node, name, resource, 2, arm_ids_, world_to_franka_base_);
}

void Controller::startROSComImpl() {
  end_effector_pose_cmd_subs_.clear();
  end_effector_pose_cmd_subs_.reserve(arm_ids_.size());
  for (std::size_t i = 0; i < arm_ids_.size(); ++i) {
    const std::string topic_name = "/" + arm_ids_.at(i) + "/arm/end_effector_pose_cmd";
    end_effector_pose_cmd_subs_.push_back(
        PandaControllerBase<Parameters, Pose>::node_->create_subscription<PoseStamped>(
            topic_name, 10, [this, i](const PoseStamped& msg) {
              this->endEffectorPoseCmdCallback(i, msg);
            }));
  }
  waypoint_timer_ = PandaControllerBase<Parameters, Pose>::node_->create_wall_timer(
      kWaypointUpdatePeriod, [this]() { this->updateWaypointTowardsGoal(); });
}

void Controller::stopROSComImpl() {
  waypoint_timer_.reset();
  end_effector_pose_cmd_subs_.clear();
  std::lock_guard<std::mutex> lock(goal_mutex_);
  has_goal_ = false;
}

void Controller::endEffectorPoseCmdCallback(std::size_t arm_index,
                                            const PoseStamped& msg) {
  if (arm_index >= world_to_franka_base_.size() ||
      world_to_franka_base_.size() < 2) {
    return;
  }

  const Pose desired_pose = this->getDesiredPoseBuffered();
  GoalMsg command_msg;
  command_msg.l_pose = toWorldPose(desired_pose.poses[0], world_to_franka_base_.at(0));
  command_msg.r_pose = toWorldPose(desired_pose.poses[1], world_to_franka_base_.at(1));
  for (std::size_t i = 0; i < 7; ++i) {
    command_msg.l_q_n[i] = desired_pose.poses[0].q_n(i);
    command_msg.r_q_n[i] = desired_pose.poses[1].q_n(i);
  }

  if (arm_index == 0) {
    command_msg.l_pose = msg.pose;
  } else {
    command_msg.r_pose = msg.pose;
  }

  Pose p_d;
  if (desiredPoseCallbackImpl(p_d, this->getCurrentPose(), command_msg)) {
    this->setDesiredPoseBuffered(p_d);
  } else {
    this->setError(true);
  }
}

bool Controller::desiredPoseCallbackImpl(Pose& p_d, 
                                         const Pose& p,
                                         const GoalMsg& msg) {
  Pose goal_pose;
  const Vector3d left_world_position(msg.l_pose.position.x,
                                     msg.l_pose.position.y,
                                     msg.l_pose.position.z);
  const Quaterniond left_world_orientation(msg.l_pose.orientation.w,
                                           msg.l_pose.orientation.x,
                                           msg.l_pose.orientation.y,
                                           msg.l_pose.orientation.z);
  const Vector3d right_world_position(msg.r_pose.position.x,
                                      msg.r_pose.position.y,
                                      msg.r_pose.position.z);
  const Quaterniond right_world_orientation(msg.r_pose.orientation.w,
                                            msg.r_pose.orientation.x,
                                            msg.r_pose.orientation.y,
                                            msg.r_pose.orientation.z);

  if (!transformWorldPoseToFrankaBase(left_world_position,
                                      left_world_orientation,
                                      world_to_franka_base_.at(0),
                                      goal_pose.poses[0].position,
                                      goal_pose.poses[0].orientation) ||
      !transformWorldPoseToFrankaBase(right_world_position,
                                      right_world_orientation,
                                      world_to_franka_base_.at(1),
                                      goal_pose.poses[1].position,
                                      goal_pose.poses[1].orientation)) {
    auto& clk = *PandaControllerBase<Parameters, Pose>::node_->get_clock();
    const auto& logger = PandaControllerBase<Parameters, Pose>::node_->get_logger();
    RCLCPP_WARN_THROTTLE(
        logger, clk, 1000,
        "dual_cartesian_impedance_controller: Discarding target pose with invalid orientation quaternion.");
    return false;
  }

  goal_pose.poses[0].q_n = Eigen::Map<const Vector7d>(msg.l_q_n.data());
  goal_pose.poses[1].q_n = Eigen::Map<const Vector7d>(msg.r_q_n.data());

  const Pose offset = PandaControllerBase<Parameters, Pose>::getOffset();
  bool left_reached_goal = false;
  bool right_reached_goal = false;
  p_d.poses[0] =
      computeWaypointPose(p.poses[0], goal_pose.poses[0], offset.poses[0], left_reached_goal);
  p_d.poses[1] =
      computeWaypointPose(p.poses[1], goal_pose.poses[1], offset.poses[1], right_reached_goal);

  {
    std::lock_guard<std::mutex> lock(goal_mutex_);
    goal_pose_ = goal_pose;
    has_goal_ = !(left_reached_goal && right_reached_goal);
  }

  if (!(left_reached_goal && right_reached_goal)) {
    auto& clk = *PandaControllerBase<Parameters, Pose>::node_->get_clock();
    const auto& logger = PandaControllerBase<Parameters, Pose>::node_->get_logger();
    RCLCPP_WARN_THROTTLE(
        logger, clk, 1000,
        "dual_cartesian_impedance_controller: Target pose exceeds step limits, tracking intermediate waypoint.");
  }
  return true;
}

void Controller::updateWaypointTowardsGoal() {
  std::lock_guard<std::mutex> lock(goal_mutex_);
  if (!has_goal_) {
    return;
  }

  const Pose current_pose = this->getCurrentPose();
  const Pose offset = PandaControllerBase<Parameters, Pose>::getOffset();
  Pose waypoint;
  bool left_reached_goal = false;
  bool right_reached_goal = false;
  waypoint.poses[0] = computeWaypointPose(current_pose.poses[0],
                                          goal_pose_.poses[0],
                                          offset.poses[0],
                                          left_reached_goal);
  waypoint.poses[1] = computeWaypointPose(current_pose.poses[1],
                                          goal_pose_.poses[1],
                                          offset.poses[1],
                                          right_reached_goal);
  this->setDesiredPoseBuffered(waypoint);
  has_goal_ = !(left_reached_goal && right_reached_goal);
}

bool Controller::setParametersCallbackImpl(Parameters& p_d, const Parameters& p,
            const std::shared_ptr<ConfigRequest>& request, const std::shared_ptr<ConfigResponse>& response) {
  for(int i=0; i<2; i++){
    p_d.params[i].stiffness = Matrix6d(request->stiffness.data());
    p_d.params[i].damping_ratio = Vector6d(request->damping_ratio.data());
    p_d.params[i].nullspace_stiffness = request->nullspace_stiffness;
  }
  return true;
}
