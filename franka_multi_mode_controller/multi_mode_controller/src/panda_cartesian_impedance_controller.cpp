#include <multi_mode_controller/controllers/panda_cartesian_impedance_controller.h>

#include <limits>

#include <multi_mode_controller/utils/controller_factory.h>
#include <multi_mode_controller/utils/world_frame_transforms.h>

using namespace panda_controllers;
using Eigen::Vector3d;
using Eigen::Quaterniond;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Pose = PandaCartesianImpedanceControllerPose;
using Parameters = PandaCartesianImpedanceControllerParams;
using GoalMsg = multi_mode_control_msgs::msg::CartesianImpedanceGoal;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using ConfigRequest = multi_mode_control_msgs::srv::SetCartesianImpedance::Request;
using ConfigResponse = multi_mode_control_msgs::srv::SetCartesianImpedance::Response;
using Controller = PandaCartesianImpedanceController;

namespace {
constexpr double kPoseFilterGain = 0.005;
constexpr double kImpedanceFilterGain = 0.005;

Quaterniond normalizedQuaternion(const Quaterniond& q) {
  if (q.norm() < std::numeric_limits<double>::epsilon()) {
    return Quaterniond::Identity();
  }
  Quaterniond normalized = q;
  normalized.normalize();
  return normalized;
}
}  // namespace


static auto registration = ControllerFactory::registerClass<Controller>(
    "panda_cartesian_impedance_controller");

bool Controller::initImpl(const std::vector<RobotData*>& /*robot_data*/,
                          rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
                          std::string name,
                          std::string resource) {
  return loadWorldToFrankaBaseTransformsForResource(
      node, name, resource, 1, arm_ids_, world_to_franka_base_);
}

void Controller::startROSComImpl() {
  {
    std::lock_guard<std::mutex> lock(filter_mutex_);
    has_filtered_pose_ = false;
  }
  {
    std::lock_guard<std::mutex> lock(parameter_mutex_);
    target_parameters_ = this->getParametersBuffered();
    has_target_parameters_ = true;
  }
  const std::string topic_name = "/" + arm_ids_.at(0) + "/arm/end_effector_pose_cmd";
  end_effector_pose_cmd_sub_ =
      PandaControllerBase<Parameters, Pose>::node_->create_subscription<PoseStamped>(
          topic_name, 10,
          [this](const PoseStamped& msg) { this->endEffectorPoseCmdCallback(msg); });
}

void Controller::stopROSComImpl() {
  end_effector_pose_cmd_sub_.reset();
  {
    std::lock_guard<std::mutex> filter_lock(filter_mutex_);
    has_filtered_pose_ = false;
  }
  {
    std::lock_guard<std::mutex> parameter_lock(parameter_mutex_);
    has_target_parameters_ = false;
  }
}

void Controller::endEffectorPoseCmdCallback(const PoseStamped& msg) {
  GoalMsg command_msg;
  command_msg.pose = msg.pose;
  const Pose desired_pose = this->getDesiredPoseBuffered();
  for (std::size_t i = 0; i < 7; ++i) {
    command_msg.q_n[i] = desired_pose.q_n(i);
  }

  Pose p_d;
  if (desiredPoseCallbackImpl(p_d, this->getCurrentPose(), command_msg)) {
    this->setDesiredPoseBuffered(p_d);
  } else {
    this->setError(true);
  }
}

bool Controller::desiredPoseCallbackImpl(Pose& p_d, 
                                         const Pose& /*p*/,
                                         const GoalMsg& msg) {
  Pose goal_pose;
  const Vector3d world_position(msg.pose.position.x, msg.pose.position.y,
                                msg.pose.position.z);
  const Quaterniond world_orientation(msg.pose.orientation.w,
                                      msg.pose.orientation.x,
                                      msg.pose.orientation.y,
                                      msg.pose.orientation.z);
  if (!transformWorldPoseToFrankaBase(world_position, world_orientation,
                                      world_to_franka_base_.at(0),
                                      goal_pose.position, goal_pose.orientation)) {
    auto& clk = *PandaControllerBase<Parameters, Pose>::node_->get_clock();
    const auto& logger = PandaControllerBase<Parameters, Pose>::node_->get_logger();
    RCLCPP_WARN_THROTTLE(
        logger, clk, 1000,
        "panda_cartesian_impedance_controller: Discarding target pose with invalid orientation quaternion.");
    return false;
  }

  goal_pose.q_n = Eigen::Map<const Vector7d>(msg.q_n.data());
  p_d = goal_pose;
  return true;
}

bool Controller::setParametersCallbackImpl(Parameters& p_d, const Parameters& p,
            const std::shared_ptr<ConfigRequest>& request, const std::shared_ptr<ConfigResponse>& response) {
  std::lock_guard<std::mutex> lock(parameter_mutex_);
  target_parameters_.stiffness = Matrix6d(request->stiffness.data());
  target_parameters_.damping_ratio = Vector6d(request->damping_ratio.data());
  target_parameters_.nullspace_stiffness = request->nullspace_stiffness;
  has_target_parameters_ = true;
  p_d = p;
  return true;
}

Parameters Controller::preprocessParametersImpl(const Parameters& p) {
  std::lock_guard<std::mutex> lock(parameter_mutex_);
  if (!has_target_parameters_) {
    target_parameters_ = p;
    has_target_parameters_ = true;
    return p;
  }

  Parameters p_filtered = p;
  p_filtered.stiffness =
      kImpedanceFilterGain * target_parameters_.stiffness +
      (1.0 - kImpedanceFilterGain) * p.stiffness;
  p_filtered.damping_ratio =
      kImpedanceFilterGain * target_parameters_.damping_ratio +
      (1.0 - kImpedanceFilterGain) * p.damping_ratio;
  p_filtered.nullspace_stiffness =
      kImpedanceFilterGain * target_parameters_.nullspace_stiffness +
      (1.0 - kImpedanceFilterGain) * p.nullspace_stiffness;
  this->setParametersBuffered(p_filtered);
  return p_filtered;
}

Pose Controller::filterTargetImpl(const Pose& target_pose) {
  std::lock_guard<std::mutex> lock(filter_mutex_);
  if (!has_filtered_pose_) {
    filtered_pose_ = target_pose;
    filtered_pose_.orientation = normalizedQuaternion(filtered_pose_.orientation);
    has_filtered_pose_ = true;
    return filtered_pose_;
  }

  filtered_pose_.position =
      kPoseFilterGain * target_pose.position +
      (1.0 - kPoseFilterGain) * filtered_pose_.position;
  filtered_pose_.q_n = target_pose.q_n;

  Quaterniond target_orientation = normalizedQuaternion(target_pose.orientation);
  Quaterniond current_orientation = normalizedQuaternion(filtered_pose_.orientation);
  if (current_orientation.coeffs().dot(target_orientation.coeffs()) < 0.0) {
    target_orientation.coeffs() = -target_orientation.coeffs();
  }
  filtered_pose_.orientation =
      current_orientation.slerp(kPoseFilterGain, target_orientation);
  filtered_pose_.orientation.normalize();
  return filtered_pose_;
}
