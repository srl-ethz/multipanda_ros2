#include <multi_mode_controller/controllers/dual_cartesian_impedance_controller.h>

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
constexpr double kPoseFilterGain = 0.01;
constexpr double kImpedanceFilterGain = 0.005;

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
  {
    std::lock_guard<std::mutex> lock(filter_mutex_);
    has_filtered_pose_ = false;
  }
  {
    std::lock_guard<std::mutex> lock(parameter_mutex_);
    target_parameters_ = this->getParametersBuffered();
    has_target_parameters_ = true;
  }
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
}

void Controller::stopROSComImpl() {
  end_effector_pose_cmd_subs_.clear();
  {
    std::lock_guard<std::mutex> filter_lock(filter_mutex_);
    has_filtered_pose_ = false;
  }
  {
    std::lock_guard<std::mutex> parameter_lock(parameter_mutex_);
    has_target_parameters_ = false;
  }
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
                                         const Pose& /*p*/,
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
  p_d = goal_pose;
  return true;
}

bool Controller::setParametersCallbackImpl(Parameters& p_d, const Parameters& p,
            const std::shared_ptr<ConfigRequest>& request, const std::shared_ptr<ConfigResponse>& response) {
  std::lock_guard<std::mutex> lock(parameter_mutex_);
  for(int i=0; i<2; i++){
    target_parameters_.params[i].stiffness = Matrix6d(request->stiffness.data());
    target_parameters_.params[i].damping_ratio = Vector6d(request->damping_ratio.data());
    target_parameters_.params[i].nullspace_stiffness = request->nullspace_stiffness;
  }
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
  for (std::size_t i = 0; i < p_filtered.params.size(); ++i) {
    p_filtered.params[i].stiffness =
        kImpedanceFilterGain * target_parameters_.params[i].stiffness +
        (1.0 - kImpedanceFilterGain) * p.params[i].stiffness;
    p_filtered.params[i].damping_ratio =
        kImpedanceFilterGain * target_parameters_.params[i].damping_ratio +
        (1.0 - kImpedanceFilterGain) * p.params[i].damping_ratio;
    p_filtered.params[i].nullspace_stiffness =
        kImpedanceFilterGain * target_parameters_.params[i].nullspace_stiffness +
        (1.0 - kImpedanceFilterGain) * p.params[i].nullspace_stiffness;
  }
  this->setParametersBuffered(p_filtered);
  return p_filtered;
}

Pose Controller::filterTargetImpl(const Pose& target_pose) {
  std::lock_guard<std::mutex> lock(filter_mutex_);
  if (!has_filtered_pose_) {
    filtered_pose_ = target_pose;
    for (auto& pose : filtered_pose_.poses) {
      pose.orientation = normalizedQuaternion(pose.orientation);
    }
    has_filtered_pose_ = true;
    return filtered_pose_;
  }

  for (std::size_t i = 0; i < filtered_pose_.poses.size(); ++i) {
    filtered_pose_.poses[i].position =
        kPoseFilterGain * target_pose.poses[i].position +
        (1.0 - kPoseFilterGain) * filtered_pose_.poses[i].position;
    filtered_pose_.poses[i].q_n = target_pose.poses[i].q_n;

    Quaterniond target_orientation =
        normalizedQuaternion(target_pose.poses[i].orientation);
    Quaterniond current_orientation =
        normalizedQuaternion(filtered_pose_.poses[i].orientation);
    if (current_orientation.coeffs().dot(target_orientation.coeffs()) < 0.0) {
      target_orientation.coeffs() = -target_orientation.coeffs();
    }
    filtered_pose_.poses[i].orientation =
        current_orientation.slerp(kPoseFilterGain, target_orientation);
    filtered_pose_.poses[i].orientation.normalize();
  }
  return filtered_pose_;
}
