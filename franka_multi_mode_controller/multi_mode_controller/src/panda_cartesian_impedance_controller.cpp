#include <multi_mode_controller/controllers/panda_cartesian_impedance_controller.h>

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
  const std::string topic_name = "/" + arm_ids_.at(0) + "/arm/end_effector_pose_cmd";
  end_effector_pose_cmd_sub_ =
      PandaControllerBase<Parameters, Pose>::node_->create_subscription<PoseStamped>(
          topic_name, 10,
          [this](const PoseStamped& msg) { this->endEffectorPoseCmdCallback(msg); });
}

void Controller::stopROSComImpl() {
  end_effector_pose_cmd_sub_.reset();
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
                                         const Pose& p,
                                         const GoalMsg& msg) {
  const Vector3d world_position(msg.pose.position.x, msg.pose.position.y,
                                msg.pose.position.z);
  const Quaterniond world_orientation(msg.pose.orientation.w,
                                      msg.pose.orientation.x,
                                      msg.pose.orientation.y,
                                      msg.pose.orientation.z);
  if (!transformWorldPoseToFrankaBase(world_position, world_orientation,
                                      world_to_franka_base_.at(0),
                                      p_d.position, p_d.orientation)) {
    auto& clk = *PandaControllerBase<Parameters, Pose>::node_->get_clock();
    const auto& logger = PandaControllerBase<Parameters, Pose>::node_->get_logger();
    RCLCPP_WARN_THROTTLE(
        logger, clk, 1000,
        "panda_cartesian_impedance_controller: Discarding target pose with invalid orientation quaternion.");
    return false;
  }
  if ((p_d.position+PandaControllerBase<Parameters, Pose>::getOffset().position-p.position).norm() > 0.1) {
    auto& clk = *PandaControllerBase<Parameters, Pose>::node_->get_clock();
    const auto& logger = PandaControllerBase<Parameters, Pose>::node_->get_logger();
    RCLCPP_WARN_THROTTLE(logger, clk, 1000, "panda_cartesian_impedance_controller: Discarding "
        "target pose that is too far away from current pose (%f m, allowed "
        "maximum is 0.1 m).",
        (p_d.position+PandaControllerBase<Parameters, Pose>::getOffset().position-p.position).norm());
    return false;
  }
  if (p.orientation.angularDistance(p_d.orientation) > 0.15) {
    auto& clk = *PandaControllerBase<Parameters, Pose>::node_->get_clock();
    const auto& logger = PandaControllerBase<Parameters, Pose>::node_->get_logger();
    RCLCPP_WARN_THROTTLE(logger, clk, 1000, "panda_cartesian_impedance_controller: Discarding "
        "target pose that rotates too far away from current pose (%f rad, "
        "allowed maximum is 0.15 rad).",
        p.orientation.angularDistance(p_d.orientation));
    return false;
  }
  p_d.q_n = Eigen::Map<const Vector7d>(msg.q_n.data());
  return true;
}

bool Controller::setParametersCallbackImpl(Parameters& p_d, const Parameters& p,
            const std::shared_ptr<ConfigRequest>& request, const std::shared_ptr<ConfigResponse>& response) {
  p_d.stiffness = Matrix6d(request->stiffness.data());
  p_d.damping_ratio = Vector6d(request->damping_ratio.data());
  p_d.nullspace_stiffness = request->nullspace_stiffness;
  return true;
}
