#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <multi_mode_controller/base/panda_controller_ros_interface.h>
#include <multi_mode_controller/controllers/comless_panda_mpc_controller.h>
#include <multi_mode_control_msgs/srv/set_mpc.hpp>

namespace panda_controllers {

using MpcServiceParameter = multi_mode_control_msgs::srv::SetMpc;
using MpcDesPoseMsg = geometry_msgs::msg::PoseStamped;

// ROS-facing Stage-1 Cartesian-tracking MPC controllet.
//
// Interfaces (resource == arm_id for a single arm):
//   * <arm_id>/arm/mpc_end_effector_pose_cmd  (geometry_msgs/PoseArray, world
//     frame): a sequence of EE poses appended to the waypoint buffer.
//   * <arm_id>/arm/mpc_clear_waypoints        (std_srvs/Trigger): clears the
//     unexecuted waypoints.
//   * <resource>/<name>/desired_pose          (geometry_msgs/PoseStamped, world
//     frame, provided by ControllerRosInterface): a single-pose convenience
//     goal appended as one waypoint.
//   * <resource>/<name>/parameters            (multi_mode_control_msgs/SetMpc):
//     sets the PD gains and MPC cost weights.
class PandaMpcController :
    public virtual ComlessPandaMpcController,
    public virtual ControllerRosInterface<MpcServiceParameter,
                                           MpcDesPoseMsg,
                                           PandaMpcControllerParams,
                                           PandaMpcControllerPose> {
 public:
  virtual ~PandaMpcController() = default;

 private:
  bool initImpl(const std::vector<RobotData*>& robot_data,
                rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
                std::string name,
                std::string resource) override final;

  bool desiredPoseCallbackImpl(PandaMpcControllerPose& p_d,
                               const PandaMpcControllerPose& p,
                               const MpcDesPoseMsg& msg) override final;

  bool setParametersCallbackImpl(
      PandaMpcControllerParams& p_d,
      const PandaMpcControllerParams& p,
      const MpcServiceParameter::Request::SharedPtr& req,
      const MpcServiceParameter::Response::SharedPtr& res) override final;

  void startROSComImpl() override final;
  void stopROSComImpl() override final;

  void waypointSequenceCallback(const geometry_msgs::msg::PoseArray& msg);
  void clearWaypointsCallback(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
      const std::shared_ptr<std_srvs::srv::Trigger::Response> res);

  std::vector<std::string> arm_ids_;
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      world_to_franka_base_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr
      waypoint_sequence_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_waypoints_srv_;
};

}  // namespace panda_controllers
