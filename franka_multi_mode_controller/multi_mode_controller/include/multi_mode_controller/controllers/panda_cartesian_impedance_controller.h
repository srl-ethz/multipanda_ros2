#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <array>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include <multi_mode_controller/base/panda_controller_ros_interface.h>
#include <multi_mode_controller/controllers/comless_panda_cartesian_impedance_controller.h>
#include <multi_mode_control_msgs/srv/set_cartesian_impedance.hpp>
#include <multi_mode_control_msgs/msg/cartesian_impedance_goal.hpp>

namespace panda_controllers {
    using GoalMsg = multi_mode_control_msgs::msg::CartesianImpedanceGoal;
    using ServiceParameter = multi_mode_control_msgs::srv::SetCartesianImpedance;
    using Pose = PandaCartesianImpedanceControllerPose;
    using Parameters = PandaCartesianImpedanceControllerParams;

class PandaCartesianImpedanceController :
    public virtual ComlessPandaCartesianImpedanceController,
    public virtual ControllerRosInterface<ServiceParameter, 
                                          GoalMsg, 
                                          Parameters, 
                                          Pose> {
public:
  virtual ~PandaCartesianImpedanceController() = default;

private:
  bool initImpl(const std::vector<RobotData*>& robot_data,
                rclcpp_lifecycle::LifecycleNode::SharedPtr& node,
                std::string name,
                std::string resource) override final;

  bool desiredPoseCallbackImpl(Pose& p_d,
                               const Pose& p,
                               const GoalMsg& msg)
                               override final;

  bool setParametersCallbackImpl(Parameters& p_d, 
        const Parameters& p,
        const ServiceParameter::Request::SharedPtr& req, 
        const ServiceParameter::Response::SharedPtr& res) 
        override final;
  Parameters preprocessParametersImpl(const Parameters& p) override final;
  void postprocessTauImpl(
      const std::vector<std::array<double, 7>*>& tau) override final;

  void startROSComImpl() override final;
  void stopROSComImpl() override final;
  void endEffectorPoseCmdCallback(const geometry_msgs::msg::PoseStamped& msg);
  void updateWaypointTowardsGoal();
  Pose lowPassPose(const Pose& target_pose);

  std::vector<std::string> arm_ids_;
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      world_to_franka_base_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      end_effector_pose_cmd_sub_;
  rclcpp::TimerBase::SharedPtr waypoint_timer_;
  Pose goal_pose_;
  Pose filtered_pose_;
  Parameters target_parameters_;
  bool has_goal_{false};
  bool has_filtered_pose_{false};
  bool has_target_parameters_{false};
  std::mutex goal_mutex_;
  std::mutex filter_mutex_;
  std::mutex parameter_mutex_;

};
}
