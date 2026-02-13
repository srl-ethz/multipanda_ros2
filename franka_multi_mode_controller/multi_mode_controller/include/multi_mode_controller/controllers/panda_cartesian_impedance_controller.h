#pragma once

#include <string>
#include <vector>

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

  void startROSComImpl() override final;
  void stopROSComImpl() override final;
  void endEffectorPoseCmdCallback(const geometry_msgs::msg::PoseStamped& msg);

  std::vector<std::string> arm_ids_;
  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>>
      world_to_franka_base_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      end_effector_pose_cmd_sub_;

};
}
