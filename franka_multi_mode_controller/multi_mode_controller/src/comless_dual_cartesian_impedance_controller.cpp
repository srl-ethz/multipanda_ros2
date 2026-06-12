#include <multi_mode_controller/controllers/comless_dual_cartesian_impedance_controller.h>

#include <algorithm>
#include <cmath>

#include <multi_mode_controller/utils/controller_factory.h>
#include <multi_mode_controller/utils/damping_design.h>
#include <multi_mode_controller/utils/nullspace_projection.h>
// SEEMS LIKE A POINTER ISSUE.
// WHEN TWO CLASSES DERIVE FROM THE SAME PARENT, IT DOESN'T SEEM TO WORK
using namespace panda_controllers;
using Eigen::Vector3d;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Eigen::Quaterniond;
using Pose = DualCartesianImpedanceControllerPose;
using Params = DualCartesianImpedanceControllerParams;
using Controller = ComlessDualCartesianImpedanceController;

namespace {
constexpr double kDeltaTauMax = 1.0;
constexpr double kControlDt = 1.0 / 1000.0;
constexpr double kTranslationalKi = 0.0;
constexpr double kRotationalKi = 0.0;

const Vector3d kTranslationClipMin = Vector3d::Constant(-0.1);
const Vector3d kTranslationClipMax = Vector3d::Constant(0.1);
const Vector3d kRotationClipMin = Vector3d::Constant(-0.2);
const Vector3d kRotationClipMax = Vector3d::Constant(0.2);

const Vector6d kIntegralClipMin =
    (Vector6d() << -0.1, -0.1, -0.1, -0.3, -0.3, -0.3).finished();
const Vector6d kIntegralClipMax =
    (Vector6d() << 0.1, 0.1, 0.1, 0.3, 0.3, 0.3).finished();

// Bound on the static task-space force/torque per arm (stiffness + integral
// terms; the damping term vanishes at rest), with margin below the FCI's
// default Cartesian collision reflex thresholds (20 N / 25 Nm) so a blocked
// end-effector pushes steadily instead of tripping the reflex. Clamping the
// *force* (not the error) keeps the bound valid for any runtime stiffness.
constexpr double kMaxStaticForce = 15.0;   // N,  0.75 * 20 N reflex default
constexpr double kMaxStaticTorque = 20.0;  // Nm, 0.80 * 25 Nm reflex default
}  // namespace

static auto registration = ControllerFactory::registerClass<Controller>(
    "comless_dual_cartesian_impedance_controller");

void Controller::computeTauImpl(const std::vector<std::array<double, 7>*>& tau,
    const Pose& desired_poses, const Params& p) {
  if (tau.size() < 2 || robot_data_.size() < 2 || desired_poses.poses.size() < 2 ||
      p.params.size() < 2 || error_integral_.size() < 2) {
    return;
  }

  // measurement setup
  // double start_time = get_wall_time();
  // std::vector<std::array<double,7>> taus_(2);
  // measurement setup

  Pose current_poses = getCurrentPose();
  if (current_poses.poses.size() < 2 || getOffset().poses.size() < 2) {
    return;
  }
  Eigen::VectorXd right_left_qs(17);
  right_left_qs << 0, 0, 0, current_poses.poses[1].q_n, current_poses.poses[0].q_n;
  // Vector14d Q = ... 
  bool isRight = false;
  for(int i = 0; i < 2; i++){
    if (tau[i] == nullptr) {
      isRight = !isRight;
      continue;
    }
    auto& desired = desired_poses.poses[i];
    auto& current = current_poses.poses[i];
    current.position -= getOffset().poses[i].position;
    Eigen::Map<const Matrix7d> inertia(robot_data_[i]->mass().data());
    Eigen::Map<const Vector7d> coriolis(robot_data_[i]->coriolis().data());
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(
        robot_data_[i]->eeZeroJacobian().data());
    Eigen::Map<const Vector7d> qD(robot_data_[i]->state().dq.data());
    Vector6d error = Vector6d::Zero();
    error.head(3) << current.position - desired.position;
    if (desired.orientation.coeffs().dot(current.orientation.coeffs()) < 0.0) {
      current.orientation.coeffs() << -current.orientation.coeffs();
    }
    Vector7d tau_collision(7);
    tau_collision.setZero();
    if(bCollisionAvoidance){
      tau_collision << -redundancy_resolution::CollisionAvoidanceGradient(right_left_qs, isRight);
      // if(isRight){
      //   RCLCPP_INFO_STREAM_THROTTLE(node_->get_logger(), *node_->get_clock(), 500, "CA tau: " <<  tau_collision.transpose());
      // }
    }
    Vector7d tau_manipulability(7);
    tau_manipulability.setZero();
    if(bManipulability){
      tau_manipulability << -redundancy_resolution::ManipulabilityGradient(right_left_qs, isRight);
    }
    isRight = !isRight;
    const Eigen::Quaterniond rot_error(
        current.orientation.inverse() * desired.orientation);
    error.tail(3) << rot_error.x(), rot_error.y(), rot_error.z();
    error.tail(3) << -current.orientation.toRotationMatrix() * error.tail(3);
    error.head(3) = error.head(3).cwiseMax(kTranslationClipMin);
    error.head(3) = error.head(3).cwiseMin(kTranslationClipMax);
    error.tail(3) = error.tail(3).cwiseMax(kRotationClipMin);
    error.tail(3) = error.tail(3).cwiseMin(kRotationClipMax);

    error_integral_.at(i) += kControlDt * error;
    error_integral_.at(i) = error_integral_.at(i).cwiseMax(kIntegralClipMin);
    error_integral_.at(i) = error_integral_.at(i).cwiseMin(kIntegralClipMax);

    Matrix6d Ki = Matrix6d::Zero();
    Ki.topLeftCorner(3, 3) = kTranslationalKi * Eigen::Matrix3d::Identity();
    Ki.bottomRightCorner(3, 3) = kRotationalKi * Eigen::Matrix3d::Identity();

    Vector7d tau_task, tau_nullspace, tau_d;
    Matrix6d D = sqrtDesign<6>(pandaCartesianInertia(jacobian, inertia),
        p.params[i].stiffness, p.params[i].damping_ratio);
    // Static (position-dependent) task force, clamped below the Cartesian
    // collision reflex thresholds so a blocked EE cannot trip the reflex.
    Vector6d f_static =
        -p.params[i].stiffness * error - Ki * error_integral_.at(i);
    f_static.head(3) = f_static.head(3).cwiseMax(-kMaxStaticForce)
                                       .cwiseMin(kMaxStaticForce);
    f_static.tail(3) = f_static.tail(3).cwiseMax(-kMaxStaticTorque)
                                       .cwiseMin(kMaxStaticTorque);
    tau_task << jacobian.transpose() * (f_static - D * (jacobian * qD));
    tau_nullspace <<
        getDynamicallyConsistentNullspaceProjection<7>(inertia, jacobian) *
        (p.params[i].nullspace_stiffness * (desired.q_n - current.q_n) -
        (2.0 * std::sqrt(p.params[i].nullspace_stiffness)) * qD);
    tau_d << tau_task + tau_nullspace + coriolis + tau_collision + tau_manipulability;

    for (size_t j = 0; j < 7; ++j) {
      (*tau[i])[j] = tau_d[j];
      // taus_[i][j] = tau_d[j]; // measurement 
    }
  }
  /*Uncomment for logging */
  // if(getCounter() < max_count_){
  //   double end_time = get_wall_time();
  //   double diff = end_time - start_time;
  //   measureTime(diff);
  //   measureTau(taus_, end_time);
  //   increaseCounter();
  // }
  // else{
  //   if(!logged_){
  //     RCLCPP_INFO_STREAM_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000, "Max count reached: " << getCounter() << "/" << max_count_);
  //     std::string file_name{"dcic"};
  //     write_tau_to_file(file_name);
  //     write_time_to_file(file_name);
  //     RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000, "Files written");
  //     logged_ = true;
  //   }
  // }
  /*Uncomment for logging */
}

void Controller::postprocessTauImpl(
    const std::vector<std::array<double, 7>*>& tau) {
  const std::size_t arm_count = std::min(tau.size(), robot_data_.size());
  for (std::size_t arm_index = 0; arm_index < arm_count; ++arm_index) {
    if (tau[arm_index] == nullptr) {
      continue;
    }
    const auto& tau_J_d = robot_data_[arm_index]->state().tau_J_d;
    for (std::size_t joint_index = 0; joint_index < 7; ++joint_index) {
      const double delta_tau =
          (*tau[arm_index])[joint_index] - tau_J_d[joint_index];
      (*tau[arm_index])[joint_index] =
          tau_J_d[joint_index] +
          std::clamp(delta_tau, -kDeltaTauMax, kDeltaTauMax);
    }
  }
}

void Controller::onDesiredPoseChangedImpl(const Pose& /*last_desired*/,
    const Pose& /*desired*/) {
  for (auto& integral : error_integral_) {
    integral.setZero();
  }
}

void Controller::startImpl() {
  for (auto& integral : error_integral_) {
    integral.setZero();
  }
}

Params Controller::defaultParameters() {
  Params p;
  for(int i = 0; i < 2; i++){
    p.params[i].stiffness.setIdentity();
    p.params[i].stiffness.topLeftCorner(3, 3) << 400 * Eigen::Matrix3d::Identity();
    p.params[i].stiffness.bottomRightCorner(3, 3) << 20 * Eigen::Matrix3d::Identity();
    p.params[i].damping_ratio = Vector6d::Constant(0.8);
    p.params[i].nullspace_stiffness = 10;
  }
    RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000, "Params set");
  
  return p;
}

Pose Controller::getCurrentPoseImpl() {
  Pose p;
  for(int i = 0; i < 2; i++){
    Eigen::Map<const Vector7d> q(robot_data_[i]->state().q.data());
    Eigen::Affine3d transform(
        Eigen::Matrix4d::Map(robot_data_[i]->state().O_T_EE.data()));
    p.poses[i].position = transform.translation();
    p.poses[i].orientation = Eigen::Quaterniond(transform.linear());
    p.poses[i].q_n = q;
  }
  // RCLCPP_INFO_STREAM_THROTTLE(node_->get_logger(), *node_->get_clock(), 500, "getCurrentPoseImpl: " << p.position.transpose() << " " << p.orientation);
  return p;
}

bool Controller::hasOffsetImpl() {
  return getOffset().poses[0].position.norm() != 0 ||
      getOffset().poses[1].position.norm() != 0;
}

void Controller::resetOffset() {
  Pose p;
  for(int i = 0; i<2; i++){
    p.poses[i].position = Vector3d::Zero();
    p.poses[i].orientation = Quaterniond(1, 0, 0, 0);
    p.poses[i].q_n = Vector7d::Zero();
  }
  setOffset(p);
}
