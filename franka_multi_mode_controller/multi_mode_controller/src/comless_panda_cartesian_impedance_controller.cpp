#include <multi_mode_controller/controllers/comless_panda_cartesian_impedance_controller.h>

#include <algorithm>
#include <cmath>

#include <multi_mode_controller/utils/controller_factory.h>
#include <multi_mode_controller/utils/damping_design.h>
#include <multi_mode_controller/utils/nullspace_projection.h>

using namespace panda_controllers;
using Eigen::Vector3d;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Eigen::Quaterniond;
using Pose = PandaCartesianImpedanceControllerPose;
using Params = PandaCartesianImpedanceControllerParams;
using Controller = ComlessPandaCartesianImpedanceController;

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
}  // namespace

static auto registration = ControllerFactory::registerClass<Controller>(
    "comless_panda_cartesian_impedance_controller");

void Controller::computeTauImpl(const std::vector<std::array<double, 7>*>& tau,
    const Pose& desired, const Params& p) {
  if (tau.empty() || tau[0] == nullptr || robot_data_.empty()) {
    return;
  }
  Pose current = getCurrentPose();
  current.position -= getOffset().position;
  Eigen::Map<const Matrix7d> inertia(robot_data_[0]->mass().data());
  Eigen::Map<const Vector7d> coriolis(robot_data_[0]->coriolis().data());
  Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(
      robot_data_[0]->eeZeroJacobian().data());
  Eigen::Map<const Vector7d> qD(robot_data_[0]->state().dq.data());
  Vector6d error = Vector6d::Zero();
  error.head(3) << current.position - desired.position;
  if (desired.orientation.coeffs().dot(current.orientation.coeffs()) < 0.0) {
    current.orientation.coeffs() << -current.orientation.coeffs();
  }
  const Eigen::Quaterniond rot_error(
      current.orientation.inverse() * desired.orientation);
  error.tail(3) << rot_error.x(), rot_error.y(), rot_error.z();
  error.tail(3) << -current.orientation.toRotationMatrix() * error.tail(3);
  error.head(3) = error.head(3).cwiseMax(kTranslationClipMin);
  error.head(3) = error.head(3).cwiseMin(kTranslationClipMax);
  error.tail(3) = error.tail(3).cwiseMax(kRotationClipMin);
  error.tail(3) = error.tail(3).cwiseMin(kRotationClipMax);

  error_integral_ += kControlDt * error;
  error_integral_ = error_integral_.cwiseMax(kIntegralClipMin);
  error_integral_ = error_integral_.cwiseMin(kIntegralClipMax);

  Matrix6d Ki = Matrix6d::Zero();
  Ki.topLeftCorner(3, 3) = kTranslationalKi * Eigen::Matrix3d::Identity();
  Ki.bottomRightCorner(3, 3) = kRotationalKi * Eigen::Matrix3d::Identity();

  Vector7d tau_task, tau_nullspace, tau_d;
  Matrix6d D = sqrtDesign<6>(pandaCartesianInertia(jacobian, inertia),
      p.stiffness, p.damping_ratio);
  tau_task << jacobian.transpose() *
      (-p.stiffness * error - D * (jacobian * qD) - Ki * error_integral_);
  tau_nullspace <<
      getDynamicallyConsistentNullspaceProjection<7>(inertia, jacobian) *
      (p.nullspace_stiffness * (desired.q_n - current.q_n) -
       (2.0 * std::sqrt(p.nullspace_stiffness)) * qD);
  tau_d << tau_task + tau_nullspace + coriolis;
  for (size_t i = 0; i < 7; ++i) {
    (*tau[0])[i] = tau_d[i];
  }
}

void Controller::postprocessTauImpl(
    const std::vector<std::array<double, 7>*>& tau) {
  if (tau.empty() || tau[0] == nullptr || robot_data_.empty()) {
    return;
  }

  const auto& tau_J_d = robot_data_[0]->state().tau_J_d;
  for (std::size_t i = 0; i < 7; ++i) {
    const double delta_tau = (*tau[0])[i] - tau_J_d[i];
    (*tau[0])[i] =
        tau_J_d[i] + std::clamp(delta_tau, -kDeltaTauMax, kDeltaTauMax);
  }
}

void Controller::onDesiredPoseChangedImpl(const Pose& /*last_desired*/,
    const Pose& /*desired*/) {
  error_integral_.setZero();
}

void Controller::startImpl() {
  error_integral_.setZero();
}

Params Controller::defaultParameters() {
  Params p;
  p.stiffness.setIdentity();
  p.stiffness.topLeftCorner(3, 3) << 400 * Eigen::Matrix3d::Identity();
  p.stiffness.bottomRightCorner(3, 3) << 20 * Eigen::Matrix3d::Identity();
  p.damping_ratio = Vector6d::Constant(0.8);
  p.nullspace_stiffness = 10;
  return p;
}

Pose Controller::getCurrentPoseImpl() {
  Pose p;
  Eigen::Map<const Vector7d> q(robot_data_[0]->state().q.data());
  Eigen::Affine3d transform(
      Eigen::Matrix4d::Map(robot_data_[0]->state().O_T_EE.data()));
  p.position = transform.translation();
  p.orientation = Eigen::Quaterniond(transform.linear());
  p.q_n = q;
  // RCLCPP_INFO_STREAM_THROTTLE(node_->get_logger(), *node_->get_clock(), 500, "getCurrentPoseImpl: " << p.position.transpose() << " " << p.orientation);
  return p;
}

bool Controller::hasOffsetImpl() {
  return getOffset().position.norm() != 0;
}

void Controller::resetOffset() {
  Pose p;
  p.position = Vector3d::Zero();
  p.orientation = Quaterniond(1, 0, 0, 0);
  p.q_n = Vector7d::Zero();
  setOffset(p);
}
