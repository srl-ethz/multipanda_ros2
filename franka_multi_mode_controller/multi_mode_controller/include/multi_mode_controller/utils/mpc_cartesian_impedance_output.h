#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include <multi_mode_controller/utils/damping_design.h>
#include <multi_mode_controller/utils/nullspace_projection.h>

namespace panda_controllers {

struct MpcCartesianImpedanceOutput {
  Eigen::Matrix<double, 7, 1> tau{Eigen::Matrix<double, 7, 1>::Zero()};
  Eigen::Matrix<double, 6, 1> static_wrench{Eigen::Matrix<double, 6, 1>::Zero()};
};

// Convert the MPC's planned joint-space feed-forward into an always-on
// Cartesian impedance output. The task component is combined with compliant
// feedback before clamping, so the steady commanded wrench stays bounded even
// when a blocked MPC continues planning toward its target.
inline MpcCartesianImpedanceOutput computeMpcCartesianImpedanceOutput(
    const Eigen::Matrix<double, 7, 1>& q,
    const Eigen::Matrix<double, 7, 1>& dq,
    const Eigen::Matrix<double, 7, 1>& q_ref,
    const Eigen::Matrix<double, 7, 1>& tau_ff,
    const Eigen::Matrix<double, 7, 7>& mass,
    const Eigen::Matrix<double, 7, 1>& coriolis,
    const Eigen::Matrix<double, 6, 7>& jacobian,
    const Eigen::Matrix<double, 6, 6>& stiffness,
    const Eigen::Matrix<double, 6, 1>& damping_ratio,
    double nullspace_stiffness,
    double max_static_force = 15.0,
    double max_static_torque = 20.0) {
  using Matrix7d = Eigen::Matrix<double, 7, 7>;
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  const Vector7d tau_motion = tau_ff - coriolis;
  const Matrix7d mass_inverse = mass.ldlt().solve(Matrix7d::Identity());
  const Eigen::Matrix<double, 6, 6> task_inverse_mass =
      jacobian * mass_inverse * jacobian.transpose();
  const Vector6d planned_task_acceleration = jacobian * mass_inverse * tau_motion;
  const Vector6d planned_task_wrench = task_inverse_mass.ldlt().solve(planned_task_acceleration);

  MpcCartesianImpedanceOutput output;
  const Vector6d task_error = jacobian * (q - q_ref);
  output.static_wrench = planned_task_wrench - stiffness * task_error;
  output.static_wrench.head(3) =
      output.static_wrench.head(3).cwiseMax(-max_static_force).cwiseMin(max_static_force);
  output.static_wrench.tail(3) =
      output.static_wrench.tail(3).cwiseMax(-max_static_torque).cwiseMin(max_static_torque);
  const Eigen::Matrix<double, 6, 6> damping =
      sqrtDesign<6>(pandaCartesianInertia(jacobian, mass), stiffness, damping_ratio);
  const Vector7d tau_task =
      jacobian.transpose() * (output.static_wrench - damping * (jacobian * dq));

  const Matrix7d nullspace = getDynamicallyConsistentNullspaceProjection<7>(mass, jacobian);
  const double nullspace_damping = 2.0 * std::sqrt(std::max(0.0, nullspace_stiffness));
  const Vector7d tau_nullspace =
      nullspace * (tau_motion + nullspace_stiffness * (q_ref - q) - nullspace_damping * dq);
  output.tau = coriolis + tau_task + tau_nullspace;
  return output;
}

}  // namespace panda_controllers
