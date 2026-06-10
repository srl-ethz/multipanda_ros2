#pragma once

#include <vector>

#include <Eigen/Dense>

namespace panda_controllers {

// Stage-1 Cartesian-tracking MPC for a single 7-DOF Panda arm.
//
// Formulation (see MPC_development_plan.md):
//   * Multiple-shooting transcription. Decision variables are the predicted
//     states x_k = [q_k; dq_k] (k = 1..T) and the joint-torque inputs
//     u_k = tau_k (k = 0..T-1).
//   * The rigid-body dynamics are FROZEN at the measured state x_0: the mass
//     matrix M, the Coriolis vector c and the end-effector Jacobian J are held
//     constant over the horizon, which turns the problem into a single convex
//     QP (no SQP re-linearization in Stage 1).
//   * Gravity is intentionally NOT modelled: libfranka (and the MuJoCo sim
//     layer) auto-compensate gravity, so the commanded torque satisfies
//     M*ddq + c = tau and the feed-forward torque already contains c.
//   * Task-space tracking is linearized as e_k = J*(q_k - q_0) - r_k, a
//     Gauss-Newton step, where r_k is the base-frame twist that moves the
//     current EE pose onto the reference pose at horizon step k.
//   * Box limits on q, dq and tau enter as simple bounds.
//
// The QP is assembled sparsely and solved with osqp-eigen. solve() is called
// from the controllet's background (~100 Hz) thread, never from the 1 kHz loop.
class CartesianMpc {
 public:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using Matrix7d = Eigen::Matrix<double, 7, 7>;
  using Matrix67d = Eigen::Matrix<double, 6, 7>;

  struct Config {
    int horizon = 10;             // T, number of stages
    double dt = 0.01;             // s, prediction timestep
    double dq_weight = 1e-2;      // joint-velocity regularization
    double terminal_scale = 10.0; // multiplier on the final-stage tracking cost
    int max_iteration = 10000;    // OSQP iteration cap

    // Safety margins applied to the official Panda limits (panda_limits.h) when
    // forming the box constraints. The limits there are the *hard* limits at
    // which the robot faults, so the MPC keeps a margin:
    //   * position: shrink [q_min, q_max] inward by joint_position_margin (rad)
    //     to stay clear of the joint-limit reflex,
    //   * velocity: allow only velocity_limit_scale * qD_max,
    //   * torque:   plan tau_ff only up to torque_limit_scale * tau_max, leaving
    //     headroom for the 1 kHz tracking PD that is added on top of tau_ff
    //     (the final command is still hard-clamped to tau_max downstream).
    double joint_position_margin = 0.1;  // rad (~5.72958 deg)
    double velocity_limit_scale = 0.95;
    double torque_limit_scale = 0.9;
  };

  // Runtime-tunable weights (settable through the SetMpc service).
  struct Weights {
    Vector6d task_weight = (Vector6d() << 1, 1, 1, 1, 1, 1).finished();
    double input_weight = 1e-3;
  };

  struct Solution {
    bool success = false;
    // Index s in [0, T-1] holds the prediction for stage s:
    //   q_ref[s], dq_ref[s] are the predicted state x_{s+1},
    //   tau_ff[s] is the input u_s.
    std::vector<Vector7d> q_ref;
    std::vector<Vector7d> dq_ref;
    std::vector<Vector7d> tau_ff;
  };

  explicit CartesianMpc(const Config& config);

  int horizon() const { return config_.horizon; }
  double dt() const { return config_.dt; }

  // Solve one QP. refs must have config_.horizon entries; refs[k] is the
  // base-frame twist [dp; dtheta] (3 translation, 3 rotation-vector) that maps
  // the current EE pose onto the reference pose at horizon step k+1.
  // Returns true and fills `solution` on success; on failure returns false and
  // leaves solution.success == false.
  bool solve(const Vector7d& q0,
             const Vector7d& dq0,
             const Matrix7d& mass,
             const Vector7d& coriolis,
             const Matrix67d& jacobian,
             const std::vector<Vector6d>& refs,
             const Weights& weights,
             Solution& solution);

 private:
  Config config_;
};

}  // namespace panda_controllers
