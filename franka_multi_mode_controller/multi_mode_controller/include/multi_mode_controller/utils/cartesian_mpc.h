#pragma once

#include <string>
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
//   * A task-space VELOCITY tracking term velocity_weight*||J*dq_k - v_k||^2_W
//     where v_k = (r_{k+1} - r_k)/dt is the reference velocity estimated by
//     finite differences of the (interpolated) reference poses. A held or
//     single-waypoint target gives constant r_k, so v_k == 0 and the term
//     degenerates to pure task-space damping; along a commanded trajectory it
//     is a velocity FEED-FORWARD that tracks the target's motion instead of
//     dragging on it.
//   * An acceleration smoothness term accel_weight*||ddq_k||^2 with
//     ddq_k = Minv*(u_k - c) (the frozen forward dynamics), quadratic in u_k,
//     discouraging jerky torque plans.
//   * A posture regularization posture_weight*||N*(q_k - q_nominal)||^2,
//     with N = I - pinv(J)*J the null-space projector of the frozen Jacobian,
//     resolves the arm's redundancy: without it the null space drifts freely,
//     pulling the arm into ill-conditioned configurations where the frozen
//     linearization is increasingly wrong. The projection is essential: an
//     UNPROJECTED posture cost fights the task cost and biases the converged
//     EE pose by ~posture_weight/(task_weight*sigma(J)) - measured ~15-20 mm
//     and ~4-6 deg at poses far from q_nominal.
//   * A TRUST REGION |q_k - q0| <= trust_region keeps every predicted state
//     inside the neighbourhood where the frozen M, c, J are still accurate
//     (the linearization error grows with ||q_k - q0||, not with the stage
//     index). Long motions are realized by the 100 Hz re-solve, each step
//     re-freezing the model at the new state.
//   * Joint-lock robustness, in two parts (no slack variables: those make the
//     active-set degenerate and break OSQP's polish step, which this solver
//     relies on for accuracy):
//       1. Feasibility guarantee. Before assembly, a frozen-dynamics BRAKING
//          rollout (tau = clamp(-M dq/dt + c)) is computed - a certified
//          feasible trajectory - and each stage's q/dq box is widened just
//          enough to contain it. In normal operation the widening is a no-op;
//          when the measured state has drifted past a margined bound the QP
//          stays feasible (it can always plan the brake) instead of failing
//          PRIMAL_INFEASIBLE, which would freeze the controller at the limit.
//       2. Active recovery. Joints near/past their margined position limit
//          get a quadratic pull w_i*(q_k - q_safe)^2 toward a point just
//          inside the limit, steering them back into the valid range over the
//          next solves. The weight w_i ramps in continuously over a band
//          inside the limit (Config::recovery_band) instead of switching on
//          at the bound: a binary switch made the plan jump between "sit on
//          the bound" and "retreat recovery_backoff inward" whenever a
//          disturbance held a joint hovering at the bound (bang-bang limit
//          cycle at the re-solve rate; see Config::recovery_weight).
//     Torque bounds stay hard (inputs are free variables, so they can never
//     cause infeasibility).
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
    // Closed-loop damping: without it the 100 Hz re-solve chain produces
    // near-minimum-time plans that fly THROUGH the target with momentum,
    // closing an underdamped loop around a constant target (measured ~1 Hz,
    // +-10..20 mm limit-cycle-like ringing for seconds). It comes from two
    // terms: (a) the per-stage velocity-tracking cost (Weights::
    // velocity_weight), which for a held target degenerates to task-space
    // damping and for a moving target tracks the reference velocity instead
    // of dragging on it, and (b) this plain ||dq_T||^2 terminal penalty,
    // which makes each plan end near rest. (b) does drag a moving target (a
    // cruise velocity v at the horizon end costs terminal_dq_weight * v^2),
    // so keep it moderate: at 10 the drag is not measurable on a 0.2/0.3 Hz,
    // 10 cm Lissajous (amp 100%, lag < 10 ms) while a 10 cm step settles to
    // 2 mm in ~0.4 s with ~6 mm overshoot and a 70 deg rotation step settles
    // to 0.7 deg in ~0.7 s with no overshoot (Stage-1.10 numbers, measured
    // with a faithful sim torque-rate limiter). (A terminal look-ahead cost
    // tracking the coasted pose J*(q_T + alpha*dq_T) was removed once the
    // velocity term carried the moving-target case; see
    // MPC_development_plan.md Stages 1.9/1.10.)
    double terminal_dq_weight = 10.0;
    int max_iteration = 10000;    // OSQP iteration cap
    // OSQP ADMM termination tolerances. The QP is poorly scaled (dt*Minv /
    // dt^2*Minv factors), so 1e-6 does NOT converge within max_iteration on
    // real-robot instances with a moving target - the runtime default is
    // therefore moderate, and accuracy comes from POLISH (exact active-set
    // refinement) plus the warm-started 100 Hz re-solve chain. Unit tests
    // that probe shallow (null-space) directions with a single cold solve
    // tighten this to 1e-6 explicitly.
    double eps_abs = 1e-5;
    double eps_rel = 1e-5;

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

    // Trust region of the frozen linearization: every predicted q_k must stay
    // within +- trust_region (rad) of the measured q0 (soft, see below). Sized
    // so the frozen M/c/J remain accurate; 0.15 rad does not bind during
    // normal tracking (<= qD_max * horizon would be ~0.22 rad flat out).
    double trust_region = 0.15;
    // Quadratic pull toward q_safe = limit -+ recovery_backoff for joints
    // near/past their margined position limit. Sized at task-weight scale so
    // it overrules the tracking cost's wish to hold the offending joint,
    // without distorting the QP scaling. The weight ramps in CONTINUOUSLY
    // over recovery_band (rad) inside the margined limit - zero at the band
    // edge, quadratic ramp to recovery_weight AT the limit, growing further
    // outside it. A binary on/off switch here (the original formulation) made
    // the planned q_ref jump by ~recovery_backoff between consecutive solves
    // whenever a disturbance (e.g. an unmodelled EE payload, which the 1 kHz
    // PD only rejects with ~tau_load/kp steady-state error) held a joint
    // hovering at the bound - a solve-rate bang-bang limit cycle, ~5 mm of EE
    // oscillation. With the ramp, the recovery pull and the task cost balance
    // at a fixed point that moves continuously with the measured state.
    double recovery_weight = 5e3;
    double recovery_backoff = 0.02;  // rad inside the margined limit
    double recovery_band = 0.05;     // rad, ramp-in width inside the limit
    // The reference twists fed to the Gauss-Newton cost are clamped to these
    // magnitudes. The trust region caps the achievable motion per solve
    // anyway, so larger reference errors only inject outsized gradients into
    // the QP (aggressive, jerky plans far outside the linearization's
    // validity).
    double max_ref_translation = 0.3;  // m
    double max_ref_rotation = 0.5;     // rad
    // REFERENCE SPEED GOVERNOR. The state boxes bound the PLAN, not the
    // PLANT: at 100 Hz only the first ~10 ms of each plan is executed, so a
    // far target whose clamped reference stays max_ref_* ahead of the EE (a
    // RECEDING CARROT) makes every plan a full-torque sprint whose planned
    // deceleration tail is perpetually postponed. The 1 kHz PD adds up to
    // tau_max (vs the plan's 0.9 tau_max) on top, and the braking-rollout
    // feasibility widening then accepts the resulting overspeed as the next
    // x0 instead of faulting - measured in sim: joints at 2x the velocity
    // box, EE at 2.5 m/s and ~380 W peak mechanical power for a 40 cm step
    // (power_limit_violation on the real robot). The fix is at the source:
    // each reference twist is additionally clamped to a per-stage SPEED CONE
    //   |r_k| <= ref_catchup + ref_speed * (k+1)*dt,
    // a slope of ref_speed plus a small catch-up allowance so mm-scale
    // tracking error does not eat into the speed budget. Because the cone
    // re-centers on the EE every re-solve, the allowance is re-granted each
    // cycle and the EFFECTIVE approach speed is ref_speed + ref_catchup /
    // (T*dt) - 0.4 m/s at the defaults, measured exactly in sim. A far
    // target becomes a governed cruise; references already moving slower
    // than ref_speed (queued trajectories, the validation Lissajous) are
    // untouched. The reference VELOCITY feed-forward is computed from the
    // governed refs, so a held far target gets the cruise velocity as
    // feed-forward instead of v == 0. Governed 41 cm step (vs un-governed):
    // peak EE 0.45 m/s (2.52), peak dq 1.41 rad/s (4.01, vs box 2.066),
    // peak total power 60 W (376), settle 1.12 s (1.17) - same arrival
    // time, the sprint bought nothing but the violation.
    double ref_speed_translation = 0.3;  // m/s, governed approach speed
    double ref_speed_rotation = 1.0;     // rad/s
    double ref_catchup_translation = 0.01;  // m, tracking-error allowance
    double ref_catchup_rotation = 0.05;     // rad
    // The reference VELOCITIES (finite differences of the governed reference
    // twists, see the class comment) are clamped likewise: a target JUMP
    // (immediate-target override replacing the waypoint queue) shows up as a
    // huge apparent target velocity for one command_dt; the clamp keeps the
    // feed-forward within what the arm can physically follow (Panda Cartesian
    // limits are ~1.7 m/s / 2.5 rad/s).
    double max_ref_velocity_translation = 1.0;  // m/s
    double max_ref_velocity_rotation = 2.0;     // rad/s
    // Nominal joint posture for the redundancy-resolving regularization
    // (default: the Panda "ready" pose used as initial_positions in sim).
    Vector7d q_nominal = (Vector7d() << 0.0, -0.785398163397448, 0.0,
                          -2.356194490192345, 0.0, 1.570796326794897,
                          0.785398163397448).finished();
  };

  // Runtime-tunable weights (settable through the SetMpc service).
  struct Weights {
    Vector6d task_weight = (Vector6d() << 1, 1, 1, 1, 1, 1).finished();
    double input_weight = 1e-3;
    // Weight of ||N*(q_k - q_nominal)||^2 per stage (N = null-space projector
    // of the frozen Jacobian). Acts only in the redundancy direction, so it
    // cannot bias the EE pose; it just has to dominate dq_weight along the
    // null space.
    double posture_weight = 10.0;
    // Weight of the task-space velocity tracking ||J*dq_k - v_k||^2_W per
    // stage (v_k = reference velocity from finite differences of the
    // GOVERNED reference twists; zero for a held in-reach target, where this
    // becomes damping, and the governed cruise speed for a held far target).
    // Shares the task_weight metric W, so its value is a time constant
    // squared (velocity_weight = tau^2: a velocity error of e/tau costs like
    // a position error of e). Keep SMALL: it acts at every stage, so with
    // v_k == 0 (held target / single-pose commands) it drags the approach
    // the same way the rejected plain terminal ||dq||^2 damper did -
    // tau^2 = 2e-3 totals ~20% of that damper's magnitude across the
    // horizon. The moving-target case is carried by the v_k feed-forward,
    // not by a large weight.
    double velocity_weight = 2e-3;
    // Weight of the acceleration smoothness ||Minv*(u_k - c)||^2 per input.
    double accel_weight = 1e-4;
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

  // Override the nominal posture (call before the solve thread starts).
  void setNominalPosture(const Vector7d& q_nominal) {
    config_.q_nominal = q_nominal;
  }

  // Why the last solve() returned false (OSQP status or setup stage), for
  // diagnostics. Only meaningful right after a failed solve(); written by the
  // same (single) thread that calls solve().
  const std::string& lastFailure() const { return last_failure_; }

  // Solver statistics of the last solve() that reached OSQP, for diagnostics
  // (same single-thread caveat as lastFailure()).
  struct Stats {
    int iterations = 0;
    double run_time = 0.0;     // s, OSQP-internal (setup + solve + polish)
    int polish_status = 0;     // 1 = successful, 0 = not tried, -1 = failed
  };
  const Stats& lastStats() const { return last_stats_; }

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
  std::string last_failure_;
  Stats last_stats_;
  // Warm-start chain: primal/dual iterate of the previous solve (also kept
  // when that solve hit the iteration cap - consecutive 10 ms-apart QPs are
  // near-identical, so a hard instance keeps converging ACROSS solve cycles
  // instead of restarting ADMM from cold every time).
  Eigen::VectorXd z_warm_;
  Eigen::VectorXd y_warm_;
  bool warm_valid_ = false;
};

}  // namespace panda_controllers
