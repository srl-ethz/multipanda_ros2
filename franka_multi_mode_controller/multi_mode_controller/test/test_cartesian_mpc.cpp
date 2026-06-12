// Unit tests for the Stage-1 Cartesian-tracking MPC QP layer (CartesianMpc).
// These exercise the solver/cost/constraint assembly in isolation (no ROS, no
// sim) with a synthetic but realistic SPD mass matrix and a full-rank Jacobian.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <multi_mode_controller/utils/cartesian_mpc.h>

using panda_controllers::CartesianMpc;
using V7 = Eigen::Matrix<double, 7, 1>;
using V6 = Eigen::Matrix<double, 6, 1>;
using M7 = Eigen::Matrix<double, 7, 7>;
using M67 = Eigen::Matrix<double, 6, 7>;

namespace {

M7 testMass() {
  M7 m = M7::Identity();
  m.diagonal() << 2.7, 2.7, 2.0, 2.0, 1.0, 0.8, 0.4;  // SPD, Panda-ish
  return m;
}

// Identity-on-the-first-6-joints Jacobian: task dimension i maps to joint i.
M67 identityJacobian() {
  M67 j = M67::Zero();
  for (int i = 0; i < 6; ++i) {
    j(i, i) = 1.0;
  }
  return j;
}

// The Panda ready pose - matches CartesianMpc::Config::q_nominal exactly, so
// the posture regularization is gradient-free at this configuration.
V7 homeQ() {
  V7 q;
  q << 0.0, -0.785398163397448, 0.0, -2.356194490192345, 0.0,
      1.570796326794897, 0.785398163397448;
  return q;
}

CartesianMpc::Weights weights() {
  CartesianMpc::Weights w;
  w.task_weight = (V6() << 1000, 1000, 1000, 100, 100, 100).finished();
  w.input_weight = 1e-3;
  w.posture_weight = 10.0;
  return w;
}

}  // namespace

// A zero reference twist must yield an (almost) exact rest solution: the
// dynamics exclude gravity, so holding the current pose needs ~zero torque.
TEST(CartesianMpc, HoldReturnsGravityFreeRest) {
  CartesianMpc mpc{CartesianMpc::Config{}};
  CartesianMpc::Solution s;
  const std::vector<V6> refs(10, V6::Zero());
  ASSERT_TRUE(mpc.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, weights(), s));
  ASSERT_TRUE(s.success);
  EXPECT_LT((s.q_ref.back() - homeQ()).norm(), 1e-3);
  EXPECT_LT(s.tau_ff[0].norm(), 1e-2);
  EXPECT_LT(s.dq_ref.back().norm(), 1e-3);
}

// A pure +x translation reference (5 cm) should drive joint 0 in the positive
// direction (task dim 0 == joint 0 here) and leave the other joints alone.
TEST(CartesianMpc, TracksTranslationStep) {
  CartesianMpc::Config cfg;
  CartesianMpc mpc{cfg};
  CartesianMpc::Solution s;
  const std::vector<V6> refs(10, (V6() << 0.05, 0, 0, 0, 0, 0).finished());
  ASSERT_TRUE(mpc.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, weights(), s));
  ASSERT_TRUE(s.success);
  EXPECT_GT(s.q_ref.back()(0) - homeQ()(0), 0.015);          // really moving
  EXPECT_LT(std::abs(s.q_ref.back()(1) - homeQ()(1)), 1e-3);  // others fixed
}

// Even an aggressive reference must keep the planned feed-forward torque within
// the margined limits (torque_limit_scale * tau_max).
TEST(CartesianMpc, RespectsTorqueLimits) {
  CartesianMpc::Config cfg;
  CartesianMpc mpc{cfg};
  CartesianMpc::Solution s;
  const std::vector<V6> refs(10, (V6() << 0.5, 0.5, 0.5, 0, 0, 0).finished());
  ASSERT_TRUE(mpc.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, weights(), s));
  ASSERT_TRUE(s.success);
  const double tau_max[7] = {87, 87, 87, 87, 12, 12, 12};
  // The optimum sits exactly ON the margined bound for this reference, so
  // allow solver-level (OSQP eps) noise; downstream the command is still
  // hard-clamped to tau_max in postprocessTauImpl.
  for (const auto& tau : s.tau_ff) {
    for (int i = 0; i < 7; ++i) {
      EXPECT_LE(std::abs(tau(i)), cfg.torque_limit_scale * tau_max[i] + 1e-3);
    }
  }
}

// A reference whose length disagrees with the horizon must be rejected.
TEST(CartesianMpc, RejectsMismatchedReferenceLength) {
  CartesianMpc mpc{CartesianMpc::Config{}};
  CartesianMpc::Solution s;
  const std::vector<V6> refs(3, V6::Zero());  // horizon is 10
  EXPECT_FALSE(mpc.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                         identityJacobian(), refs, weights(), s));
  EXPECT_FALSE(s.success);
}

// Joint 6 is outside the task Jacobian (null space). With it displaced from
// the nominal posture and a hold reference, the posture regularization must
// pull it back toward nominal while the task-space joints stay put.
TEST(CartesianMpc, PostureRegularizationResolvesNullspace) {
  // The posture pull acts along shallow (null-space) directions whose dual
  // residual is attenuated by the dt*Minv dynamics factors; a single COLD
  // solve needs a tight tolerance to converge them (at runtime the warm-start
  // chain does this across cycles instead).
  CartesianMpc::Config cfg;  // q_nominal defaults to homeQ()
  cfg.eps_abs = 1e-6;
  cfg.eps_rel = 1e-6;
  CartesianMpc mpc{cfg};
  CartesianMpc::Solution s;
  V7 q = homeQ();
  q(6) += 0.3;  // null-space displacement
  const std::vector<V6> refs(10, V6::Zero());
  ASSERT_TRUE(mpc.solve(q, V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, weights(), s));
  ASSERT_TRUE(s.success);
  // Moved back toward nominal by a meaningful amount over the 100 ms horizon
  // (the acceleration smoothness term slows the pull slightly vs the
  // accel-free 0.05).
  EXPECT_LT(s.q_ref.back()(6) - homeQ()(6), 0.3 - 0.035);
  EXPECT_GT(s.q_ref.back()(6) - homeQ()(6), 0.0);  // no overshoot past nominal
  // Task-space joints undisturbed.
  EXPECT_LT(std::abs(s.q_ref.back()(1) - q(1)), 1e-3);
}

// Regression for the steady-state bias: the posture cost is PROJECTED into
// the Jacobian's null space, so a task-space joint displaced from q_nominal
// (here joint 1, which maps 1:1 to task dim 1) must NOT be pulled back toward
// nominal when the reference says "hold" - an unprojected posture cost would
// drag it (and thus the EE) toward q_nominal, converging ~2 cm / ~5 deg off
// target.
TEST(CartesianMpc, PostureDoesNotBiasTaskSpace) {
  CartesianMpc::Config cfg;
  cfg.eps_abs = 1e-6;  // single cold solve, probe a fine-grained equilibrium
  cfg.eps_rel = 1e-6;
  CartesianMpc mpc{cfg};
  CartesianMpc::Solution s;
  V7 q = homeQ();
  q(1) += 0.2;  // task-space displacement away from q_nominal
  const std::vector<V6> refs(10, V6::Zero());
  ASSERT_TRUE(mpc.solve(q, V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, weights(), s));
  ASSERT_TRUE(s.success);
  // Stays at the commanded (displaced) pose, not pulled toward nominal.
  EXPECT_LT(std::abs(s.q_ref.back()(1) - q(1)), 1e-3);
}

// Regression for the ~1 Hz ringing around a constant target: the terminal
// ||dq_T||^2 penalty must BOUND the momentum each plan carries into the
// target - near-minimum-time plans would arrive at full cruise and the
// re-solve chain then closes an underdamped loop. A full stop at the horizon
// end is NOT required (that was the removed look-ahead cost's contract, see
// MPC_development_plan.md Stage 1.9): the residual momentum is shed across
// the re-solve chain, measured ~20 mm overshoot / 2.4 s settle on a 10 cm
// step in sim.
TEST(CartesianMpc, PlanBoundsArrivalMomentum) {
  CartesianMpc::Config cfg;
  CartesianMpc mpc{cfg};
  CartesianMpc::Solution s;
  const std::vector<V6> refs(10, (V6() << 0.05, 0, 0, 0, 0, 0).finished());
  ASSERT_TRUE(mpc.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, weights(), s));
  ASSERT_TRUE(s.success);
  // Mid-horizon the plan moves (cruise unconstrained) ...
  EXPECT_GT(s.dq_ref[4].cwiseAbs().maxCoeff(), 0.1);
  // ... but the arrival momentum is bounded ...
  EXPECT_LT(s.dq_ref.back().cwiseAbs().maxCoeff(), 0.5);
  // ... and measurably damped vs the same plan without the terminal penalty.
  CartesianMpc::Config cfg_undamped = cfg;
  cfg_undamped.terminal_dq_weight = 0.0;
  CartesianMpc mpc_undamped{cfg_undamped};
  CartesianMpc::Solution s_undamped;
  ASSERT_TRUE(mpc_undamped.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                                 identityJacobian(), refs, weights(),
                                 s_undamped));
  EXPECT_LT(s.dq_ref.back().cwiseAbs().maxCoeff(),
            0.8 * s_undamped.dq_ref.back().cwiseAbs().maxCoeff());
}

// Velocity feed-forward: refs that RAMP at a constant rate (a moving target
// sampled dt apart) produce a finite-difference reference velocity v_ref that
// the velocity term tracks. The plan must reach cruise and SUSTAIN it through
// the terminal stage (no end-of-horizon sag), and a stronger velocity weight
// must track v_ref more closely than no velocity term at all.
TEST(CartesianMpc, VelocityFeedforwardSustainsCruise) {
  CartesianMpc::Config cfg;
  CartesianMpc mpc{cfg};
  // Target moving at +0.2 m/s in task dim 1 (== joint 1): refs[i] is the pose
  // at t = (i+1)*dt, so the per-stage increment is 0.2*dt.
  std::vector<V6> refs(10, V6::Zero());
  for (int i = 0; i < 10; ++i) {
    refs[i](1) = 0.2 * cfg.dt * static_cast<double>(i + 1);
  }
  CartesianMpc::Weights w = weights();  // default velocity_weight
  CartesianMpc::Solution s;
  ASSERT_TRUE(mpc.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, w, s));
  ASSERT_TRUE(s.success);
  // Reaches cruise mid-horizon and sustains it at the end (the old plain
  // ||dq_T||^2 damper at weight 100 dragged exactly this case).
  EXPECT_GT(s.dq_ref[4](1), 0.1);
  EXPECT_LT(s.dq_ref[4](1), 0.35);
  EXPECT_GT(s.dq_ref.back()(1), 0.1);

  // Comparative: average velocity-tracking error must shrink when the
  // velocity term is strengthened vs disabled.
  const auto mean_v_err = [&](const CartesianMpc::Solution& sol) {
    double e = 0.0;
    for (const auto& dq : sol.dq_ref) {
      e += std::abs(dq(1) - 0.2);
    }
    return e / static_cast<double>(sol.dq_ref.size());
  };
  CartesianMpc::Weights w_off = w;
  w_off.velocity_weight = 0.0;
  CartesianMpc mpc_off{cfg};
  CartesianMpc::Solution s_off;
  ASSERT_TRUE(mpc_off.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                            identityJacobian(), refs, w_off, s_off));
  CartesianMpc::Weights w_hi = w;
  w_hi.velocity_weight = 0.1;
  CartesianMpc mpc_hi{cfg};
  CartesianMpc::Solution s_hi;
  ASSERT_TRUE(mpc_hi.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                           identityJacobian(), refs, w_hi, s_hi));
  EXPECT_LT(mean_v_err(s_hi), mean_v_err(s_off));
}

// Held target (constant refs) -> v_ref is identically zero by construction,
// so the velocity term must act as pure damping and NOT bias the converged
// pose (a non-zero v_ref here would steadily push the arm off target).
TEST(CartesianMpc, HeldTargetHasZeroVelocityReference) {
  CartesianMpc::Config cfg;
  cfg.eps_abs = 1e-6;
  cfg.eps_rel = 1e-6;
  CartesianMpc mpc{cfg};
  CartesianMpc::Weights w = weights();
  w.velocity_weight = 0.1;  // exaggerate: any v_ref leak would show up
  CartesianMpc::Solution s;
  const std::vector<V6> refs(10, V6::Zero());
  ASSERT_TRUE(mpc.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, w, s));
  ASSERT_TRUE(s.success);
  EXPECT_LT((s.q_ref.back() - homeQ()).norm(), 1e-3);
  EXPECT_LT(s.dq_ref.back().norm(), 1e-3);
}

// The acceleration cost must smooth the plan: for the same step reference, a
// higher accel_weight must reduce the peak joint acceleration while the plan
// still makes progress toward the target.
TEST(CartesianMpc, AccelWeightSmoothsPlan) {
  CartesianMpc::Config cfg;
  const std::vector<V6> refs(10, (V6() << 0.05, 0, 0, 0, 0, 0).finished());
  const auto max_ddq = [&](const CartesianMpc::Solution& sol) {
    double m = 0.0;
    V7 prev = V7::Zero();  // dq_0 = 0 (start at rest)
    for (const auto& dq : sol.dq_ref) {
      m = std::max(m, ((dq - prev) / cfg.dt).cwiseAbs().maxCoeff());
      prev = dq;
    }
    return m;
  };
  CartesianMpc::Weights w_lo = weights();
  w_lo.accel_weight = 0.0;
  CartesianMpc mpc_lo{cfg};
  CartesianMpc::Solution s_lo;
  ASSERT_TRUE(mpc_lo.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                           identityJacobian(), refs, w_lo, s_lo));
  CartesianMpc::Weights w_hi = weights();
  w_hi.accel_weight = 1e-2;
  CartesianMpc mpc_hi{cfg};
  CartesianMpc::Solution s_hi;
  ASSERT_TRUE(mpc_hi.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                           identityJacobian(), refs, w_hi, s_hi));
  EXPECT_LT(max_ddq(s_hi), max_ddq(s_lo));
  // Still moves toward the target despite the smoothing.
  EXPECT_GT(s_hi.q_ref.back()(0) - homeQ()(0), 0.01);
}

// The trust region must cap the per-solve joint excursion so the frozen
// linearization stays valid, even for a huge (0.5 m) reference step.
TEST(CartesianMpc, TrustRegionBoundsExcursion) {
  CartesianMpc::Config cfg;
  CartesianMpc mpc{cfg};
  CartesianMpc::Solution s;
  const std::vector<V6> refs(10, (V6() << 0.5, 0.5, 0.5, 0, 0, 0).finished());
  const V7 q0 = homeQ();
  ASSERT_TRUE(mpc.solve(q0, V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, weights(), s));
  ASSERT_TRUE(s.success);
  for (const auto& q : s.q_ref) {
    EXPECT_LE((q - q0).cwiseAbs().maxCoeff(), cfg.trust_region + 1e-3);
  }
}

// A state that has already drifted past a margined position bound used to make
// the QP primal-infeasible -> solve failure -> permanent hold at the limit
// (joint lock). With soft bounds the solve must succeed and steer the joint
// back inside the margined range.
TEST(CartesianMpc, RecoversFromMarginViolation) {
  CartesianMpc::Config cfg;
  CartesianMpc mpc{cfg};
  CartesianMpc::Solution s;
  V7 q = homeQ();
  // Inside the hard limit (-2.8973) but 0.05 rad past the margined bound.
  q(0) = -2.8973 + cfg.joint_position_margin - 0.05;
  const std::vector<V6> refs(10, V6::Zero());
  ASSERT_TRUE(mpc.solve(q, V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, weights(), s));
  ASSERT_TRUE(s.success);
  // Steers back toward the margined bound, not further out (full recovery
  // happens across the re-solve chain).
  EXPECT_GT(s.q_ref.back()(0), q(0) + 0.008);
  EXPECT_GT(s.dq_ref.back()(0), 0.0);  // still moving inward at horizon end
}
