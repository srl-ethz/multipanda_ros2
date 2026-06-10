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

V7 homeQ() {
  V7 q;
  q << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
  return q;
}

CartesianMpc::Weights weights() {
  CartesianMpc::Weights w;
  w.task_weight = (V6() << 1000, 1000, 1000, 100, 100, 100).finished();
  w.input_weight = 1e-3;
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
  CartesianMpc mpc{CartesianMpc::Config{}};
  CartesianMpc::Solution s;
  const std::vector<V6> refs(10, (V6() << 0.05, 0, 0, 0, 0, 0).finished());
  ASSERT_TRUE(mpc.solve(homeQ(), V7::Zero(), testMass(), V7::Zero(),
                        identityJacobian(), refs, weights(), s));
  ASSERT_TRUE(s.success);
  EXPECT_GT(s.q_ref.back()(0) - homeQ()(0), 0.03);          // moved toward +x
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
  for (const auto& tau : s.tau_ff) {
    for (int i = 0; i < 7; ++i) {
      EXPECT_LE(std::abs(tau(i)), cfg.torque_limit_scale * tau_max[i] + 1e-6);
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
