#include <multi_mode_controller/utils/cartesian_mpc.h>

#include <algorithm>
#include <string>

#include <Eigen/Sparse>
#include <OsqpEigen/OsqpEigen.h>

#include <multi_mode_controller/utils/panda_limits.h>

namespace panda_controllers {

namespace {
constexpr int kNx = 14;  // state dimension [q; dq]
constexpr int kNu = 7;   // input dimension [tau]
constexpr int kNq = 7;   // joints
using Triplet = Eigen::Triplet<double>;
}  // namespace

CartesianMpc::CartesianMpc(const Config& config) : config_(config) {}

bool CartesianMpc::solve(const Vector7d& q0,
                         const Vector7d& dq0,
                         const Matrix7d& mass,
                         const Vector7d& coriolis,
                         const Matrix67d& jacobian,
                         const std::vector<Vector6d>& refs,
                         const Weights& weights,
                         Solution& solution) {
  const int T = config_.horizon;
  solution.success = false;
  last_failure_.clear();
  if (T <= 0 || static_cast<int>(refs.size()) != T) {
    last_failure_ = "bad horizon / reference length";
    return false;
  }
  if (!q0.allFinite() || !dq0.allFinite() || !mass.allFinite() ||
      !coriolis.allFinite() || !jacobian.allFinite()) {
    last_failure_ = "non-finite input data";
    return false;
  }

  const int n = T * kNx + T * kNu;       // decision variables
  const int n_eq = T * kNx;              // dynamics equalities
  const int m = n_eq + n;                // equalities + box bounds

  // Variable block offsets.
  // State x_k (k = 1..T) starts at (k-1)*kNx; q at +0, dq at +kNq.
  // Input u_k (k = 0..T-1) starts at T*kNx + k*kNu.
  const auto ix = [](int k) { return (k - 1) * kNx; };
  const auto iu = [&](int k) { return T * kNx + k * kNu; };

  // ----- Frozen discrete-time dynamics: x_{k+1} = A x_k + B u_k + a -----
  const Matrix7d minv = mass.llt().solve(Matrix7d::Identity());
  const Vector7d a0 = -minv * coriolis;  // gravity excluded (auto-compensated)
  const double dt = config_.dt;
  const double dt2 = dt * dt;

  // A = [[I, dt I], [0, I]]  (kNx x kNx)
  // B = [[dt^2 Minv], [dt Minv]]  (kNx x kNu)
  // a = [[dt^2 a0], [dt a0]]  (kNx)
  Eigen::Matrix<double, kNx, kNx> A_dyn = Eigen::Matrix<double, kNx, kNx>::Zero();
  A_dyn.topLeftCorner(kNq, kNq).setIdentity();
  A_dyn.topRightCorner(kNq, kNq) = dt * Matrix7d::Identity();
  A_dyn.bottomRightCorner(kNq, kNq).setIdentity();

  Eigen::Matrix<double, kNx, kNu> B_dyn;
  B_dyn.topRows(kNq) = dt2 * minv;
  B_dyn.bottomRows(kNq) = dt * minv;

  Eigen::Matrix<double, kNx, 1> a_dyn;
  a_dyn.head(kNq) = dt2 * a0;
  a_dyn.tail(kNq) = dt * a0;

  Eigen::Matrix<double, kNx, 1> x0;
  x0.head(kNq) = q0;
  x0.tail(kNq) = dq0;

  // ----- Constraint matrix and bounds -----
  std::vector<Triplet> a_triplets;
  a_triplets.reserve(T * (kNx + kNx * kNx + kNx * kNu) + n);
  Eigen::VectorXd lower(m);
  Eigen::VectorXd upper(m);

  for (int k = 0; k < T; ++k) {
    const int row = k * kNx;
    // +I on x_{k+1}
    for (int i = 0; i < kNx; ++i) {
      a_triplets.emplace_back(row + i, ix(k + 1) + i, 1.0);
    }
    // -B on u_k
    for (int i = 0; i < kNx; ++i) {
      for (int j = 0; j < kNu; ++j) {
        a_triplets.emplace_back(row + i, iu(k) + j, -B_dyn(i, j));
      }
    }
    Eigen::Matrix<double, kNx, 1> rhs = a_dyn;
    if (k == 0) {
      rhs += A_dyn * x0;  // x_0 is a known constant, folded into the RHS
    } else {
      // -A on x_k
      for (int i = 0; i < kNx; ++i) {
        for (int j = 0; j < kNx; ++j) {
          a_triplets.emplace_back(row + i, ix(k) + j, -A_dyn(i, j));
        }
      }
    }
    lower.segment(row, kNx) = rhs;
    upper.segment(row, kNx) = rhs;
  }

  // State bounds: one identity row per decision variable. The official Panda
  // limits (panda_limits.h) are the hard fault thresholds, so apply margins;
  // additionally intersect the q box with the trust region around q0 (the
  // region where the frozen linearization stays accurate).
  const Vector7d q_lim_lo = panda_limits::q_min.array() + config_.joint_position_margin;
  const Vector7d q_lim_hi = panda_limits::q_max.array() - config_.joint_position_margin;
  const Vector7d trust = Vector7d::Constant(config_.trust_region);
  const Vector7d q_lo = q_lim_lo.cwiseMax(q0 - trust);
  const Vector7d q_hi = q_lim_hi.cwiseMin(q0 + trust);
  const Vector7d dq_hi = config_.velocity_limit_scale * panda_limits::qD_max;
  const Vector7d tau_hi = config_.torque_limit_scale * panda_limits::tau_max;

  // Feasibility guarantee (joint-lock robustness): roll out a BRAKING
  // trajectory under the frozen dynamics - the torque that drives dq to zero
  // as fast as the (margined) torque limits allow - and widen each stage's
  // state box just enough to contain it. In normal operation this is a no-op;
  // when the measured state has already drifted past a margined bound (e.g.
  // into the position-margin band, or pushed over a velocity limit) the QP
  // then still has a feasible point ("brake") instead of returning
  // PRIMAL_INFEASIBLE forever, which would freeze the controller at the
  // limit. The recovery cost below steers back into the valid range.
  constexpr double kWidenEps = 1e-6;
  std::vector<Eigen::Matrix<double, kNx, 1>> x_brake(T);
  {
    Eigen::Matrix<double, kNx, 1> x_k = x0;
    for (int k = 0; k < T; ++k) {
      const Vector7d tau_brake =
          (-(mass * x_k.tail(kNq)) / dt + coriolis).cwiseMax(-tau_hi).cwiseMin(tau_hi);
      x_k = A_dyn * x_k + B_dyn * tau_brake + a_dyn;
      x_brake[k] = x_k;
    }
  }

  for (int v = 0; v < n; ++v) {
    a_triplets.emplace_back(n_eq + v, v, 1.0);
  }
  for (int k = 1; k <= T; ++k) {
    const auto& brk = x_brake[k - 1];
    for (int i = 0; i < kNq; ++i) {
      lower(n_eq + ix(k) + i) = std::min(q_lo(i), brk(i) - kWidenEps);
      upper(n_eq + ix(k) + i) = std::max(q_hi(i), brk(i) + kWidenEps);
      lower(n_eq + ix(k) + kNq + i) = std::min(-dq_hi(i), brk(kNq + i) - kWidenEps);
      upper(n_eq + ix(k) + kNq + i) = std::max(dq_hi(i), brk(kNq + i) + kWidenEps);
    }
  }
  // Hard torque box (inputs are free variables: never a feasibility risk).
  for (int k = 0; k < T; ++k) {
    lower.segment(n_eq + iu(k), kNu) = -tau_hi;
    upper.segment(n_eq + iu(k), kNu) = tau_hi;
  }

  Eigen::SparseMatrix<double> constraints(m, n);
  constraints.setFromTriplets(a_triplets.begin(), a_triplets.end());

  // ----- Cost: 0.5 z' P z + g' z -----
  // Tracking: sum_k (J q_k - b_k)' W_k (J q_k - b_k), b_k = J q0 + r_k.
  //   -> P block on q_k += 2 J' W_k J ; g block on q_k += -2 J' W_k b_k
  // Velocity tracking: velocity_weight (J dq_k - v_k)' W (J dq_k - v_k) with
  //   v_k the REFERENCE velocity, estimated by finite differences of the raw
  //   reference twists (the refs are poses sampled dt apart along the
  //   commanded trajectory, so their differences ARE the interpolated target
  //   velocity). A held / single-waypoint target gives constant refs ->
  //   v_k == 0 and the term is pure task-space damping; along a moving
  //   trajectory it feeds the target's velocity forward instead of dragging
  //   on it (which is what made a plain ||dq||^2 damper lag moving targets).
  //   -> P block on dq_k += 2 vw J' W J ; g block on dq_k += -2 vw J' W v_k
  // Acceleration smoothness: accel_weight ||Minv (u_k - c)||^2 (the frozen
  //   forward dynamics, gravity excluded), quadratic in the input:
  //   -> P block on u_k += 2 aw Minv^2 ; g block on u_k += -2 aw Minv^2 c.
  // Posture: posture_weight ||N (q_k - q_nominal)||^2 with N = I - pinv(J) J
  //   the null-space projector of the frozen Jacobian (damped pseudoinverse).
  //   The projection confines the pull to the redundancy direction; an
  //   unprojected posture cost fights the task cost and biases the converged
  //   EE pose (~2 cm / ~5 deg at poses far from q_nominal). Near a
  //   singularity the damping shrinks pinv(J), so the (near-)uncontrollable
  //   directions fall back into N and get regularized too - exactly when
  //   that is wanted.
  // Recovery: recovery_weight (q_k_i - q_safe_i)^2 for joints measured
  //   outside their margined position limit, pulling back just inside.
  // Regularization: dq_weight ||dq_k||^2 and input_weight ||u_k||^2.
  // Closed-loop damping (see Config): the velocity-tracking term above (pure
  //   damping for a held target, where v_k == 0) plus the terminal
  //   terminal_dq_weight ||dq_T||^2 penalty, which makes each plan end near
  //   rest. Without these the re-solve chain closes an underdamped loop that
  //   rings around a constant target at ~1 Hz.
  // The reference twists are clamped (max_ref_*) so far-away targets do not
  // produce outsized Gauss-Newton steps the frozen linearization cannot
  // honor; the 100 Hz re-solve turns them into a bounded-speed approach.
  std::vector<Triplet> p_triplets;
  p_triplets.reserve(T * (2 * kNq * kNq + 3 * kNq + kNu * kNu));
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(n);

  // Recovery target/weight per joint (zero weight when inside the limits).
  Vector7d recovery_w = Vector7d::Zero();
  Vector7d q_safe = Vector7d::Zero();
  for (int i = 0; i < kNq; ++i) {
    if (q0(i) < q_lim_lo(i)) {
      recovery_w(i) = config_.recovery_weight;
      q_safe(i) = q_lim_lo(i) + config_.recovery_backoff;
    } else if (q0(i) > q_lim_hi(i)) {
      recovery_w(i) = config_.recovery_weight;
      q_safe(i) = q_lim_hi(i) - config_.recovery_backoff;
    }
  }

  const Eigen::DiagonalMatrix<double, 6> W = weights.task_weight.asDiagonal();
  const Eigen::Matrix<double, 7, 6> JtW = jacobian.transpose() * W;
  const Matrix7d JtWJ_base = JtW * jacobian;

  // Clamp the reference twists (max_ref_*: a far target must not inject
  // outsized Gauss-Newton steps) and estimate the per-stage reference
  // VELOCITY by finite differences of the RAW refs - forward difference,
  // repeated for the final stage. Constant refs (held target / empty queue)
  // give v_ref == 0 exactly. v_ref is clamped separately
  // (max_ref_velocity_*): a target jump looks like a huge one-command_dt
  // velocity that the arm cannot follow.
  std::vector<Vector6d> ref_clamped(T);
  std::vector<Vector6d> v_ref(T, Vector6d::Zero());
  for (int i = 0; i < T; ++i) {
    Vector6d ref = refs[i];
    const double t_norm = ref.head(3).norm();
    if (t_norm > config_.max_ref_translation) {
      ref.head(3) *= config_.max_ref_translation / t_norm;
    }
    const double r_norm = ref.tail(3).norm();
    if (r_norm > config_.max_ref_rotation) {
      ref.tail(3) *= config_.max_ref_rotation / r_norm;
    }
    ref_clamped[i] = ref;
  }
  if (T > 1) {
    for (int i = 0; i < T; ++i) {
      const int hi = std::min(i + 1, T - 1);
      Vector6d v = (refs[hi] - refs[hi - 1]) / dt;
      const double tv_norm = v.head(3).norm();
      if (tv_norm > config_.max_ref_velocity_translation) {
        v.head(3) *= config_.max_ref_velocity_translation / tv_norm;
      }
      const double rv_norm = v.tail(3).norm();
      if (rv_norm > config_.max_ref_velocity_rotation) {
        v.tail(3) *= config_.max_ref_velocity_rotation / rv_norm;
      }
      v_ref[i] = v;
    }
  }
  // Null-space projector of the frozen Jacobian (damped pseudoinverse; the
  // damping only matters near singularities, see the cost comment above).
  constexpr double kPinvDamping = 1e-6;
  const Eigen::Matrix<double, 7, 6> j_pinv =
      jacobian.transpose() *
      (jacobian * jacobian.transpose() + kPinvDamping * Eigen::Matrix<double, 6, 6>::Identity())
          .llt()
          .solve(Eigen::Matrix<double, 6, 6>::Identity());
  const Matrix7d null_proj = Matrix7d::Identity() - j_pinv * jacobian;
  // N'N (not N: with damping N is only approximately idempotent, and N'N is
  // PSD by construction).
  const Matrix7d posture_h = null_proj.transpose() * null_proj;
  const Vector7d posture_g = posture_h * config_.q_nominal;
  for (int k = 1; k <= T; ++k) {
    const double scale = (k == T) ? config_.terminal_scale : 1.0;
    const Matrix7d JtWJ = 2.0 * scale * JtWJ_base;
    const Vector6d b_k = jacobian * q0 + ref_clamped[k - 1];
    const Vector7d g_q = -2.0 * scale * (JtW * b_k);
    // Upper triangle of the (symmetric) q_k block: tracking + projected
    // posture (both dense); recovery is diagonal (setFromTriplets sums
    // duplicates).
    const Matrix7d q_block = JtWJ + 2.0 * weights.posture_weight * posture_h;
    for (int i = 0; i < kNq; ++i) {
      for (int j = i; j < kNq; ++j) {
        p_triplets.emplace_back(ix(k) + i, ix(k) + j, q_block(i, j));
      }
      p_triplets.emplace_back(ix(k) + i, ix(k) + i, 2.0 * recovery_w(i));
    }
    gradient.segment(ix(k), kNq) +=
        g_q - 2.0 * weights.posture_weight * posture_g -
        2.0 * recovery_w.cwiseProduct(q_safe);
    // dq_k velocity regularization (diagonal); the final stage additionally
    // carries the terminal_dq_weight end-near-rest penalty.
    const double dq_w = config_.dq_weight +
                        ((k == T) ? config_.terminal_dq_weight : 0.0);
    for (int i = 0; i < kNq; ++i) {
      p_triplets.emplace_back(ix(k) + kNq + i, ix(k) + kNq + i, 2.0 * dq_w);
    }
    // Task-space velocity tracking toward v_ref (dense dq block; skipped at
    // weight 0 to keep the Hessian sparse).
    if (weights.velocity_weight > 0.0) {
      const Matrix7d dq_block = 2.0 * weights.velocity_weight * JtWJ_base;
      for (int i = 0; i < kNq; ++i) {
        for (int j = i; j < kNq; ++j) {
          p_triplets.emplace_back(ix(k) + kNq + i, ix(k) + kNq + j,
                                  dq_block(i, j));
        }
      }
      gradient.segment(ix(k) + kNq, kNq) +=
          -2.0 * weights.velocity_weight * (JtW * v_ref[k - 1]);
    }
  }
  // Input regularization (diagonal) + acceleration smoothness (dense in u_k:
  // ddq_k = Minv (u_k - c), so the Hessian block is 2 aw Minv^2; Minv is
  // symmetric, hence Minv'Minv = Minv^2). Skipped at weight 0 for sparsity.
  const Matrix7d minv2 = minv * minv;
  const Vector7d accel_g = minv2 * coriolis;
  for (int k = 0; k < T; ++k) {
    for (int i = 0; i < kNu; ++i) {
      p_triplets.emplace_back(iu(k) + i, iu(k) + i, 2.0 * weights.input_weight);
    }
    if (weights.accel_weight > 0.0) {
      for (int i = 0; i < kNu; ++i) {
        for (int j = i; j < kNu; ++j) {
          p_triplets.emplace_back(iu(k) + i, iu(k) + j,
                                  2.0 * weights.accel_weight * minv2(i, j));
        }
      }
      gradient.segment(iu(k), kNu) += -2.0 * weights.accel_weight * accel_g;
    }
  }

  Eigen::SparseMatrix<double> hessian(n, n);
  hessian.setFromTriplets(p_triplets.begin(), p_triplets.end());

  // ----- Solve (fresh solver instance: simplest robust path for Stage 1) -----
  OsqpEigen::Solver solver;
  // Must be true or osqp_solve() cold-starts (zeroes) the iterate that
  // setPrimalVariable/setDualVariable seed below.
  solver.settings()->setWarmStart(true);
  solver.settings()->setVerbosity(false);
  solver.settings()->setMaxIteration(config_.max_iteration);
  // The problem is poorly scaled: the dynamics rows carry dt*Minv / dt^2*Minv
  // factors (~1e-2..1e-5) that ATTENUATE the dual residual of the inputs and
  // of null-space directions. A tight tolerance (1e-6) would converge those
  // shallow directions in one cold solve, but on real-robot instances with a
  // moving target it exceeds max_iteration. So: moderate eps (Config) +
  // POLISH (exact active-set refinement; reliable here because the constraint
  // rows are plain per-variable boxes, no degenerate slack constructions) +
  // warm-start chaining across the 100 Hz re-solves, which keeps refining the
  // same shallow directions instead of restarting ADMM from cold.
  solver.settings()->setAbsoluteTolerance(config_.eps_abs);
  solver.settings()->setRelativeTolerance(config_.eps_rel);
  solver.settings()->setScaledTerimination(false);  // sic: osqp-eigen API typo
  solver.settings()->setPolish(true);
  solver.settings()->setPolishRefineIter(5);
  solver.data()->setNumberOfVariables(n);
  solver.data()->setNumberOfConstraints(m);
  if (!solver.data()->setHessianMatrix(hessian) ||
      !solver.data()->setGradient(gradient) ||
      !solver.data()->setLinearConstraintsMatrix(constraints) ||
      !solver.data()->setLowerBound(lower) ||
      !solver.data()->setUpperBound(upper)) {
    last_failure_ = "OSQP data setup failed";
    return false;
  }
  if (!solver.initSolver()) {
    last_failure_ = "OSQP init failed";
    return false;
  }
  // Warm-start from the previous solve's iterate: consecutive QPs (10 ms
  // apart) are near-identical, which cuts the ADMM iteration count by orders
  // of magnitude vs a cold start at eps 1e-6.
  if (warm_valid_ && z_warm_.size() == n && y_warm_.size() == m) {
    solver.setPrimalVariable(z_warm_);
    solver.setDualVariable(y_warm_);
  }
  const OsqpEigen::ErrorExitFlag exit_flag = solver.solveProblem();
  if (const auto& ws = solver.workspace(); ws && ws->info) {
    last_stats_.iterations = static_cast<int>(ws->info->iter);
    last_stats_.run_time = ws->info->run_time;
    last_stats_.polish_status = static_cast<int>(ws->info->status_polish);
  }
  if (exit_flag != OsqpEigen::ErrorExitFlag::NoError) {
    last_failure_ = "OSQP solve error";
    warm_valid_ = false;
    return false;
  }
  const OsqpEigen::Status status = solver.getStatus();
  // Keep the final iterate as the next warm start even when this solve hit
  // the iteration cap (the iterate is still the best approximation so far);
  // drop it on infeasible/non-convex statuses, where it is meaningless.
  if (status == OsqpEigen::Status::Solved ||
      status == OsqpEigen::Status::SolvedInaccurate ||
      status == OsqpEigen::Status::MaxIterReached) {
    z_warm_ = solver.getSolution();
    y_warm_ = solver.getDualSolution();
    warm_valid_ = true;
  } else {
    warm_valid_ = false;
  }
  if (status != OsqpEigen::Status::Solved &&
      status != OsqpEigen::Status::SolvedInaccurate) {
    switch (status) {
      case OsqpEigen::Status::MaxIterReached:
        last_failure_ = "max iterations reached";
        break;
      case OsqpEigen::Status::PrimalInfeasible:
      case OsqpEigen::Status::PrimalInfeasibleInaccurate:
        last_failure_ = "primal infeasible";
        break;
      case OsqpEigen::Status::DualInfeasible:
      case OsqpEigen::Status::DualInfeasibleInaccurate:
        last_failure_ = "dual infeasible";
        break;
      case OsqpEigen::Status::NonCvx:
        last_failure_ = "non-convex";
        break;
      default:
        last_failure_ = "OSQP status " +
                        std::to_string(static_cast<int>(status));
        break;
    }
    return false;
  }

  const Eigen::VectorXd z = solver.getSolution();
  solution.q_ref.resize(T);
  solution.dq_ref.resize(T);
  solution.tau_ff.resize(T);
  for (int k = 1; k <= T; ++k) {
    solution.q_ref[k - 1] = z.segment(ix(k), kNq);
    solution.dq_ref[k - 1] = z.segment(ix(k) + kNq, kNq);
  }
  for (int k = 0; k < T; ++k) {
    solution.tau_ff[k] = z.segment(iu(k), kNu);
  }
  solution.success = true;
  return true;
}

}  // namespace panda_controllers
