#include <multi_mode_controller/utils/cartesian_mpc.h>

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
  if (T <= 0 || static_cast<int>(refs.size()) != T) {
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

  // Box bounds: one identity row per decision variable.
  for (int v = 0; v < n; ++v) {
    a_triplets.emplace_back(n_eq + v, v, 1.0);
  }
  for (int k = 1; k <= T; ++k) {
    lower.segment(n_eq + ix(k), kNq) = panda_limits::q_min;
    upper.segment(n_eq + ix(k), kNq) = panda_limits::q_max;
    lower.segment(n_eq + ix(k) + kNq, kNq) = -panda_limits::qD_max;
    upper.segment(n_eq + ix(k) + kNq, kNq) = panda_limits::qD_max;
  }
  for (int k = 0; k < T; ++k) {
    lower.segment(n_eq + iu(k), kNu) = -panda_limits::tau_max;
    upper.segment(n_eq + iu(k), kNu) = panda_limits::tau_max;
  }

  Eigen::SparseMatrix<double> constraints(m, n);
  constraints.setFromTriplets(a_triplets.begin(), a_triplets.end());

  // ----- Cost: 0.5 z' P z + g' z -----
  // Tracking: sum_k (J q_k - b_k)' W_k (J q_k - b_k), b_k = J q0 + r_k.
  //   -> P block on q_k += 2 J' W_k J ; g block on q_k += -2 J' W_k b_k
  // Regularization: dq_weight ||dq_k||^2 and input_weight ||u_k||^2.
  std::vector<Triplet> p_triplets;
  p_triplets.reserve(T * (kNq * kNq + kNq + kNu));
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(n);

  const Eigen::DiagonalMatrix<double, 6> W = weights.task_weight.asDiagonal();
  for (int k = 1; k <= T; ++k) {
    const double scale = (k == T) ? config_.terminal_scale : 1.0;
    const Matrix7d JtWJ = 2.0 * scale * (jacobian.transpose() * (W * jacobian));
    const Vector6d b_k = jacobian * q0 + refs[k - 1];
    const Vector7d g_q = -2.0 * scale * (jacobian.transpose() * (W * b_k));
    // Upper triangle of the (symmetric) q_k block.
    for (int i = 0; i < kNq; ++i) {
      for (int j = i; j < kNq; ++j) {
        p_triplets.emplace_back(ix(k) + i, ix(k) + j, JtWJ(i, j));
      }
    }
    gradient.segment(ix(k), kNq) += g_q;
    // dq_k velocity regularization (diagonal).
    for (int i = 0; i < kNq; ++i) {
      p_triplets.emplace_back(ix(k) + kNq + i, ix(k) + kNq + i,
                              2.0 * config_.dq_weight);
    }
  }
  for (int k = 0; k < T; ++k) {
    for (int i = 0; i < kNu; ++i) {
      p_triplets.emplace_back(iu(k) + i, iu(k) + i, 2.0 * weights.input_weight);
    }
  }

  Eigen::SparseMatrix<double> hessian(n, n);
  hessian.setFromTriplets(p_triplets.begin(), p_triplets.end());

  // ----- Solve (fresh solver instance: simplest robust path for Stage 1) -----
  OsqpEigen::Solver solver;
  solver.settings()->setWarmStart(false);
  solver.settings()->setVerbosity(false);
  solver.settings()->setMaxIteration(config_.max_iteration);
  // The problem is poorly scaled (constraint entries span 1 down to
  // dt^2*Minv ~ 1e-5). A moderate ADMM tolerance keeps convergence fast on hard
  // instances (large references with binding velocity limits); POLISH then does
  // an exact active-set solve to sharpen the result - so the hold case is
  // accurate AND hard QPs don't hit the iteration cap. (Tightening ADMM to 1e-6
  // instead caused MAX_ITER_REACHED on large-rotation references.)
  solver.settings()->setAbsoluteTolerance(1e-4);
  solver.settings()->setRelativeTolerance(1e-4);
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
    return false;
  }
  if (!solver.initSolver()) {
    return false;
  }
  if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) {
    return false;
  }
  const OsqpEigen::Status status = solver.getStatus();
  if (status != OsqpEigen::Status::Solved &&
      status != OsqpEigen::Status::SolvedInaccurate) {
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
