#include <multi_mode_controller/controllers/comless_panda_mpc_controller.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include <multi_mode_controller/utils/controller_factory.h>
#include <multi_mode_controller/utils/panda_limits.h>

using namespace panda_controllers;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Pose = PandaMpcControllerPose;
using Params = PandaMpcControllerParams;
using Controller = ComlessPandaMpcController;

namespace {
constexpr double kSolvePeriod = 0.01;   // s, ~100 Hz worker rate
constexpr double kStaleTimeout = 0.05;  // s, drop to hold if solution older
constexpr double kDeltaTauMax = 1.0;    // Nm, per-cycle torque rate limit
// Upper bound on buffered waypoints. Each command extends the buffer (per the
// spec), so this bounds memory / look-ahead under high-rate republishing.
// ~200 s of trajectory at the default 0.1 s spacing - ample for one sequence.
constexpr std::size_t kMaxWaypoints = 2000;

double steadyNow() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

Eigen::Vector3d rotationVector(const Eigen::Matrix3d& rotation) {
  const Eigen::AngleAxisd aa(rotation);
  return aa.axis() * aa.angle();
}
}  // namespace

static auto registration =
    ControllerFactory::registerClass<Controller>("comless_panda_mpc_controller");

Controller::ComlessPandaMpcController()
    : mpc_(CartesianMpc::Config{}) {}

Controller::~ComlessPandaMpcController() {
  stopThread();
}

Params Controller::defaultParameters() {
  Params p;
  p.kp = (Vector7d() << 400, 400, 400, 400, 150, 100, 40).finished();
  p.kd = (Vector7d() << 40, 40, 40, 40, 15, 10, 5).finished();
  p.task_weight = (Vector6d() << 1000, 1000, 1000, 100, 100, 100).finished();
  p.input_weight = 1e-3;
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
  return p;
}

void Controller::startImpl() {
  {
    std::lock_guard<std::mutex> lock(snapshot_mutex_);
    snapshot_.valid = false;
  }
  {
    std::lock_guard<std::mutex> lock(solution_mutex_);
    solution_.valid = false;
  }
  active_solution_.valid = false;
  clearWaypoints();
  if (!robot_data_.empty()) {
    hold_q_ = Eigen::Map<const Vector7d>(robot_data_[0]->state().q.data());
  }
  startThread();
}

void Controller::stopImpl() {
  stopThread();
  std::lock_guard<std::mutex> lock(solution_mutex_);
  solution_.valid = false;
}

void Controller::computeTauImpl(const std::vector<std::array<double, 7>*>& tau,
                                const Pose& /*desired*/, const Params& p) {
  if (tau.empty() || tau[0] == nullptr || robot_data_.empty()) {
    return;
  }

  Eigen::Map<const Vector7d> q(robot_data_[0]->state().q.data());
  Eigen::Map<const Vector7d> dq(robot_data_[0]->state().dq.data());

  // Publish a fresh state snapshot for the worker (non-blocking).
  if (snapshot_mutex_.try_lock()) {
    snapshot_.q = q;
    snapshot_.dq = dq;
    snapshot_.mass = robot_data_[0]->mass();
    snapshot_.coriolis = robot_data_[0]->coriolis();
    snapshot_.jacobian = robot_data_[0]->eeZeroJacobian();
    Eigen::Affine3d transform(
        Eigen::Matrix4d::Map(robot_data_[0]->state().O_T_EE.data()));
    snapshot_.position = transform.translation();
    snapshot_.rotation = transform.linear();
    snapshot_.valid = true;
    snapshot_mutex_.unlock();
  }

  // Refresh the RT-private cached solution (non-blocking).
  if (solution_mutex_.try_lock()) {
    if (solution_.valid) {
      active_solution_ = solution_;
    }
    solution_mutex_.unlock();
  }

  const double now = steadyNow();
  Vector7d q_ref;
  Vector7d dq_ref = Vector7d::Zero();
  Vector7d tau_ff = Vector7d::Zero();
  const bool fresh = active_solution_.valid &&
                     !active_solution_.q_ref.empty() &&
                     (now - active_solution_.solve_time) < kStaleTimeout;
  if (fresh) {
    const int T = static_cast<int>(active_solution_.q_ref.size());
    int idx = static_cast<int>(
        std::floor((now - active_solution_.solve_time) / active_solution_.dt));
    idx = std::clamp(idx, 0, T - 1);
    q_ref = active_solution_.q_ref[idx];
    dq_ref = active_solution_.dq_ref[idx];
    tau_ff = active_solution_.tau_ff[idx];
    hold_q_ = q_ref;  // remember last tracked reference for the hold fallback
  } else {
    q_ref = hold_q_;  // MPC stale/failed: hold the last reference, damp velocity
  }

  // 1 kHz joint-space tracking PD around the MPC feed-forward torque. Gravity is
  // auto-compensated by the hardware layer; tau_ff already contains Coriolis.
  Vector7d tau_d =
      tau_ff + p.kp.cwiseProduct(q_ref - q) + p.kd.cwiseProduct(dq_ref - dq);
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
    // Rate-limit relative to the previously commanded torque.
    const double delta_tau = (*tau[0])[i] - tau_J_d[i];
    double tau_cmd = tau_J_d[i] + std::clamp(delta_tau, -kDeltaTauMax, kDeltaTauMax);
    // Saturate to the joint torque limits.
    tau_cmd = std::clamp(tau_cmd, -panda_limits::tau_max[i], panda_limits::tau_max[i]);
    (*tau[0])[i] = tau_cmd;
  }
}

// ---------------------------------------------------------------------------
// Waypoint buffer
// ---------------------------------------------------------------------------
void Controller::setCommandDt(double command_dt) {
  std::lock_guard<std::mutex> lock(waypoint_mutex_);
  if (command_dt > 0.0) {
    command_dt_ = command_dt;
  }
}

void Controller::clearWaypoints() {
  std::lock_guard<std::mutex> lock(waypoint_mutex_);
  waypoints_.clear();
}

void Controller::appendWaypoints(
    const std::vector<std::pair<Eigen::Vector3d, Eigen::Quaterniond>>& poses) {
  if (poses.empty()) {
    return;
  }
  const double now = steadyNow();
  bool dropped = false;
  {
    std::lock_guard<std::mutex> lock(waypoint_mutex_);
    // Reclaim space by dropping already-executed waypoints (keep one anchor).
    while (waypoints_.size() >= 2 && waypoints_[1].time <= now) {
      waypoints_.pop_front();
    }
    const double base = waypoints_.empty() ? now : waypoints_.back().time;
    for (std::size_t i = 0; i < poses.size(); ++i) {
      if (waypoints_.size() >= kMaxWaypoints) {
        dropped = true;
        break;
      }
      TimedPose wp;
      wp.time = base + static_cast<double>(i + 1) * command_dt_;
      wp.position = poses[i].first;
      wp.orientation = poses[i].second.normalized();
      waypoints_.push_back(wp);
    }
  }
  if (dropped && node_) {
    RCLCPP_WARN_THROTTLE(
        node_->get_logger(), *node_->get_clock(), 2000,
        "panda_mpc_controller: waypoint buffer full (%zu); dropping excess. "
        "Publish a sequence once instead of streaming, or clear first.",
        kMaxWaypoints);
  }
  // Mark the controllet as moving and keep getDesiredPose meaningful.
  Pose desired = getCurrentPose();
  desired.position = poses.back().first;
  desired.orientation = poses.back().second.normalized();
  setDesiredPoseBuffered(desired);
}

void Controller::pruneWaypoints(double now) {
  std::lock_guard<std::mutex> lock(waypoint_mutex_);
  while (waypoints_.size() >= 2 && waypoints_[1].time <= now) {
    waypoints_.pop_front();
  }
}

bool Controller::sampleWaypoints(double t, Eigen::Vector3d& position,
                                 Eigen::Quaterniond& orientation) {
  std::lock_guard<std::mutex> lock(waypoint_mutex_);
  if (waypoints_.empty()) {
    return false;
  }
  if (t <= waypoints_.front().time) {
    position = waypoints_.front().position;
    orientation = waypoints_.front().orientation;
    return true;
  }
  if (t >= waypoints_.back().time) {
    position = waypoints_.back().position;
    orientation = waypoints_.back().orientation;
    return true;
  }
  for (std::size_t i = 0; i + 1 < waypoints_.size(); ++i) {
    const TimedPose& lo = waypoints_[i];
    const TimedPose& hi = waypoints_[i + 1];
    if (t >= lo.time && t < hi.time) {
      const double span = hi.time - lo.time;
      const double alpha = span > 1e-9 ? (t - lo.time) / span : 0.0;
      position = (1.0 - alpha) * lo.position + alpha * hi.position;
      orientation = lo.orientation.slerp(alpha, hi.orientation);
      return true;
    }
  }
  position = waypoints_.back().position;
  orientation = waypoints_.back().orientation;
  return true;
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------
void Controller::startThread() {
  if (running_.exchange(true)) {
    return;  // already running
  }
  worker_ = std::thread(&ComlessPandaMpcController::mpcLoop, this);
}

void Controller::stopThread() {
  if (!running_.exchange(false)) {
    return;  // not running
  }
  if (worker_.joinable()) {
    worker_.join();
  }
}

void Controller::mpcLoop() {
  using clock = std::chrono::steady_clock;
  const auto period = std::chrono::duration_cast<clock::duration>(
      std::chrono::duration<double>(kSolvePeriod));
  const int T = mpc_.horizon();
  const double dt = mpc_.dt();

  while (running_.load()) {
    const auto loop_start = clock::now();

    StateSnapshot snap;
    {
      std::lock_guard<std::mutex> lock(snapshot_mutex_);
      snap = snapshot_;
    }

    if (snap.valid) {
      const double now = steadyNow();
      pruneWaypoints(now);

      std::vector<Vector6d> refs(T, Vector6d::Zero());
      for (int k = 0; k < T; ++k) {
        Eigen::Vector3d p_ref;
        Eigen::Quaterniond o_ref;
        if (sampleWaypoints(now + static_cast<double>(k + 1) * dt, p_ref, o_ref)) {
          Vector6d twist;
          twist.head(3) = p_ref - snap.position;
          twist.tail(3) =
              rotationVector(o_ref.toRotationMatrix() * snap.rotation.transpose());
          refs[k] = twist;
        }
        // else: empty buffer -> zero twist (hold current pose)
      }

      const Params params = getParametersBuffered();
      CartesianMpc::Weights weights;
      weights.task_weight = params.task_weight;
      weights.input_weight = params.input_weight;

      CartesianMpc::Solution sol;
      const bool solved = mpc_.solve(snap.q, snap.dq, snap.mass, snap.coriolis,
                                     snap.jacobian, refs, weights, sol);
      if (solved) {
        std::lock_guard<std::mutex> lock(solution_mutex_);
        solution_.valid = true;
        solution_.solve_time = now;
        solution_.dt = dt;
        solution_.q_ref = std::move(sol.q_ref);
        solution_.dq_ref = std::move(sol.dq_ref);
        solution_.tau_ff = std::move(sol.tau_ff);
      } else if (node_) {
        // Leave the last solution in place; the RT loop drops to hold once it
        // goes stale. Just warn.
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                             "panda_mpc_controller: QP solve failed.");
      }
    }

    std::this_thread::sleep_until(loop_start + period);
  }
}
