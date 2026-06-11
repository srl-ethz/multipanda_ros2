#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <multi_mode_controller/base/panda_controller_base.h>
#include <multi_mode_controller/utils/cartesian_mpc.h>

namespace panda_controllers {

// Single Cartesian pose target (Franka base frame). Used by the base-class
// machinery (getCurrentPose / moving); the MPC tracks the waypoint buffer, not
// this single pose.
struct PandaMpcControllerPose {
  Eigen::Matrix<double, 7, 1> q_n;
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
};

struct PandaMpcControllerParams {
  Eigen::Matrix<double, 7, 1> kp;          // 1 kHz tracking PD proportional gains
  Eigen::Matrix<double, 7, 1> kd;          // 1 kHz tracking PD derivative gains
  Eigen::Matrix<double, 6, 1> task_weight; // MPC task-space tracking weight
  double input_weight;                     // MPC torque regularization weight
  double posture_weight;                   // MPC nominal-posture regularization
  double velocity_weight;                  // MPC reference-velocity tracking
  double accel_weight;                     // MPC acceleration smoothness
};

// A reference pose stamped on the steady clock (Franka base frame).
struct TimedPose {
  double time;
  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;
};

// Comless core of the Stage-1 Cartesian-tracking MPC controllet.
//
// Two-rate architecture:
//   * computeTauImpl runs at 1 kHz (the ros2_control loop). It snapshots the
//     robot state for the solver and applies a joint-space tracking PD on the
//     latest MPC reference: tau = tau_ff + Kp.(q_ref - q) + Kd.(dq_ref - dq).
//   * A background thread (~100 Hz) reads the snapshot, builds the reference
//     twists from the waypoint buffer and solves the QP (CartesianMpc).
// The two threads exchange data through try-locked snapshot/solution buffers so
// the 1 kHz loop never blocks on the solver.
class ComlessPandaMpcController :
    public virtual PandaControllerBase<PandaMpcControllerParams,
                                       PandaMpcControllerPose> {
 public:
  ComlessPandaMpcController();
  virtual ~ComlessPandaMpcController();

  // Waypoint interface used by the com (ROS) layer. Poses are already in the
  // Franka base frame. appendWaypoints() time-stamps the sequence using
  // command_dt, chaining onto any waypoints already buffered.
  void appendWaypoints(
      const std::vector<std::pair<Eigen::Vector3d, Eigen::Quaterniond>>& poses);
  // Replace the whole trajectory with a single target: atomically purges the
  // buffer and inserts `pose` as the next step. Use for single-pose commands
  // (e.g. end_effector_pose_cmd) that should override, not extend, the queue.
  void setImmediateTarget(const Eigen::Vector3d& position,
                          const Eigen::Quaterniond& orientation);
  void clearWaypoints();
  void setCommandDt(double command_dt);
  // Override the nominal posture of the QP's redundancy-resolving
  // regularization. Call from init (before the solve thread starts).
  void setNominalPosture(const Eigen::Matrix<double, 7, 1>& q_nominal);

 protected:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  using Matrix7d = Eigen::Matrix<double, 7, 7>;
  using Matrix67d = Eigen::Matrix<double, 6, 7>;

 private:
  void computeTauImpl(const std::vector<std::array<double, 7>*>& tau,
      const PandaMpcControllerPose& desired,
      const PandaMpcControllerParams& p) override;
  void postprocessTauImpl(
      const std::vector<std::array<double, 7>*>& tau) override;
  void startImpl() override;
  void stopImpl() override;
  PandaMpcControllerParams defaultParameters() override;
  PandaMpcControllerPose getCurrentPoseImpl() override;

  // ----- worker thread -----
  void mpcLoop();
  void startThread();
  void stopThread();

  // ----- shared between RT loop and worker thread -----
  struct StateSnapshot {
    bool valid = false;
    Vector7d q = Vector7d::Zero();
    Vector7d dq = Vector7d::Zero();
    Matrix7d mass = Matrix7d::Identity();
    Vector7d coriolis = Vector7d::Zero();
    Matrix67d jacobian = Matrix67d::Zero();
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  };
  struct SolutionBuffer {
    bool valid = false;
    double solve_time = 0.0;  // steady-clock seconds
    double dt = 0.01;
    std::vector<Vector7d> q_ref;
    std::vector<Vector7d> dq_ref;
    std::vector<Vector7d> tau_ff;
  };

  // Sample the waypoint buffer at steady-time `t` (Franka base frame). Returns
  // false when the buffer is empty (caller should hold the current pose).
  bool sampleWaypoints(double t, Eigen::Vector3d& position,
                       Eigen::Quaterniond& orientation);
  void pruneWaypoints(double now);

  CartesianMpc mpc_;

  std::mutex snapshot_mutex_;
  StateSnapshot snapshot_;

  std::mutex solution_mutex_;
  SolutionBuffer solution_;          // written by worker, read by RT loop
  SolutionBuffer active_solution_;   // RT-loop-private cached copy

  std::mutex waypoint_mutex_;
  std::deque<TimedPose> waypoints_;
  double command_dt_{0.1};

  Vector7d hold_q_{Vector7d::Zero()};  // RT fallback target (hold position)

  std::thread worker_;
  std::atomic<bool> running_{false};
};

}  // namespace panda_controllers
