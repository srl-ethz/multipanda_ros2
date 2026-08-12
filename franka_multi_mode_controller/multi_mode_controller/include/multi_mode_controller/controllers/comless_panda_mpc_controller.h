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

// Shaping filter applied to SINGLE-POSE commands (end_effector_pose_cmd) only.
//
// Why: a policy trained against panda_cartesian_impedance_controller has
// implicitly learned that controller's command->motion map. Measured on the
// faithful sim plant (2 cm chirp, single-pose path), the two controllers are
// nowhere near each other:
//
//                       -3 dB      -90 deg    5 cm step rise    ss error
//   impedance           0.34 Hz    0.66 Hz    0.89 s            -2.5 mm
//   MPC (ungoverned)    1.01 Hz    1.41 Hz    0.43 s            -0.1 mm
//
// The MPC is ~3x the bandwidth, and its step response is a constant-rate
// approach (~94 mm/s regardless of step size) where the impedance law's is
// error-proportional. Re-weighting the QP cannot fix this: the shapes differ,
// not just the speeds. So the command is instead shaped by an explicit
// second-order reference model that the MPC then tracks - model-reference
// control, with the QP demoted to a tracking layer that still enforces every
// joint/torque constraint.
//
// The rollout is re-ANCHORED on the measured EE pose and twist at every solve
// (x0 = 0 in twist-from-current coordinates, v0 = J*dq), NOT integrated
// open-loop from the command. This is essential and deliberate: an open-loop
// reference model is an integrator, so a blocked end-effector would let the
// reference march into the obstacle and the push would grow until something
// faults. Anchoring reproduces the impedance law's own structure (its damping
// term likewise acts on the measured J*dq) and preserves the MPC's existing
// compliance - a blocked EE leads the reference by a bounded amount and stops.
//
// The queued path (mpc_end_effector_pose_cmd) is deliberately NOT shaped: those
// waypoints are already time-parameterized, and the horizon previews them.
struct MpcReferenceModel {
  bool enabled = true;
  double filter_tau = 0.1;   // s, EMA on the target (matches the impedance
                             // controllet's 1 kHz gain-0.01 pose filter)
  // Translation. Seeded from a fit to the measured impedance chirp response
  // (wn=7.11 rad/s, zeta=1.48 -> real poles at 2.8 and 18.3 rad/s), then wn
  // raised to 8.0 so the CASCADE with the MPC's own tracking dynamics lands on
  // the impedance response rather than the reference model alone.
  //
  // The fit's DC gain of 0.91 is deliberately NOT reproduced: that is the
  // impedance law's compliance defect (mm-scale steady-state error under load),
  // not a property worth building into a controller meant to reach its target.
  //
  // max_accel bounds only the stiffness term, so it sets the saturated
  // approach speed at max_accel/(2*zeta*omega_n) = 0.148 m/s while leaving
  // small signals linear. It is a compromise, and the one place this model
  // cannot match the impedance law on both ends at once: the impedance
  // controller saturates via a 15 N force clamp against a large task damping
  // (ratio ~55/s), the reduced-order model has ratio 23.7/s, so one value
  // cannot reproduce both its 0.09 m/s large-step speed and its Lissajous
  // amplitude. 3.5 favours the streamed small/medium-signal regime a policy
  // actually drives; 2.13 matches 20 cm steps but collapses Lissajous
  // amplitude to 62%/47%. See MPC_development_plan.md Stage 1.13.
  double omega_n = 8.0;      // rad/s
  double zeta = 1.48;
  double max_velocity = 0.30;    // m/s,   safety cap, not normally binding
  double max_accel = 3.5;        // m/s^2, bounds the stiffness term only
  // Rotation (same model on the rotation-vector error).
  double omega_n_rot = 8.0;      // rad/s
  double zeta_rot = 1.48;
  double max_velocity_rot = 1.0;  // rad/s
  double max_accel_rot = 20.0;    // rad/s^2
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
  // Configure the single-pose reference model. Call from init (before the
  // solve thread starts).
  void setReferenceModel(const MpcReferenceModel& model);

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
  // Build the horizon's reference twists by rolling the reference model
  // forward from the measured state. Worker-thread only.
  void rolloutReferenceModel(const StateSnapshot& snap,
                             const Eigen::Vector3d& target_position,
                             const Eigen::Quaterniond& target_orientation,
                             double dt, std::vector<Vector6d>& refs);

  CartesianMpc mpc_;

  std::mutex snapshot_mutex_;
  StateSnapshot snapshot_;

  std::mutex solution_mutex_;
  SolutionBuffer solution_;          // written by worker, read by RT loop
  SolutionBuffer active_solution_;   // RT-loop-private cached copy

  std::mutex waypoint_mutex_;
  std::deque<TimedPose> waypoints_;
  double command_dt_{0.1};
  // True while the buffer holds a single-pose override (setImmediateTarget),
  // i.e. the end_effector_pose_cmd path the reference model shapes. Guarded by
  // waypoint_mutex_ together with the buffer it describes.
  bool single_pose_mode_{false};

  // Reference model: config is set once at init, the EMA state is touched only
  // by the worker thread (mpcLoop), so neither needs a lock.
  MpcReferenceModel ref_model_;
  Eigen::Vector3d ref_filter_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond ref_filter_orientation_{Eigen::Quaterniond::Identity()};
  bool ref_filter_valid_{false};

  Vector7d hold_q_{Vector7d::Zero()};  // RT fallback target (hold position)

  std::thread worker_;
  std::atomic<bool> running_{false};
};

}  // namespace panda_controllers
