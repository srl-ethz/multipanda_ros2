#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace panda_controllers {

// Continuous-time first-order filter discretized exactly over dt. A non-
// positive time constant disables filtering (alpha == 1).
inline double firstOrderFilterAlpha(double dt, double time_constant) {
  if (dt <= 0.0) {
    return 0.0;
  }
  if (time_constant <= std::numeric_limits<double>::epsilon()) {
    return 1.0;
  }
  return std::clamp(1.0 - std::exp(-dt / time_constant), 0.0, 1.0);
}

inline Eigen::Quaterniond normalizedShortestTarget(const Eigen::Quaterniond& from,
                                                   const Eigen::Quaterniond& target) {
  Eigen::Quaterniond normalized = target;
  if (normalized.norm() <= std::numeric_limits<double>::epsilon()) {
    normalized = Eigen::Quaterniond::Identity();
  } else {
    normalized.normalize();
  }
  if (from.coeffs().dot(normalized.coeffs()) < 0.0) {
    normalized.coeffs() = -normalized.coeffs();
  }
  return normalized;
}

inline Eigen::Quaterniond firstOrderFilterOrientation(const Eigen::Quaterniond& current,
                                                      const Eigen::Quaterniond& target,
                                                      double alpha) {
  Eigen::Quaterniond from = current;
  if (from.norm() <= std::numeric_limits<double>::epsilon()) {
    from = Eigen::Quaterniond::Identity();
  } else {
    from.normalize();
  }
  const Eigen::Quaterniond to = normalizedShortestTarget(from, target);
  Eigen::Quaterniond filtered = from.slerp(std::clamp(alpha, 0.0, 1.0), to);
  filtered.normalize();
  return filtered;
}

}  // namespace panda_controllers
