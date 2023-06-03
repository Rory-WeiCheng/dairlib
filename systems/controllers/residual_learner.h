#pragma once

#include <vector>
#include <map>
#include <string>
#include <deque>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "systems/framework/output_vector.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/framework/context.h"
#include "solvers/lcs.h"

#include <random>


using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using drake::systems::LeafSystem;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace dairlib {
namespace systems {
namespace controllers {

class ResidualLearner : public LeafSystem<double> {
 public:
  ResidualLearner();
  void CalcResidual(const drake::systems::Context<double>& context,
                   solvers::LCS* residual_lcs) const;

 private:

  int num_state;
  int num_velocity;
  int num_control;
  int num_lambda;

  int state_input_port_;
  int lcs_output_port_;
};

}  // namespace controllers
}  // namespace systems
}  // namespace dairlib