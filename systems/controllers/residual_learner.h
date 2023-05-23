#pragma once

#include <vector>
#include <map>
#include <string>
#include <deque>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "systems/framework/output_vector.h"

#include "drake/systems/framework/leaf_system.h"
#include <drake/multibody/parsing/parser.h>
#include <gflags/gflags.h>

#include "common/find_resource.h"
#include "multibody/geom_geom_collider.h"
#include "multibody/kinematic/kinematic_evaluator_set.h"
#include "solvers/lcs_factory_franka.h"
#include "solvers/lcs.h"

#include "drake/common/autodiff.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/optimization/manipulator_equation_constraint.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include "drake/common/trajectories/piecewise_polynomial.h"

#include "examples/franka_trajectory_following/c3_parameters.h"
#include "yaml-cpp/yaml.h"
#include "drake/common/yaml/yaml_io.h"
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

 private:
  void CalcResidual(const drake::systems::Context<double>& context,
                   solvers::LCS* residual_lcs) const;

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