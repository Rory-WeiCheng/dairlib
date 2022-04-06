#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "systems/framework/output_vector.h"

#include "drake/systems/framework/leaf_system.h"
//#include
//"external/drake/common/_virtual_includes/autodiff/drake/common/eigen_autodiff_types.h"
//#include
//"external/drake/tools/install/libdrake/_virtual_includes/drake_shared_library/drake/systems/framework/context.h"
#include <drake/multibody/parsing/parser.h>
#include <gflags/gflags.h>

#include "common/find_resource.h"
#include "multibody/geom_geom_collider.h"
#include "multibody/kinematic/kinematic_evaluator_set.h"
#include "solvers/lcs_factory.h"

#include "drake/common/autodiff.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/optimization/manipulator_equation_constraint.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"

using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using drake::systems::LeafSystem;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace dairlib {
namespace systems {
namespace controllers {

class C3Controller : public LeafSystem<double> {
 public:
  C3Controller(
      const drake::multibody::MultibodyPlant<double>& plant,
      drake::systems::Context<double>& context,
      const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      drake::systems::Context<drake::AutoDiffXd>& context_ad,
      std::vector<drake::geometry::GeometryId>
          contact_geoms,
      int num_friction_directions, double mu,
      const std::vector<Eigen::MatrixXd>& Q,
      const std::vector<Eigen::MatrixXd>& R,
      const std::vector<Eigen::MatrixXd>& G,
      const std::vector<Eigen::MatrixXd>& U,
      const std::vector<Eigen::VectorXd>& xdesired
      );

  const drake::systems::InputPort<double>& get_input_port_config() const {
    return this->get_input_port(state_input_port_);
  }

  const drake::systems::OutputPort<double>& get_input_port_output() const {
    return this->get_output_port(control_output_port_);
  }

  // void AddConstraint() const;

 private:
  void CalcControl(const drake::systems::Context<double>& context,
                   drake::systems::BasicVector<double>* output) const;

  int state_input_port_;
  int control_output_port_;
  const MultibodyPlant<double>& plant_;
  const MultibodyPlant<drake::AutoDiffXd>& plant_ad_;
  drake::systems::Context<double>& context_;
  drake::systems::Context<drake::AutoDiffXd>& context_ad_;
  const std::vector<drake::geometry::GeometryId>
      contact_geoms_;
  int num_friction_directions_;
  double mu_;
  const std::vector<Eigen::MatrixXd> Q_;
  const std::vector<Eigen::MatrixXd> R_;
  const std::vector<Eigen::MatrixXd> U_;
  const std::vector<Eigen::MatrixXd> G_;
  const std::vector<Eigen::VectorXd> xdesired_;
};

}  // namespace controllers
}  // namespace systems
}  // namespace dairlib