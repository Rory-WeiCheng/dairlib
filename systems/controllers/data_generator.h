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
#include "solvers/learning_data.h"

#include "drake/common/autodiff.h"
#include "drake/geometry/scene_graph.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/context.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"

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
using solvers::LearningData;
namespace controllers {

class Data_Generator : public LeafSystem<double> {
 public:
  Data_Generator(
      const drake::multibody::MultibodyPlant<double>& plant,
      drake::multibody::MultibodyPlant<double>& plant_f,
      const drake::multibody::MultibodyPlant<double>& plant_franka,
      drake::systems::Context<double>& context,
      drake::systems::Context<double>& context_f,
      drake::systems::Context<double>& context_franka,
      const drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad,
      drake::multibody::MultibodyPlant<drake::AutoDiffXd>& plant_ad_f,
      drake::systems::Context<drake::AutoDiffXd>& context_ad,
      drake::systems::Context<drake::AutoDiffXd>& context_ad_f,
      const drake::geometry::SceneGraph<double>& scene_graph,
      const drake::systems::Diagram<double>& diagram,
      std::vector<drake::geometry::GeometryId> contact_geoms,
      int num_friction_directions, double mu);

 private:
  void CalcData(const drake::systems::Context<double>& context,
                   LearningData* data_pack) const;

  int franka_state_input_port_;
  int c3_state_input_port_;
  int data_output_port_;
  const MultibodyPlant<double>& plant_;
  MultibodyPlant<double>& plant_f_;
  const MultibodyPlant<double>& plant_franka_;
  drake::systems::Context<double>& context_;
  drake::systems::Context<double>& context_f_;
  drake::systems::Context<double>& context_franka_;
  const MultibodyPlant<drake::AutoDiffXd>& plant_ad_;
  MultibodyPlant<drake::AutoDiffXd>& plant_ad_f_;
  drake::systems::Context<drake::AutoDiffXd>& context_ad_;
  drake::systems::Context<drake::AutoDiffXd>& context_ad_f_;
  const drake::geometry::SceneGraph<double>& scene_graph_;
  const drake::systems::Diagram<double>& diagram_;
  std::vector<drake::geometry::GeometryId> contact_geoms_;
  int num_friction_directions_;
  double mu_;
  C3Parameters param_;

  // TODO: make all the mutable variables drake states
  // velocity
  mutable double prev_timestamp_{0};
  mutable bool received_first_message_{false};
  mutable double first_message_time_{-1.0};
};

}  // namespace controllers
}  // namespace systems
}  // namespace dairlib