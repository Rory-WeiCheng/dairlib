#include <vector>
#include <math.h>
#include <gflags/gflags.h>

#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include <drake/lcm/drake_lcm.h>
#include <drake/multibody/tree/multibody_element.h>
#include <drake/multibody/parsing/parser.h>
#include "drake/math/autodiff.h"


#include "systems/robot_lcm_systems.h"
#include "dairlib/lcmt_robot_input.hpp"
#include "dairlib/lcmt_robot_output.hpp"
#include "dairlib/lcmt_c3.hpp"
#include "dairlib/lcmt_lcs.hpp"
#include "systems/system_utils.h"

#include "systems/robot_lcm_systems.h"
#include "systems/controllers/residual_learner.h"
#include "systems/framework/lcm_driven_loop.h"

// add scv reading utils for reading the learnt lcs matrices and make lcs
// just a rough way to incooperate learning part for sanity check
#include "common/file_utils.h"
#include "solvers/lcs.h"

DEFINE_int32(TTL, 0,
              "TTL level for publisher. "
              "Default value is 0.");

namespace dairlib {

using drake::geometry::SceneGraph;
using drake::multibody::MultibodyPlant;
using drake::multibody::AddMultibodyPlantSceneGraph;
using drake::math::RigidTransform;
using drake::systems::DiagramBuilder;
using drake::systems::lcm::LcmPublisherSystem;
using drake::systems::lcm::LcmSubscriberSystem;
using drake::systems::Context;
using drake::multibody::Parser;

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Map;
using solvers::LCS;

int DoMain(int argc, char* argv[]){
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  C3Parameters param = drake::yaml::LoadYamlFile<C3Parameters>(
    "examples/franka_trajectory_following/parameters.yaml");
  drake::lcm::DrakeLcm drake_lcm;
  drake::lcm::DrakeLcm drake_lcm_network("udpm://239.255.76.67:7667?ttl=1");

  DiagramBuilder<double> builder;

  DiagramBuilder<double> builder_franka;
  double sim_dt = 1e-4;

  auto [plant_franka, scene_graph_franka] = AddMultibodyPlantSceneGraph(&builder_franka, sim_dt);
  Parser parser_franka(&plant_franka);
  parser_franka.AddModelFromFile("examples/franka_trajectory_following/robot_properties_fingers/urdf/franka_box.urdf");
  parser_franka.AddModelFromFile("examples/franka_trajectory_following/robot_properties_fingers/urdf/sphere_model.urdf");
  RigidTransform<double> X_WI_franka = RigidTransform<double>::Identity();
  plant_franka.WeldFrames(plant_franka.world_frame(), plant_franka.GetFrameByName("panda_link0"), X_WI_franka);
  plant_franka.Finalize();
  auto context_franka = plant_franka.CreateDefaultContext();

  auto state_receiver = builder.AddSystem<systems::RobotOutputReceiver>(plant_franka);
  auto learner = builder.AddSystem<systems::controllers::ResidualLearner>();
  auto lcs_sender = builder.AddSystem<systems::RobotLCSSender>();

  builder.Connect(state_receiver->get_output_port(0), learner->get_input_port(0));
  builder.Connect(learner->get_output_port(), lcs_sender->get_input_port(0));

  // determine if ttl 0 or 1 should be used for publishing
  drake::lcm::DrakeLcm* pub_lcm;
  if (FLAGS_TTL == 0) {
    std::cout << "Using TTL=0" << std::endl;
    pub_lcm = &drake_lcm;
  }
  else if (FLAGS_TTL == 1) {
    std::cout << "Using TTL=1" << std::endl;
    pub_lcm = &drake_lcm_network;
  }

  // settings for publishing lcs message to the lcm Channel
  auto lcs_publisher = builder.AddSystem(
      LcmPublisherSystem::Make<dairlib::lcmt_lcs>("RESIDUAL_LCS", pub_lcm,
        {drake::systems::TriggerType::kForced}, 0.0));
  builder.Connect(lcs_sender->get_output_port(),lcs_publisher->get_input_port());

  auto diagram = builder.Build();
  // DrawAndSaveDiagramGraph(*diagram, "examples/franka_trajectory_following/diagram_lcm_control_demo");
  auto context_d = diagram->CreateDefaultContext();
  // Run lcm-driven simulation
  systems::LcmDrivenLoop<dairlib::lcmt_robot_output> loop(
      &drake_lcm, std::move(diagram), state_receiver, "FRANKA_STATE_ESTIMATE", true);
  loop.Simulate(std::numeric_limits<double>::infinity());

  return 0;
}
} // namespace dairlib

int main(int argc, char* argv[]) { dairlib::DoMain(argc, argv);}
