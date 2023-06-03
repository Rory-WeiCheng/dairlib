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
#include <drake/common/trajectories/piecewise_polynomial.h>
#include <drake/math/rigid_transform.h>
#include "drake/math/autodiff.h"


#include "systems/robot_lcm_systems.h"
#include "dairlib/lcmt_robot_output.hpp"
#include "dairlib/lcmt_c3.hpp"
#include "dairlib/lcmt_learning_data.hpp"
#include "multibody/multibody_utils.h"
#include "systems/system_utils.h"

#include "examples/franka_trajectory_following/c3_parameters.h"
#include "systems/robot_lcm_systems.h"
#include "systems/controllers/data_generator.h"
#include "systems/framework/lcm_driven_loop.h"

// add scv reading utils for reading the learnt lcs matrices and make lcs
// just a rough way to incooperate learning part for sanity check
#include "common/file_utils.h"

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
using multibody::makeNameToPositionsMap;
using multibody::makeNameToVelocitiesMap;
using drake::trajectories::PiecewisePolynomial;

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

  /* ------------------------------------ plant -------------------------------------------------*/

  /// parse plant from urdfs
  MultibodyPlant<double> plant(0.0);
  Parser parser(&plant);
  parser.package_map().Add("robot_properties_fingers",
                        "examples/franka_trajectory_following/robot_properties_fingers");
  parser.AddModelFromFile("examples/franka_trajectory_following/robot_properties_fingers/urdf/trifinger_minimal_collision_2.urdf");
  parser.AddModelFromFile("examples/franka_trajectory_following/robot_properties_fingers/urdf/sphere_model.urdf");

  /// Fix base of finger to world
  RigidTransform<double> X_WI = RigidTransform<double>::Identity();
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link"), X_WI);
  plant.Finalize();

  DiagramBuilder<double> builder;

  /* -------------------------------------- plant_f ------------------------------------------------*/

  DiagramBuilder<double> builder_f;

  auto [plant_f, scene_graph] = AddMultibodyPlantSceneGraph(&builder_f, 0.0);
  Parser parser_f(&plant_f);
  parser_f.package_map().Add("robot_properties_fingers",
                        "examples/franka_trajectory_following/robot_properties_fingers");
  parser_f.AddModelFromFile("examples/franka_trajectory_following/robot_properties_fingers/urdf/trifinger_minimal_collision_2.urdf");
  parser_f.AddModelFromFile("examples/franka_trajectory_following/robot_properties_fingers/urdf/sphere_model.urdf");
  RigidTransform<double> X_WI_f = RigidTransform<double>::Identity();
  plant_f.WeldFrames(plant_f.world_frame(), plant_f.GetFrameByName("base_link"), X_WI_f);
  plant_f.Finalize();

  std::unique_ptr<MultibodyPlant<drake::AutoDiffXd>> plant_ad_f =
    drake::systems::System<double>::ToAutoDiffXd(plant_f);
  auto context_ad_f = plant_ad_f->CreateDefaultContext();
  auto diagram_f = builder_f.Build();
  std::unique_ptr<Context<double>> diagram_context = diagram_f->CreateDefaultContext();
  auto& context_f = diagram_f->GetMutableSubsystemContext(plant_f, diagram_context.get());

  /* ------------------------------------ plant_franka -------------------------------------------------*/

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

  /* ----------------------------- contact parameters and geometry ------------------------------------*/

  double mu = param.mu;
  int num_friction_directions = 2;

  drake::geometry::GeometryId finger_geoms = 
    plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("tip_link_1_real"))[0];
  drake::geometry::GeometryId sphere_geoms = 
    plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("sphere"))[0];
  drake::geometry::GeometryId ground_geoms = 
    plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("box"))[0];
  std::vector<drake::geometry::GeometryId> contact_geoms = 
    {finger_geoms, sphere_geoms, ground_geoms};

  /* --------------------------------- to AutoDiff -------------------------------------------------*/
  std::unique_ptr<MultibodyPlant<drake::AutoDiffXd>> plant_ad = 
    drake::systems::System<double>::ToAutoDiffXd(plant);
  auto context = plant.CreateDefaultContext();
  auto context_ad = plant_ad->CreateDefaultContext();

  /* --------------------------------- Draw block diagram -----------------------------------------*/
  /// input port 0: state from franka
  auto state_receiver = builder.AddSystem<systems::RobotOutputReceiver>(plant_franka);

  /// input port 1: control from c3
  auto c3_subscriber = builder.AddSystem(LcmSubscriberSystem::Make<dairlib::lcmt_c3>(
          "CONTROLLER_INPUT", &drake_lcm));
  auto c3_receiver =
      builder.AddSystem<systems::RobotC3Receiver>(14, 9, 6, 12);

  /// data_generator
  auto generator = builder.AddSystem<systems::controllers::Data_Generator>(
                                  plant, plant_f, plant_franka, *context, 
                                  context_f, *context_franka, *plant_ad, 
                                  *plant_ad_f, *context_ad, *context_ad_f, 
                                  scene_graph, *diagram_f, contact_geoms, 
                                  num_friction_directions, mu);

  /// data set sender and publisher
  auto data_set_sender = builder.AddSystem<systems::RobotDataSender>();

  /// connect blocks
  builder.Connect(state_receiver->get_output_port(0), generator->get_input_port(0));

  // lcs subscriber to taken in the lcm message about residual lcs and connect port
  builder.Connect(c3_subscriber->get_output_port(0),c3_receiver->get_input_port(0));
  builder.Connect(c3_receiver->get_output_port(0),generator->get_input_port(1));

  builder.Connect(generator->get_output_port(), data_set_sender->get_input_port(0));
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
  auto data_set_publisher = builder.AddSystem(
      LcmPublisherSystem::Make<dairlib::lcmt_learning_data>(
        "LEARNING_DATASET", pub_lcm,
        {drake::systems::TriggerType::kForced}, 0.0));
  builder.Connect(data_set_sender->get_output_port(),
                  data_set_publisher->get_input_port());

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
