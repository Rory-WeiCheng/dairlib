#include <vector>
#include <math.h>
#include <gflags/gflags.h>
#include <signal.h>

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
#include "drake/systems/primitives/trajectory_source.h"


#include "systems/robot_lcm_systems.h"
#include "dairlib/lcmt_robot_input.hpp"
#include "dairlib/lcmt_robot_output.hpp"
#include "dairlib/lcmt_c3.hpp"
#include "multibody/multibody_utils.h"
#include "systems/system_utils.h"

#include "examples/franka_trajectory_following/c3_parameters.h"
#include "examples/franka_trajectory_following/systems/c3_trajectory_source.h"
#include "examples/franka_trajectory_following/systems/gravity_compensator.h"
#include "systems/robot_lcm_systems.h"
#include "systems/controllers/impedance_controller.h"
#include "systems/framework/lcm_driven_loop.h"

#define ROS

#ifdef ROS

#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"
#include "systems/ros/ros_publisher_system.h"
#include "systems/ros/c3_ros_conversions.h"

void SigintHandler(int sig) {
  ros::shutdown();
  exit(sig);
}

#endif

DEFINE_string(channel, "FRANKA_OUTPUT",
              "LCM channel for receiving state. "
              "Use FRANKA_OUTPUT to get state from simulator, and "
              "use FRANKA_ROS_OUTPUT to get state from franka_ros");

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

int DoMain(int argc, char* argv[]){
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  C3Parameters param = drake::yaml::LoadYamlFile<C3Parameters>(
    "examples/franka_trajectory_following/parameters.yaml");

  drake::lcm::DrakeLcm drake_lcm;
  // drake::lcm::DrakeLcm drake_network("udpm://239.255.76.67:7667?ttl=1");

  MultibodyPlant<double> plant(0.0);
  Parser parser(&plant);
  parser.AddModelFromFile("examples/franka_trajectory_following/robot_properties_fingers/urdf/franka_box.urdf");
  parser.AddModelFromFile("examples/franka_trajectory_following/robot_properties_fingers/urdf/sphere.urdf");
  
  /// Fix base of finger to world
  RigidTransform<double> X_WI = RigidTransform<double>::Identity();
  plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"), X_WI);
  plant.Finalize();

  DiagramBuilder<double> builder;
  auto state_receiver = builder.AddSystem<systems::RobotOutputReceiver>(plant);

  /* -------------------------------------------------------------------------------------------*/

  DiagramBuilder<double> builder_f;
  auto [plant_f, scene_graph] = AddMultibodyPlantSceneGraph(&builder_f, 0.0);
  Parser parser_f(&plant_f);
  parser_f.AddModelFromFile("examples/franka_trajectory_following/robot_properties_fingers/urdf/franka_box.urdf");
  parser_f.AddModelFromFile("examples/franka_trajectory_following/robot_properties_fingers/urdf/sphere.urdf");

  plant_f.WeldFrames(plant_f.world_frame(), plant_f.GetFrameByName("panda_link0"), X_WI);
  plant_f.Finalize();

  auto diagram_f = builder_f.Build();
  auto diagram_context_f = diagram_f->CreateDefaultContext();
  auto& context_f = diagram_f->GetMutableSubsystemContext(plant_f, diagram_context_f.get());

  /* -------------------------------------------------------------------------------------------*/

  int nq = plant.num_positions();

  VectorXd q = VectorXd::Zero(nq);
  auto context = plant.CreateDefaultContext();

  double translational_stiffness = param.translational_stiffness;
  double rotational_stiffness = param.rotational_stiffness;
  double damping_ratio = param.damping_ratio;

  MatrixXd K = MatrixXd::Zero(6,6);
  MatrixXd B = MatrixXd::Zero(6,6);
  K.block(0,0,3,3) << rotational_stiffness * MatrixXd::Identity(3,3);
  K.block(3,3,3,3) << translational_stiffness * MatrixXd::Identity(3,3);
  B.block(0,0,3,3) << 2 * damping_ratio * sqrt(rotational_stiffness) * MatrixXd::Identity(3,3);
  B.block(3,3,3,3) << 2 * damping_ratio * sqrt(translational_stiffness) * MatrixXd::Identity(3,3);

  MatrixXd K_null = param.stiffness_null * MatrixXd::Identity(7,7);
  MatrixXd B_null = param.damping_null * MatrixXd::Identity(7,7);
  VectorXd qd = param.q_null_desired;

  drake::geometry::GeometryId sphere_geoms = 
    plant_f.GetCollisionGeometriesForBody(plant.GetBodyByName("sphere"))[0];
  drake::geometry::GeometryId EE_geoms = 
    plant_f.GetCollisionGeometriesForBody(plant.GetBodyByName("panda_link10"))[0];
  std::vector<drake::geometry::GeometryId> contact_geoms = {EE_geoms, sphere_geoms};

  int num_friction_directions = 2;
  double moving_offset = param.moving_offset;
  double pushing_offset = param.pushing_offset;

  auto controller = builder.AddSystem<systems::controllers::ImpedanceController>(
      plant, plant_f, *context, context_f, K, B, K_null, B_null, qd,
      contact_geoms, num_friction_directions, moving_offset, pushing_offset);
  auto gravity_compensator = builder.AddSystem<systems::GravityCompensator>(plant, *context);

  /* -------------------------------------------------------------------------------------------*/
  /// create trajectory source
  // double time_inc = 10;
  // double num_points = 10;

  // std::vector<MatrixXd> points(num_points);
  // std::vector<double> times;
  // points[0] = Vector3d(0.55, 0.0, 0.12);
  // points[1] = Vector3d(0.55, 0.0, 0.12);
  // points[2] = Vector3d(0.55, 0.0, 0.12);
  // points[3] = Vector3d(0.55, 0.0, 0.12);
  // points[4] = Vector3d(0.55, 0.1, 0.12);
  // points[5] = Vector3d(0.3, 0.1, 0.12);
  // points[6] = Vector3d(0.5, -0.1, 0.12);
  // points[7] = Vector3d(0.55, -0.1, 0.2);
  // points[8] = Vector3d(0.55, -0.1, 0.1);
  // points[9] = Vector3d(0.55, 0.0, 0.12);

  // double time_inc = 10;
  // double num_points = 2;

  // std::vector<MatrixXd> points(num_points);
  // std::vector<double> times;
  // points[0] = Vector3d(0.55, 0.0, 0.12);
  // points[1] = Vector3d(0.60, 0.05, 0.20);

  double time_inc = 2;
  double num_points = 5;

  double l = 0.15;

  std::vector<MatrixXd> points(num_points);
  std::vector<double> times;
  points[0] = Vector3d(0.55, 0.0, 0.12);
  points[1] = Vector3d(0.55, 0.0+l, 0.12);
  points[2] = Vector3d(0.55-l, 0.0+l, 0.12);
  points[3] = Vector3d(0.55-l, 0.0, 0.12);
  points[4] = Vector3d(0.55, 0.0, 0.12);


  for (int i = 0; i < num_points; i++){
    times.push_back(i*time_inc);
  }

  auto ee_trajectory = drake::trajectories::PiecewisePolynomial<
    double>::FirstOrderHold(times, points);

  auto input_trajectory = builder.AddSystem<systems::C3TrajectorySource>(
      ee_trajectory);
  builder.Connect(state_receiver->get_output_port(0),
    input_trajectory->get_input_port(0));
  builder.Connect(input_trajectory->get_output_port(0),
    controller->get_input_port(1));

  /* -------------------------------------------------------------------------------------------*/

  builder.Connect(state_receiver->get_output_port(0), 
    controller->get_input_port(0));

  // sim
  if (FLAGS_channel == "FRANKA_OUTPUT"){
    auto control_sender = builder.AddSystem<systems::RobotCommandSender>(plant);
    builder.Connect(controller->get_output_port(), gravity_compensator->get_input_port());
    builder.Connect(gravity_compensator->get_output_port(), control_sender->get_input_port());

    auto control_publisher = builder.AddSystem(
        LcmPublisherSystem::Make<dairlib::lcmt_robot_input>(
          "FRANKA_INPUT", &drake_lcm, 
          {drake::systems::TriggerType::kForced}, 0.0));
    builder.Connect(control_sender->get_output_port(),
        control_publisher->get_input_port());
  }

  /* -------------------------------------------------------------------------------------------*/
#ifdef ROS
  else if (FLAGS_channel == "FRANKA_ROS_OUTPUT"){
    /// Publish to ROS topic
    ros::init(argc, argv, "impedance_controller");
    ros::NodeHandle node_handle;
    signal(SIGINT, SigintHandler);

    auto impedance_to_ros = builder.AddSystem<systems::TimestampedVectorToROS>(7);
    // try making this kForced
    auto ros_publisher = builder.AddSystem(
        systems::RosPublisherSystem<std_msgs::Float64MultiArray>::Make("/c3/franka_input", &node_handle, .0005));
    
    builder.Connect(controller->get_output_port(), impedance_to_ros->get_input_port());
    builder.Connect(impedance_to_ros->get_output_port(), ros_publisher->get_input_port());

    auto ros_lcm_sender = builder.AddSystem<systems::RobotCommandSender>(plant);
    auto echo_ros_lcm = builder.AddSystem(
        LcmPublisherSystem::Make<dairlib::lcmt_robot_input>(
          "FRANKA_INPUT_WO_G", &drake_lcm, 
          {drake::systems::TriggerType::kForced}, 0.0));
    builder.Connect(controller->get_output_port(),
        ros_lcm_sender->get_input_port());
    builder.Connect(ros_lcm_sender->get_output_port(),
        echo_ros_lcm->get_input_port());
  }
#endif

  auto diagram = builder.Build();
  // DrawAndSaveDiagramGraph(*diagram, "examples/franka_trajectory_following/diagram_run_impedance_experiment");


  auto context_d = diagram->CreateDefaultContext();
  auto& receiver_context = diagram->GetMutableSubsystemContext(*state_receiver, context_d.get());
  (void) receiver_context; // suppressed unused variable warning

  systems::LcmDrivenLoop<dairlib::lcmt_robot_output> loop(
      &drake_lcm, std::move(diagram), state_receiver, FLAGS_channel, true);
  
  loop.Simulate(std::numeric_limits<double>::infinity());

  return 0;
}

} // namespace dairlib

int main(int argc, char* argv[]) { dairlib::DoMain(argc, argv);}
