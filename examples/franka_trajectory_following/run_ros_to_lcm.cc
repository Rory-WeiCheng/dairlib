#include <memory>
#include <signal.h>
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/lcm/lcm_interface_system.h"
#include "drake/systems/lcm/lcm_publisher_system.h"
#include "drake/systems/lcm/lcm_subscriber_system.h"
#include <drake/lcm/drake_lcm.h>
#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"

#include "systems/ros/ros_subscriber_system.h"
#include "systems/ros/ros_publisher_system.h"
#include "systems/ros/c3_ros_conversions.h"
#include "systems/system_utils.h"

using drake::systems::DiagramBuilder;
using drake::systems::Simulator;
using drake::systems::lcm::LcmPublisherSystem;
using drake::systems::lcm::LcmSubscriberSystem;

using dairlib::systems::RosSubscriberSystem;
using dairlib::systems::RosPublisherSystem;
using dairlib::systems::ROSToRobotOutputLCM;
using dairlib::systems::ROSToC3LCM;

// Shutdown ROS gracefully and then exit
void SigintHandler(int sig) {
  ros::shutdown();
  exit(sig);
}

namespace dairlib {

int DoMain(int argc, char* argv[]){
  ros::init(argc, argv, "test_ros_subscriber_system");
  ros::NodeHandle node_handle;

  DiagramBuilder<double> builder;

  /// systems
  auto ros_subscriber =
      builder.AddSystem(RosSubscriberSystem<std_msgs::Float64MultiArray>::Make(
          "/c3/franka_state", &node_handle));
  auto to_robot_output = builder.AddSystem(ROSToRobotOutputLCM::Make(4, 3, 3));  
  // change this to output correctly (i.e. when ros subscriber gets new message)
  auto robot_output_pub = builder.AddSystem(
    LcmPublisherSystem::Make<dairlib::lcmt_robot_output>(
      "LCM_ROBOT_OUTPUT_TEST", &drake_lcm, 
      {drake::systems::TriggerType::kPeriodic}, 0.25));

  /// connections
  builder.Connect(ros_subscriber->get_output_port(), to_robot_output->get_input_port());
  builder.Connect(to_robot_output->get_input_port(), robot_output_pub->get_input_port());
  
  auto sys = builder.Build();
  DrawAndSaveDiagramGraph(*diagram, "examples/franka_trajectory_following/diagram_run_ros_to_lcm");

  Simulator<double> simulator(*sys); 
  simulator.Initialize();
  simulator.set_target_realtime_rate(1.0);

  // figure out what the arguments to this mean
  ros::AsyncSpinner spinner(1);
  spinner.start();
  signal(SIGINT, SigintHandler);
  simulator.AdvanceTo(std::numeric_limits<double>::infinity());

  return 0;
}

} // namespace dairlib

int main(int argc, char* argv[]) { dairlib::DoMain(argc, argv);}