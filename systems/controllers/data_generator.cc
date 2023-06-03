#include "data_generator.h"

#include <utility>
#include <chrono>

#include "external/drake/tools/install/libdrake/_virtual_includes/drake_shared_library/drake/common/sorted_pair.h"
#include "external/drake/tools/install/libdrake/_virtual_includes/drake_shared_library/drake/multibody/plant/multibody_plant.h"
#include "multibody/multibody_utils.h"
#include "solvers/lcs_factory_franka_ref.h"

#include "multibody/geom_geom_collider.h"
#include "multibody/kinematic/kinematic_evaluator_set.h"
#include "drake/math/autodiff_gradient.h"

using std::vector;

using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::MatrixX;
using drake::geometry::GeometryId;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;
using drake::multibody::JacobianWrtVariable;
using drake::math::RotationMatrix;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Quaterniond;

// task space helper function that generates the target trajectory
// modified version of code from impedance_controller.cc
namespace dairlib {
namespace systems {
using solvers::LearningData;
using solvers::LCS;
namespace controllers {

Data_Generator::Data_Generator(
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
    int num_friction_directions, double mu)
    : plant_(plant),
      plant_f_(plant_f),
      plant_franka_(plant_franka),
      context_(context),
      context_f_(context_f),
      context_franka_(context_franka),
      plant_ad_(plant_ad),
      plant_ad_f_(plant_ad_f),
      context_ad_(context_ad),
      context_ad_f_(context_ad_f),
      scene_graph_(scene_graph),
      diagram_(diagram),
      contact_geoms_(contact_geoms),
      num_friction_directions_(num_friction_directions),
      mu_(mu){
  franka_state_input_port_ =
      this->DeclareVectorInputPort("x, u, t", OutputVector<double>(14, 13, 7))
          .get_index();

  // c3 input port
  c3_state_input_port_ =
      this->DeclareVectorInputPort(
              "xee, xball, xee_dot, xball_dot, lambda, visualization, input",
              TimestampedVector<double>(41))
          .get_index();

  // output port: the state, input, prediction and reference LCS
  data_output_port_ =
      this->DeclareAbstractOutputPort("state, input, state_pred, Reference LCS",
                                      &Data_Generator::CalcData)
          .get_index();
  // get c3_parameters
  param_ = drake::yaml::LoadYamlFile<C3Parameters>(
      "examples/franka_trajectory_following/parameters.yaml");
}

void Data_Generator::CalcData(const Context<double>& context,
                              LearningData* data_pack) const {
  // get values
  auto robot_output =
      (OutputVector<double>*) this->EvalVectorInput(context, franka_state_input_port_);

  auto c3_output =
      (TimestampedVector<double>*) this->EvalVectorInput(context, c3_state_input_port_);
//  VectorXd c3_input = c3_stuff->get_data();

  double timestamp = robot_output->get_timestamp();
//
//  if (!received_first_message_) {
//    received_first_message_ = true;
//    first_message_time_ = timestamp;
//  }
//
//  double settling_time = param_.stabilize_time1 + param_.move_time + param_.stabilize_time2 + first_message_time_;
//  if (timestamp <= settling_time) {
//    // hard code for now, improve in the future
//    VectorXd state = VectorXd::Zero(19);
//    VectorXd u = VectorXd::Zero(3);
//    VectorXd state_next = VectorXd::Zero(19);
//
//    MatrixXd A = MatrixXd::Zero(9, 19);
//    MatrixXd B = MatrixXd::Zero(9, 3);
//    MatrixXd D = MatrixXd::Zero(9, 12);
//    VectorXd d = VectorXd::Zero(9);
//
//    MatrixXd E = MatrixXd::Zero(12, 19);
//    MatrixXd F = MatrixXd::Zero(12, 12);
//    MatrixXd H = MatrixXd::Zero(12, 3);
//    VectorXd c = VectorXd::Zero(12);
//
//    LCS LCS_model(A, B, D, d, E, F, H, c, 1);
//
//    LearningData data(state, u, state_next, LCS_model, timestamp);
//    *data_pack = data;
//    prev_timestamp_ = timestamp;
////    std::cout<< 'ok'<<std::endl;
//    return;
//  }
//
//  /// FK
//  // update context once for FK
//  plant_franka_.SetPositions(&context_franka_, robot_output->GetPositions());
//  plant_franka_.SetVelocities(&context_franka_, robot_output->GetVelocities());
//  Vector3d EE_offset_ = param_.EE_offset;
//  const drake::math::RigidTransform<double> H_mat =
//      plant_franka_.EvalBodyPoseInWorld(
//          context_franka_, plant_franka_.GetBodyByName("panda_link10"));
//  const RotationMatrix<double> R_current = H_mat.rotation();
//  Vector3d end_effector = H_mat.translation() + R_current * EE_offset_;
//
//  // jacobian and end_effector_dot
//  auto EE_frame_ = &plant_franka_.GetBodyByName("panda_link10").body_frame();
//  auto world_frame_ = &plant_franka_.world_frame();
//  MatrixXd J_fb(6, plant_franka_.num_velocities());
//  plant_franka_.CalcJacobianSpatialVelocity(
//      context_franka_, JacobianWrtVariable::kV, *EE_frame_, EE_offset_,
//      *world_frame_, *world_frame_, &J_fb);
//  MatrixXd J_franka = J_fb.block(0, 0, 6, 7);
//  VectorXd end_effector_dot =
//      (J_franka * (robot_output->GetVelocities()).head(7)).tail(3);
//
//  VectorXd q_plant = robot_output->GetPositions();
//  VectorXd v_plant = robot_output->GetVelocities();
//
//  // parse franka state info
//  VectorXd ball = q_plant.tail(7);
//  Vector3d ball_xyz = ball.tail(3);
//  VectorXd ball_dot = v_plant.tail(6);
//  Vector3d v_ball = ball_dot.tail(3);
//
//  VectorXd q(10);
//  q << end_effector, ball;
//  VectorXd v(9);
//  v << end_effector_dot, ball_dot;
//  VectorXd u = c3_input.tail(3);
//
//  VectorXd state(plant_.num_positions() + plant_.num_velocities());
//  state << end_effector, q_plant.tail(7), end_effector_dot, v_plant.tail(6);
//
//  /// update autodiff
//  VectorXd xu(plant_f_.num_positions() + plant_f_.num_velocities() +
//              plant_f_.num_actuators());
//  xu << q, v, u;
//  auto xu_ad = drake::math::InitializeAutoDiff(xu);
//
//  plant_ad_f_.SetPositionsAndVelocities(
//      &context_ad_f_,
//      xu_ad.head(plant_f_.num_positions() + plant_f_.num_velocities()));
//  multibody::SetInputsIfNew<AutoDiffXd>(
//      plant_ad_f_, xu_ad.tail(plant_f_.num_actuators()), &context_ad_f_);
//
//  /// upddate context
//  plant_f_.SetPositions(&context_f_, q);
//  plant_f_.SetVelocities(&context_f_, v);
//  multibody::SetInputsIfNew<double>(plant_f_, u, &context_f_);
//
//  double dt = timestamp - prev_timestamp_;
//
//  /// use the LCSFactoryFrankaRef which does not fix with residual lcs and scaling
//  auto system_scaling_pair =solvers::LCSFactoryFrankaRef::LinearizePlantToLCS(
//      plant_f_, context_f_, plant_ad_f_, context_ad_f_,
//      contact_geoms_, num_friction_directions_, mu_, dt);
//
//  LCS LCS_model = system_scaling_pair.first;
//  VectorXd state_next = LCS_model.Simulate(state, u);
//
//  LearningData data(state, u, state_next, LCS_model, timestamp);
//  *data_pack = data;
//  prev_timestamp_ = timestamp;
//  std::cout<< 'ok'<<std::endl;

  VectorXd state = VectorXd::Zero(19);
  VectorXd u = VectorXd::Zero(3);
  VectorXd state_next = VectorXd::Zero(19);

  MatrixXd A = MatrixXd::Zero(9, 19);
  MatrixXd B = MatrixXd::Zero(9, 3);
  MatrixXd D = MatrixXd::Zero(9, 12);
  VectorXd d = VectorXd::Zero(9);

  MatrixXd E = MatrixXd::Zero(12, 19);
  MatrixXd F = MatrixXd::Zero(12, 12);
  MatrixXd H = MatrixXd::Zero(12, 3);
  VectorXd c = VectorXd::Zero(12);

  LCS LCS_model(A, B, D, d, E, F, H, c, 1);
  double timestamp_test = 1;

  LearningData data(state, u, state_next, LCS_model, timestamp_test);
  *data_pack = data;
}
} // controllers
} // systems
} // dairlib