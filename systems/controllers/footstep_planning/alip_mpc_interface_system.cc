#include "alip_mpc_interface_system.h"

#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>

#include "multibody/multibody_utils.h"
#include "systems/controllers/minimum_snap_trajectory_generation.h"

#include "drake/math/saturate.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/common/trajectories/bspline_trajectory.h"
#include "drake/common/trajectories/path_parameterized_trajectory.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/pass_through.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"

using dairlib::systems::controllers::alip_utils::PointOnFramed;

using std::endl;
using std::string;

using Eigen::MatrixXd;
using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using drake::Vector1d;

using drake::multibody::Frame;
using drake::multibody::JacobianWrtVariable;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::ConstantVectorSource;
using drake::systems::PassThrough;
using drake::systems::Adder;
using drake::systems::DiscreteUpdateEvent;
using drake::systems::DiscreteValues;
using drake::systems::EventStatus;
using drake::trajectories::PiecewisePolynomial;
using drake::trajectories::BsplineTrajectory;
using drake::trajectories::PathParameterizedTrajectory;
using drake::trajectories::Trajectory;


namespace dairlib {
namespace systems {
namespace controllers {

SwingFootInterfaceSystem::SwingFootInterfaceSystem(
    const drake::multibody::MultibodyPlant<double> &plant,
    drake::systems::Context<double> *context,
    const SwingFootInterfaceSystemParams& params)
    : plant_(plant),
      plant_context_(context),
      world_(plant_.world_frame()),
      left_right_support_fsm_states_(params.left_right_support_fsm_states),
      com_height_(params.com_height_),
      mid_foot_height_(params.mid_foot_height),
      desired_final_foot_height_(params.desired_final_foot_height),
      foot_height_offset_(params.foot_height_offset_),
      desired_final_vertical_foot_velocity_(
          params.desired_final_vertical_foot_velocity),
      relative_to_com_(params.relative_to_com) {

  this->set_name("swing_ft_traj");
  DRAKE_DEMAND(left_right_support_fsm_states_.size() == 2);
  DRAKE_DEMAND(params.left_right_foot.size() == 2);

  // Input/Output Setup
  state_port_ = this->DeclareVectorInputPort(
          "x, u, t", OutputVector<double>(plant.num_positions(),
                                          plant.num_velocities(),
                                          plant.num_actuators()))
      .get_index();
  fsm_port_ = this->DeclareVectorInputPort("fsm", 1).get_index();
  liftoff_time_port_ =
      this->DeclareVectorInputPort("t_liftoff", 1).get_index();
  touchdown_time_port_ =
      this->DeclareVectorInputPort("t_touchdown", 1).get_index();
  footstep_target_port_ =
      this->DeclareVectorInputPort("desired footstep target", 3).get_index();

  // Provide an instance to allocate the memory first (for the output)
  PathParameterizedTrajectory<double> pp(
      PiecewisePolynomial<double>(VectorXd::Zero(1)),
      PiecewisePolynomial<double>(VectorXd::Zero(1))
  );
  drake::trajectories::Trajectory<double> &traj_instance = pp;

  swing_foot_traj_output_port_ = this->DeclareAbstractOutputPort(
     "swing_foot_xyz", traj_instance, &SwingFootInterfaceSystem::CalcSwingTraj)
     .get_index();

  com_height_offset_output_port_ = this->DeclareVectorOutputPort(
      "com_z_next_td_offset", 1, &SwingFootInterfaceSystem::CopyComHeightOffset)
      .get_index();

  // State variables inside this controller block
  DeclarePerStepDiscreteUpdateEvent(
      &SwingFootInterfaceSystem::DiscreteVariableUpdate);

  // The swing foot position in the beginning of the swing phase
  liftoff_swing_foot_pos_idx_ = this->DeclareDiscreteState(3);

  // The last state of FSM
  prev_fsm_state_idx_ = this->DeclareDiscreteState(
      -std::numeric_limits<double>::infinity() * VectorXd::Ones(1));

  // Construct maps
  stance_foot_map_.insert(
      {params.left_right_support_fsm_states.at(0), params.left_right_foot.at(0)});
  stance_foot_map_.insert(
      {params.left_right_support_fsm_states.at(1), params.left_right_foot.at(1)});
  stance_foot_map_.insert(
      {params.post_left_post_right_fsm_states.at(0), params.left_right_foot.at(0)});
  stance_foot_map_.insert(
      {params.post_left_post_right_fsm_states.at(1), params.left_right_foot.at(1)});
  swing_foot_map_.insert(
      {params.left_right_support_fsm_states.at(0), params.left_right_foot.at(1)});
  swing_foot_map_.insert(
      {params.left_right_support_fsm_states.at(1), params.left_right_foot.at(0)});
}

EventStatus SwingFootInterfaceSystem::DiscreteVariableUpdate(
    const Context<double> &context,
    DiscreteValues<double> *discrete_state) const {
  // Read from ports
  int fsm_state = EvalVectorInput(context, fsm_port_)->get_value()(0);
  const auto robot_output = dynamic_cast<const OutputVector<double>*>(
      EvalVectorInput(context, state_port_));

  auto prev_fsm_state = discrete_state->get_mutable_value(prev_fsm_state_idx_);

  // when entering a new state which is in left_right_support_fsm_states
  if (fsm_state != prev_fsm_state(0) && is_single_support(fsm_state)) {
    prev_fsm_state(0) = fsm_state;

    VectorXd q = robot_output->GetPositions();
    multibody::SetPositionsIfNew<double>(plant_, q, plant_context_);
    auto swing_foot_pos_at_liftoff = discrete_state->get_mutable_vector(
        liftoff_swing_foot_pos_idx_).get_mutable_value();

    auto swing_foot = swing_foot_map_.at(fsm_state);
    plant_.CalcPointsPositions(*plant_context_, swing_foot.second, swing_foot.first,
                               world_, &swing_foot_pos_at_liftoff);

    if (relative_to_com_) {
      swing_foot_pos_at_liftoff =
          multibody::ReExpressWorldVector3InBodyYawFrame(
              plant_, *plant_context_, "pelvis",
              swing_foot_pos_at_liftoff -
                  plant_.CalcCenterOfMassPositionInWorld(*plant_context_));
    }
  }
  return EventStatus::Succeeded();
}

drake::trajectories::PathParameterizedTrajectory<double>
SwingFootInterfaceSystem::CreateSplineForSwingFoot(
    double start_time, double end_time, const Vector3d &init_pos,
    const Vector3d &final_pos) const {

  const Vector2d time_scaling_breaks(start_time, end_time);
  auto time_scaling_trajectory = PiecewisePolynomial<double>::FirstOrderHold(
      time_scaling_breaks, Vector2d(0, 1).transpose());

  std::vector<double> path_breaks = {0, 0.5, 1.0};
  Eigen::Matrix3d control_points = Matrix3d::Zero();
  control_points.col(0) = init_pos;
  control_points.col(2) = final_pos;
  double hdiff = final_pos(2) - init_pos(2);
  double tadj = 0.25;

  if (hdiff > mid_foot_height_ / 4.0) {
    control_points.col(1) = init_pos + (init_pos - final_pos) * 2 * tadj /
                            (end_time - start_time);
    control_points.col(1)(2) = final_pos(2) + mid_foot_height_ ;
    path_breaks.at(1) = tadj;
  } else if (-hdiff > mid_foot_height_ / 4.0) {
    control_points.col(1) = final_pos;
    control_points.col(1)(2) = init_pos(2) + mid_foot_height_ ;
    path_breaks.at(1) = 1.0 - tadj;
  } else {
    control_points.col(1) = 0.5 * (init_pos + final_pos);
    control_points.col(1)(2) += mid_foot_height_;
  }
  control_points.rightCols<1>()(2) += desired_final_foot_height_;
  auto swing_foot_path = minsnap::MakeMinSnapTrajFromWaypoints(
      control_points, path_breaks, desired_final_vertical_foot_velocity_);

  auto swing_foot_spline = PathParameterizedTrajectory<double>(
      swing_foot_path, time_scaling_trajectory);

  return swing_foot_spline;
}

bool SwingFootInterfaceSystem::is_single_support(int fsm_state) const {
  // Find fsm_state in left_right_support_fsm_states
  auto it = find(left_right_support_fsm_states_.begin(),
                 left_right_support_fsm_states_.end(), fsm_state);

  // swing phase if current state is in left_right_support_fsm_states_
  bool is_single_support_phase = it != left_right_support_fsm_states_.end();
  return is_single_support_phase;
}

void SwingFootInterfaceSystem::CalcSwingTraj(
    const Context<double> &context,
    drake::trajectories::Trajectory<double> *traj) const {

  // Get discrete states
  const auto swing_foot_pos_at_liftoff =
      context.get_discrete_state(liftoff_swing_foot_pos_idx_).get_value();
  // Read in finite state machine switch time
  double liftoff_time =
      EvalVectorInput(context, liftoff_time_port_)->get_value()(0);
  double touchdown_time =
      EvalVectorInput(context, touchdown_time_port_)->get_value()(0);

  // Read in finite state machine
  int fsm_state = this->EvalVectorInput(context, fsm_port_)->get_value()(0);

  // Generate trajectory if it's currently in swing phase.
  // Otherwise, generate a constant trajectory
  if (is_single_support(fsm_state)) {
    // Ensure current_time < end_time_of_this_interval to avoid error in
    // creating trajectory.
    double start_time_of_this_interval = std::clamp(
        liftoff_time, -std::numeric_limits<double>::infinity(),
        touchdown_time - 0.001);

    // Swing foot position at touchdown
    Vector3d footstep_target =
        this->EvalVectorInput(context, footstep_target_port_)->get_value();

    if (relative_to_com_) {
      footstep_target(2) = -com_height_;
    }

    // Assign traj
    auto pp_traj = dynamic_cast<PathParameterizedTrajectory<double> *>(traj);
    *pp_traj = CreateSplineForSwingFoot(
        start_time_of_this_interval, touchdown_time,
        swing_foot_pos_at_liftoff, footstep_target);
  } else {
    // Assign a constant traj
    auto pp_traj = dynamic_cast<PathParameterizedTrajectory<double> *>(traj);
    *pp_traj = PathParameterizedTrajectory<double>(
        PiecewisePolynomial<double>(Vector3d::Zero()),
        PiecewisePolynomial<double>(Vector1d::Ones())
    );
  }
}

void SwingFootInterfaceSystem::CopyComHeightOffset(
    const Context<double> &context,
    BasicVector<double> *com_height_offset) const {
  const auto robot_output = dynamic_cast<const OutputVector<double>*>(
      EvalVectorInput(context, state_port_));
  int fsm_state = this->EvalVectorInput(context, fsm_port_)->get_value()(0);
  const VectorXd& q = robot_output->GetPositions();
  multibody::SetPositionsIfNew<double>(plant_, q, plant_context_);
  Vector3d stance_foot_pos = Vector3d::Zero();

  plant_.CalcPointsPositions(
      *plant_context_, stance_foot_map_.at(fsm_state).second,
      stance_foot_map_.at(fsm_state).first, world_, &stance_foot_pos);

  // Swing foot position at touchdown
  const Vector3d& footstep_target =
      this->EvalVectorInput(context, footstep_target_port_)->get_value();
  double offset = footstep_target(2) - stance_foot_pos(2) + foot_height_offset_;
  com_height_offset->set_value(drake::Vector1d(offset));
}

AlipMPCInterfaceSystem::AlipMPCInterfaceSystem(
    const drake::multibody::MultibodyPlant<double> &plant,
    drake::systems::Context<double> *context,
    ALIPTrajGeneratorParams com_params,
    SwingFootInterfaceSystemParams swing_params) {

  drake::systems::DiagramBuilder<double> builder;
  auto swing_interface =
      builder.AddSystem<SwingFootInterfaceSystem>(plant, context, swing_params);
  auto com_interface =
      builder.AddSystem<ALIPTrajGenerator>(plant, context, com_params);
  auto com_height_source = builder.AddSystem<ConstantVectorSource<double>>(
      com_params.desired_com_height);
  auto com_height_to_traj_gen = builder.AddSystem<Adder<double>>(2, 1);

  // Connect com traj
  builder.Connect(swing_interface->get_output_port_com_height_offset(),
                  com_height_to_traj_gen->get_input_port(0));
  builder.Connect(com_height_source->get_output_port(),
                  com_height_to_traj_gen->get_input_port(1));
  builder.Connect(com_height_to_traj_gen->get_output_port(),
                  com_interface->get_input_port_target_com_z());

  // Export ports
  state_port_ = ExportSharedInput(
      builder,
      swing_interface->get_input_port_state(),
      com_interface->get_input_port_state(),
      "x, u, t");

  fsm_port_ = ExportSharedInput(
      builder,
      swing_interface->get_input_port_fsm(),
      com_interface->get_input_port_fsm(),
      "fsm");

  next_touchdown_time_port_ = ExportSharedInput(
      builder,
      swing_interface->get_input_port_next_fsm_switch_time(),
      com_interface->get_input_port_next_fsm_switch_time(),
      "tnext");

  prev_liftoff_time_port_ = ExportSharedInput(
      builder,
      swing_interface->get_input_port_fsm_switch_time(),
      com_interface->get_input_port_fsm_switch_time(),
      "tprev");

  footstep_target_port_ =
      builder.ExportInput(swing_interface->get_input_port_footstep_target());
  com_traj_port_ = builder.ExportOutput(com_interface->get_output_port_com());
  swing_traj_port_ =
      builder.ExportOutput(swing_interface->get_output_port_swing_foot_traj());

  builder.BuildInto(this);

}

const drake::systems::InputPortIndex AlipMPCInterfaceSystem::ExportSharedInput(
    drake::systems::DiagramBuilder<double>& builder,
    const drake::systems::InputPort<double> &p1,
    const drake::systems::InputPort<double> &p2, std::string name) {

  const drake::systems::InputPortIndex idx = builder.ExportInput(p1, name);
  builder.ConnectInput(name, p2);
  return idx;
}

}  // namespace controllers
}  // namespace systems
}  // namespace dairlib
