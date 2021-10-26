#include <drake/common/trajectories/exponential_plus_piecewise_polynomial.h>
#include "drake/systems/controllers/finite_horizon_linear_quadratic_regulator.h"
#include "systems/controllers/finite_horizon_lqr_swing_ft_traj_gen.h"
#include "systems/framework/output_vector.h"
#include "multibody/multibody_utils.h"

namespace dairlib::systems {

using drake::multibody::JacobianWrtVariable;
using drake::multibody::MultibodyPlant;

using drake::systems::LinearSystem;
using drake::systems::Context;
using drake::systems::BasicVector;
using drake::systems::controllers::FiniteHorizonLinearQuadraticRegulator;
using drake::systems::controllers::FiniteHorizonLinearQuadraticRegulatorOptions;
using drake::systems::controllers::FiniteHorizonLinearQuadraticRegulatorResult;

using drake::trajectories::PiecewisePolynomial;
using drake::trajectories::Trajectory;
using drake::trajectories::ExponentialPlusPiecewisePolynomial;

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Matrix;

using multibody::SetPositionsAndVelocitiesIfNew();

FiniteHorizonLqrSwingFootTrajGenerator::FiniteHorizonLqrSwingFootTrajGenerator(
    const MultibodyPlant<double> &plant,
    const drake::systems::LinearSystem<double> &double_integrator,
    const std::vector<int> left_right_support_fsm_states,
    const std::vector<double> left_right_support_durations,
    const std::vector<std::pair<const Eigen::Vector3d,
                                const drake::multibody::Frame<double> &>> pts,
    const SwingFootTajGenOptions opts) :
    plant_(plant),
    plant_context_(plant_.CreateDefaultContext()),
    double_integrator_(double_integrator),
    double_integrator_context_(double_integrator_.CreateDefaultContext()),
    opts_(opts),
    left_right_support_fsm_states_(left_right_support_fsm_states),
    left_right_support_durations_(left_right_support_durations),
    pts_(pts) {

  state_port_ = this->DeclareVectorInputPort(
      "x, u, t", OutputVector<double>(
          plant.num_positions(),
          plant.num_velocities(),
          plant.num_actuators())).get_index();
  fsm_port_ =
      this->DeclareVectorInputPort(
          "fsm", BasicVector<double>(1)).get_index();
  liftoff_time_port_ =
      this->DeclareVectorInputPort(
          "t_liftoff", BasicVector<double>(1)).get_index();

  foot_target_port_ =
      this->DeclareVectorInputPort(
          "foot_target", BasicVector<double>(3)).get_index();

  PiecewisePolynomial<double> pp(VectorXd::Zero(0));
  drake::trajectories::Trajectory<double> &traj_instance = pp;

  this->DeclareAbstractOutputPort("swing_foot_xyz", traj_instance,
                                  &FiniteHorizonLqrSwingFootTrajGenerator::CalcTrajs);
  // Construct maps
  duration_map_.insert({left_right_support_fsm_states.at(0),
                        left_right_support_durations.at(0)});
  duration_map_.insert({left_right_support_fsm_states.at(1),
                        left_right_support_durations.at(1)});
  swing_foot_map_.insert(
      {left_right_support_fsm_states.at(0), pts.at(1)});
  swing_foot_map_.insert(
      {left_right_support_fsm_states.at(1), pts.at(0)});
  stance_foot_map_.insert(
      {left_right_support_fsm_states.at(0), pts.at(0)});
  stance_foot_map_.insert(
      {left_right_support_fsm_states.at(1), pts.at(1)});

}

void FiniteHorizonLqrSwingFootTrajGenerator::CalcTrajs(
    const Context<double> &context, Trajectory<double> *traj) const {
  auto *pp_traj =
      (PiecewisePolynomial<double> *) dynamic_cast<PiecewisePolynomial<double> *>(
          traj);

  *pp_traj = PiecewisePolynomial<double>(Vector3d::Zero());
  int fsm_state = this->EvalVectorInput(context, fsm_port_)->get_value()[0];

  // swing phase if current state is in left_right_support_fsm_states_
  auto it = find(left_right_support_fsm_states_.begin(),
                 left_right_support_fsm_states_.end(), fsm_state);
  bool is_single_support_phase = it != left_right_support_fsm_states_.end();

  if (is_single_support_phase) {
    // get current state and time
    const OutputVector<double>* robot_output =
      (OutputVector<double>*)this->EvalVectorInput(context, state_port_);
    const Vector2d pdes =
        this->EvalVectorInput(context, foot_target_port_)->get_value().head(2);
    const Vector4d x_di = CalcSwingFootState(robot_output->GetState(),
        swing_foot_map_.at(fsm_state));

    double timestamp = robot_output->get_timestamp();
    double liftoff_time =
        this->EvalVectorInput(context, liftoff_time_port_)->get_value()[0];
    double end_time = liftoff_time + duration_map_.at(fsm_state);
    Vector4d xd_vec;
    xd_vec << pdes, Vector2d::Zero();

    PiecewisePolynomial<double> xd = PiecewisePolynomial<double>(xd_vec);
    auto opts = FiniteHorizonLinearQuadraticRegulatorOptions();
    opts.xd = &xd;
    auto result = FiniteHorizonLinearQuadraticRegulator(
        double_integrator_, *double_integrator_context_,
        timestamp, end_time, Q_, R_, opts);

    auto u =
        result.K->value(timestamp) * (x_di - xd_vec) -result.k0->value(timestamp);
    *pp_traj = CreateSplineForSwingFoot(
        liftoff_time, end_time, timestamp, x_di, u,
        CalcStanceFootHeight(
            robot_output->GetState(),stance_foot_map_.at(fsm_state)));
  }
}

PiecewisePolynomial<double>
    FiniteHorizonLqrSwingFootTrajGenerator::CreateSplineForSwingFoot(
        const double start_time_of_this_interval,
        const double end_time_of_this_interval,
        const double timestamp,
        const Eigen::Vector4d& init_swing_foot_xy_state,
        const Eigen::Vector2d& u,
        double stance_foot_height) const {
  VectorXd start = VectorXd::Zero(6);
  VectorXd end = VectorXd::Zero(6);

  Vector2d breaks(start_time_of_this_interval,
                  end_time_of_this_interval);

  double dt = end_time_of_this_interval - start_time_of_this_interval;
  
}

Eigen::Vector4d FiniteHorizonLqrSwingFootTrajGenerator::CalcSwingFootState(
    const VectorXd& x,
    const std::pair<
        const Eigen::Vector3d, const drake::multibody::Frame<double>&> pt) const {
  SetPositionsAndVelocitiesIfNew<double>(plant_, x, plant_context_.get());
  Vector3d pos;
  MatrixXd jac = MatrixXd::Zero(3, plant_.num_velocities());
  plant_.CalcPointsPositions(
      *plant_context_, pt.second, pt.first,
      plant_.world_frame(), &pos);
  plant_.CalcJacobianTranslationalVelocity(
      *plant_context_, JacobianWrtVariable::kV,
      pt.second, pt.first,
      plant_.world_frame(), plant_.world_frame(), &jac);
  Vector3d vel = jac * x.tail(plant_.num_velocities());
  Vector4d x_di;
  x_di << pos.head(2), vel.head((2));
  return x_di;
}

double FiniteHorizonLqrSwingFootTrajGenerator::CalcStanceFootHeight(
    const Eigen::VectorXd& x,
    const std::pair<
    const Eigen::Vector3d, const drake::multibody::Frame<double>&> pt) const {
  SetPositionsAndVelocitiesIfNew<double>(plant_, x, plant_context_.get());
  Vector3d pos;
  plant_.CalcPointsPositions(
      *plant_context_, pt.second, pt.first,
      plant_.world_frame(), &pos);
  return pos(2);
}

}