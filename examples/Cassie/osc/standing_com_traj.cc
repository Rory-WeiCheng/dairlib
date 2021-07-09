#include "examples/Cassie/osc/standing_com_traj.h"

#include <math.h>

#include <dairlib/lcmt_cassie_out.hpp>
#include <drake/math/saturate.h>

#include "multibody/multibody_utils.h"

using std::cout;
using std::endl;

using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

using dairlib::systems::OutputVector;
using drake::multibody::Frame;
using drake::multibody::MultibodyPlant;
using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::LeafSystem;
using drake::trajectories::PiecewisePolynomial;

namespace dairlib {
namespace cassie {
namespace osc {

StandingComTraj::StandingComTraj(
    const MultibodyPlant<double>& plant,
    Context<double>* context,
    const std::vector<std::pair<const Vector3d, const Frame<double>&>>&
        feet_contact_points,
    double height, bool use_radio)
    : plant_(plant),
      context_(context),
      world_(plant_.world_frame()),
      feet_contact_points_(feet_contact_points),
      height_(height),
      use_radio_(use_radio){
  // Input/Output Setup
  state_port_ =
      this->DeclareVectorInputPort(OutputVector<double>(plant.num_positions(),
                                                        plant.num_velocities(),
                                                        plant.num_actuators()))
          .get_index();
  target_height_port_ =
      this->DeclareAbstractInputPort(
              "lcmt_target_standing_height",
              drake::Value<dairlib::lcmt_target_standing_height>{})
          .get_index();
  radio_port_ =
      this->DeclareAbstractInputPort("lcmt_cassie_output",
                                     drake::Value<dairlib::lcmt_cassie_out>{})
          .get_index();
  // Provide an instance to allocate the memory first (for the output)
  PiecewisePolynomial<double> pp(VectorXd(0));
  drake::trajectories::Trajectory<double>& traj_inst = pp;
  this->DeclareAbstractOutputPort("com_traj", traj_inst,
                                  &StandingComTraj::CalcDesiredTraj);
}

void StandingComTraj::CalcDesiredTraj(
    const Context<double>& context,
    drake::trajectories::Trajectory<double>* traj) const {
  // Read in current state
  const OutputVector<double>* robot_output =
      (OutputVector<double>*)this->EvalVectorInput(context, state_port_);
  const auto& cassie_out =
      this->EvalInputValue<dairlib::lcmt_cassie_out>(context, radio_port_);


  double target_height = height_;

  // Get target height from radio or lcm
  if (use_radio_) {
    target_height = kTargetHeightMean + kTargetHeightScale * cassie_out->pelvis.radio.channel[6];
  } else {
    if (this->EvalInputValue<dairlib::lcmt_target_standing_height>(
        context, target_height_port_)->timestamp > 1e-3) {
      target_height = this->EvalInputValue<dairlib::lcmt_target_standing_height>(
          context, target_height_port_)->target_height;
    }
  }

  // Add offset position from sticks
  target_height += kHeightScale * cassie_out->pelvis.radio.channel[0];

  // Saturate based on min and max height
  target_height = drake::math::saturate(target_height, kMinHeight, kMaxHeight);
  double x_offset = kCoMXScale * cassie_out->pelvis.radio.channel[4];
  double y_offset = kCoMYScale * cassie_out->pelvis.radio.channel[5];
  VectorXd q = robot_output->GetPositions();

  multibody::SetPositionsIfNew<double>(plant_, q, context_);

  // Get center of left/right feet contact points positions
  Vector3d contact_pos_sum = Vector3d::Zero();
  Vector3d position;
  MatrixXd J(3, plant_.num_velocities());
  for (const auto& point_and_frame : feet_contact_points_) {
    plant_.CalcPointsPositions(*context_, point_and_frame.second,
                               point_and_frame.first, world_, &position);
    contact_pos_sum += position;
  }
  Vector3d feet_center_pos = contact_pos_sum / 4;
  Vector3d desired_com_pos(0.0 + x_offset,
                           0.0 + y_offset,
                           0.0 + target_height);

  // Assign traj
  PiecewisePolynomial<double>* pp_traj =
      (PiecewisePolynomial<double>*)dynamic_cast<PiecewisePolynomial<double>*>(
          traj);
  *pp_traj = PiecewisePolynomial<double>(desired_com_pos);
}

}  // namespace osc
}  // namespace cassie
}  // namespace dairlib
