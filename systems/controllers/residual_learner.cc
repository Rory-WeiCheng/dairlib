#include "residual_learner.h"

#include <utility>
#include <chrono>


#include "external/drake/tools/install/libdrake/_virtual_includes/drake_shared_library/drake/common/sorted_pair.h"
#include "external/drake/tools/install/libdrake/_virtual_includes/drake_shared_library/drake/multibody/plant/multibody_plant.h"
#include "multibody/multibody_utils.h"
#include "solvers/c3.h"
#include "solvers/c3_miqp.h"
#include "solvers/lcs_factory.h"
#include "solvers/lcs.h"

#include "drake/solvers/moby_lcp_solver.h"
#include "multibody/geom_geom_collider.h"
#include "multibody/kinematic/kinematic_evaluator_set.h"
#include "drake/math/autodiff_gradient.h"

using std::vector;

using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::MatrixX;
using drake::SortedPair;
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

namespace dairlib {
namespace systems {
using solvers::LCS;

namespace controllers {
ResidualLearner::ResidualLearner(){
  num_state = 19;
  num_velocity = 9;
  num_control = 3;
  num_lambda = 12;

  state_input_port_ =
      this->DeclareVectorInputPort(
              "x, u, t",
              OutputVector<double>(14, 13, 7))
          .get_index();

  lcs_output_port_ =
      this->DeclareAbstractOutputPort(
             "Residual LCS",
             &ResidualLearner::CalcResidual)
      .get_index();
}

void ResidualLearner::CalcResidual(const Context<double>& context,
                                      LCS* residual_lcs) const {

  // get lcm_robot_output values
  auto robot_output = (OutputVector<double>*)this->EvalVectorInput(context, state_input_port_);
  double timestamp = robot_output->get_timestamp();

  int N = 1;
  MatrixXd A = 0 * MatrixXd::Identity(num_velocity, num_state);
  MatrixXd B = 0 * MatrixXd::Identity(num_velocity, num_control);
  MatrixXd D = 0 * MatrixXd::Identity(num_velocity, num_lambda);
  VectorXd d = 0 * VectorXd::Ones(num_velocity);

  MatrixXd E = 0 * MatrixXd::Identity(num_lambda, num_state);
  MatrixXd F = 0 * MatrixXd::Identity(num_lambda, num_lambda);
  MatrixXd H = 0 * MatrixXd::Identity(num_lambda, num_control);
  VectorXd c = 0 * VectorXd::Ones(num_lambda);

  LCS res_lcs(A, B, D, d, E, F, H, c, N);
  *residual_lcs = res_lcs;
}

}
}
}
