#include "residual_learner.h"

#include <utility>
#include <chrono>
using std::vector;
using drake::MatrixX;
using drake::systems::Context;

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace dairlib {
namespace systems {
using solvers::LCS;

namespace controllers {
ResidualLearner::ResidualLearner(){
  num_state = 19;
  num_velocity = 9;
  num_control = 3;
  num_lambda = 8;

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
