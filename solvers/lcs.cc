#include "solvers/lcs.h"
#include "common/eigen_utils.h"

#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/moby_lcp_solver.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/solve.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace dairlib {
namespace solvers {

LCS::LCS(const vector<MatrixXd>& A, const vector<MatrixXd>& B,
         const vector<MatrixXd>& D, const vector<VectorXd>& d,
         const vector<MatrixXd>& E, const vector<MatrixXd>& F,
         const vector<MatrixXd>& H, const vector<VectorXd>& c)
    : A_(A), B_(B), D_(D), d_(d), E_(E), F_(F), H_(H), c_(c), N_(A.size()) {}

LCS::LCS(const MatrixXd& A, const MatrixXd& B, const MatrixXd& D,
         const VectorXd& d, const MatrixXd& E, const MatrixXd& F,
         const MatrixXd& H, const VectorXd& c, const int& N)
    : LCS(vector<MatrixXd>(N, A), vector<MatrixXd>(N, B),
          vector<MatrixXd>(N, D), vector<VectorXd>(N, d),
          vector<MatrixXd>(N, E), vector<MatrixXd>(N, F),
          vector<MatrixXd>(N, H), vector<VectorXd>(N, c)) {}

VectorXd LCS::Simulate(VectorXd& x_init, VectorXd& input) {
  VectorXd x_final;

  // calculate force
  drake::solvers::MobyLCPSolver<double> LCPSolver;
  VectorXd force;

//  disturbance[3] = -5;
//  disturbance[4] = -5;
//  disturbance[5] = -5;

//VectorXd dummy_input = VectorXd::Zero(9);

  auto flag = LCPSolver.SolveLcpLemke(F_[0], E_[0] * x_init + c_[0] + H_[0] * input,
                          &force);

//  VectorXd qval = E_[0] * x_init + c_[0] + H_[0] * input;
//  MatrixXd Fval = F_[0];
////
////
//  auto flag = LCPSolver.SolveLcpLemkeRegularized(Fval,qval, &force);

//  std::cout << flag << std::endl;

//  if (flag == 1){
//
//      std::cout << "LCS force estimate" << std::endl;
//    std::cout << force << std::endl;
//    std::cout << "LCS force estimate" << std::endl;
//  }

//  std::cout << "eig" << std::endl;
//  std::cout << (F_[0] + F_[0].transpose()).eigenvalues() << std::endl;
//  std::cout << "eig" << std::endl;

//  std::cout << "gap" << std::endl;
//  std::cout << E_[0] * x_init + c_[0] + H_[0] * input + F_[0] * force << std::endl;
//  std::cout << "gap" << std::endl;

//  //print force
//  std::cout << "LCS force estimate" << std::endl;
//  std::cout << force << std::endl;
//  std::cout << "LCS force estimate" << std::endl;

//  double count = 0;
//  for (int i = 3; i < 5; i++) {
//    count = count + force(i);
//  }
//
//  if ( count >= 0.00001){
//    std::cout << "here" << std::endl;
//  }

  // update
  x_final = A_[0] * x_init + B_[0] * input + D_[0] * force + d_[0];

    if (flag == 1){




//      std::cout << "Next state prediction" << std::endl;
//    std::cout << x_final<< std::endl;
//    std::cout << "Next state prediction" << std::endl;

//    std::cout << "Jn * v" << std::endl;
//   std::cout << D_[0].block(10,0,9,9) * x_final_v << std::endl;
//    std::cout << "Jn * v" << std::endl;

//
//    std::cout << "LCS force estimate" << std::endl;
//   std::cout << force << std::endl;
//    std::cout << "LCS force estimate" << std::endl;

//      std::cout << "D" << std::endl;
//    std::cout << D_[0] << std::endl;
//      std::cout << "D" << std::endl;

//      std::cout << "D times LCS force estimate" << std::endl;
//    std::cout << D_[0] * force + d_[0] << std::endl;
//    std::cout << "D times LCS force estimate" << std::endl;


  }


  return x_final;
}

/// methods to move the LCS type back and forth from LCM, used for RobotLCSSender and any of the plant port
/// that need to declare a Abstract input of LCS class, currently mainly used to send the residual lcs

/// CopyLCSToLcm: used in the LCS sender, grab the matrices out from LCS class and send them to lcm type
void LCS::CopyLCSToLcm(lcmt_lcs *lcs_msg) const {
  lcs_msg->num_state = A_[0].cols();
  lcs_msg->num_velocity = A_[0].rows();
  lcs_msg->num_control = B_[0].cols();
  lcs_msg->num_lambda = D_[0].cols();

  for (int i = 0; i < A_[0].rows(); i++) {
      lcs_msg->A.push_back(
          CopyVectorXdToStdVector(A_[0].block(i, 0, 1, A_[0].cols()).transpose())
      );
      lcs_msg->B.push_back(
          CopyVectorXdToStdVector(B_[0].block(i, 0, 1, B_[0].cols()).transpose())
      );
      lcs_msg->D.push_back(
          CopyVectorXdToStdVector(D_[0].block(i, 0, 1, D_[0].cols()).transpose())
      );
    }
  lcs_msg->d = CopyVectorXdToStdVector(d_[0]);

  for (int i = 0; i < E_[0].rows(); i++) {
      lcs_msg->E.push_back(
          CopyVectorXdToStdVector(E_[0].block(i, 0, 1, E_[0].cols()).transpose())
      );
      lcs_msg->F.push_back(
          CopyVectorXdToStdVector(F_[0].block(i, 0, 1, F_[0].cols()).transpose())
      );
      lcs_msg->H.push_back(
          CopyVectorXdToStdVector(H_[0].block(i, 0, 1, H_[0].cols()).transpose())
      );
    }
  lcs_msg->c = CopyVectorXdToStdVector(c_[0]);
}

/// OutputLCS: used in the LCS receiver, grab the data from lcm type, recover the matrixXD and form an LCS class
LCS LCS::CopyLCSFromLcm(const lcmt_lcs& lcs_msg) {
  int num_state = lcs_msg.num_state;
  int num_velocity = lcs_msg.num_velocity;
  int num_control = lcs_msg.num_control;
  int num_lambda = lcs_msg.num_lambda;
  int N = 1;

  MatrixXd A = MatrixXd(num_velocity, num_state);
  MatrixXd B = MatrixXd(num_velocity, num_control);
  MatrixXd D = MatrixXd(num_velocity, num_lambda);

  for (int i = 0; i < num_velocity; ++i) {
    A.row(i) = VectorXd::Map(&lcs_msg.A[i][0], num_state);
    B.row(i) = VectorXd::Map(&lcs_msg.B[i][0], num_control);
    D.row(i) = VectorXd::Map(&lcs_msg.D[i][0], num_lambda);
  }
  VectorXd d = VectorXd::Map(lcs_msg.d.data(), num_velocity);

  MatrixXd E = MatrixXd(num_lambda, num_state);
  MatrixXd F = MatrixXd(num_lambda, num_lambda);
  MatrixXd H = MatrixXd(num_lambda, num_control);

  for (int i = 0; i < num_lambda; ++i) {
    E.row(i) = VectorXd::Map(&lcs_msg.E[i][0], num_state);
    F.row(i) = VectorXd::Map(&lcs_msg.F[i][0], num_lambda);
    H.row(i) = VectorXd::Map(&lcs_msg.H[i][0], num_control);
  }
  VectorXd c = VectorXd::Map(lcs_msg.c.data(), num_lambda);

  LCS lcs(A, B, D, d, E, F, H, c, N);
  return lcs;
}

}  // namespace solvers
}  // namespace dairlib
