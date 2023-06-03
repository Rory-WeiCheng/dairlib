#include "solvers/learning_data.h"
#include "common/eigen_utils.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

namespace dairlib {
namespace solvers {

LearningData::LearningData(const Eigen::VectorXd& state,
         const Eigen::VectorXd& input,
         const Eigen::VectorXd& state_pred,
         const LCS& LCS_model,
         const double& timestamp)
    : state_(state), input_(input), state_pred_(state_pred), LCS_model_(LCS_model), timestamp_(timestamp) {}

/// methods to move the LearningData type back and forth from LCM, used for RobotLearningDataSender
/// CopyLCSToLcm: used in the LCS sender, grab the matrices out from LCS class and send them to lcm type
void LearningData::CopyLearningDataToLcm(lcmt_learning_data *msg) const {
  msg->num_state = LCS_model_.A_[0].cols();
  msg->num_velocity = LCS_model_.A_[0].rows();
  msg->num_control = LCS_model_.B_[0].cols();
  msg->num_lambda = LCS_model_.D_[0].cols();

  for (int i = 0; i < LCS_model_.A_[0].rows(); i++) {
      msg->A.push_back(
          CopyVectorXdToStdVector(LCS_model_.A_[0].block(i, 0, 1, LCS_model_.A_[0].cols()).transpose())
      );
      msg->B.push_back(
          CopyVectorXdToStdVector(LCS_model_.B_[0].block(i, 0, 1, LCS_model_.B_[0].cols()).transpose())
      );
      msg->D.push_back(
          CopyVectorXdToStdVector(LCS_model_.D_[0].block(i, 0, 1, LCS_model_.D_[0].cols()).transpose())
      );
    }
  msg->d = CopyVectorXdToStdVector(LCS_model_.d_[0]);

  for (int i = 0; i < LCS_model_.E_[0].rows(); i++) {
      msg->E.push_back(
          CopyVectorXdToStdVector(LCS_model_.E_[0].block(i, 0, 1, LCS_model_.E_[0].cols()).transpose())
      );
      msg->F.push_back(
          CopyVectorXdToStdVector(LCS_model_.F_[0].block(i, 0, 1, LCS_model_.F_[0].cols()).transpose())
      );
      msg->H.push_back(
          CopyVectorXdToStdVector(LCS_model_.H_[0].block(i, 0, 1, LCS_model_.H_[0].cols()).transpose())
      );
    }
  msg->c = CopyVectorXdToStdVector(LCS_model_.c_[0]);

  msg->state = CopyVectorXdToStdVector(state_);
  msg->input = CopyVectorXdToStdVector(input_);
  msg->state_pred = CopyVectorXdToStdVector(state_pred_);
  msg->utime = timestamp_ * 1e6;
}

}  // namespace solvers
}  // namespace dairlib
