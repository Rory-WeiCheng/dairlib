#include "alip_miqp.h"
#include "solvers/optimization_utils.h"
#include "drake/solvers/solve.h"
#include "drake/solvers/branch_and_bound.h"
#include <iostream>

namespace dairlib::systems::controllers {

using Eigen::VectorXd;
using Eigen::Vector4d;
using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::Matrix4d;
using Eigen::RowVector3d;
using Eigen::RowVectorXd;

using drake::solvers::GurobiSolver;
using solvers::LinearBigMConstraint;
using solvers::LinearBigMEqualityConstraint;

void AlipMIQP::AddMode() {
  std::string nm = std::to_string(nmodes_);
  pp_.push_back(prog_->NewContinuousVariables(np_, "pp_" + nm));
  xx_.push_back(prog_->NewContinuousVariables(nx_ * nknots_, "xx_" + nm));
  uu_.push_back(prog_->NewContinuousVariables(nu_ * (nknots_ - 1), "uu_" + nm));
  nmodes_ += 1;
}

void AlipMIQP::Build(const drake::solvers::SolverOptions& options) {
  prog_->SetSolverOptions(options);
  Build();
}

void AlipMIQP::Build() {
  DRAKE_ASSERT(td_.size() == nmodes_);
  tt_ = VectorXd::Zero(nmodes_);
  for (int i = 0; i < nmodes_; i++) {
    tt_(i) = td_.at(i);
  }
  MakeFootholdConstraints();
  MakeNoCrossoverConstraint();
  MakeResetConstraints();
  MakeDynamicsConstraints();
  MakeWorkspaceConstraints();
  MakeInputBoundConstaints();
  MakeNextFootstepReachabilityConstraint();
  MakeInitialStateConstraint();
  MakeInitialFootstepConstraint();
  built_ = true;
}

void AlipMIQP::MakeFootholdConstraints() {
  constexpr double bigM = 4.0;
  for (int j = 1; j < nmodes_; j++) {
    const std::string& js = std::to_string(j -1);
    zz_.push_back(prog_->NewBinaryVariables(kMaxFootholds, "zz" + js));
    vector<LinearBigMConstraint> tmp;
    vector<LinearBigMEqualityConstraint> tmp_eq;
    integer_sum_constraints_.push_back(
        prog_->AddLinearEqualityConstraint(
            RowVectorXd::Zero(kMaxFootholds),
            VectorXd::Zero(1), zz_.at(j-1)
        )
    );
    for (int i = 0; i < kMaxFootholds; i++) {
      const auto& z = zz_.at(j-1);
      tmp.push_back(LinearBigMConstraint(
          *prog_,
          RowVector3d::Zero(),
          VectorXd::Zero(1),
          bigM,
          pp_.at(j),
          z(i)));
      tmp_eq.push_back(LinearBigMEqualityConstraint(
          *prog_,
          RowVector3d::Zero(),
          VectorXd::Zero(1),
          bigM,
          pp_.at(j),
          z(i)));
    }
    foothold_constraints_.push_back(tmp);
    foothold_equality_constraints_.push_back(tmp_eq);
  }
}

void AlipMIQP::UpdateFootholdConstraints() {
  const int n = footholds_.size();
  for (int i = 0; i < n; i++) {
    const auto& [Aeq, beq] = footholds_.at(i).GetEqualityConstraintMatrices();
    const auto& [A, b] = footholds_.at(i).GetConstraintMatrices();
    for (int j = 0; j < nmodes_ - 1; j++) {
      foothold_constraints_.at(j).at(i).update(A, b);
      foothold_equality_constraints_.at(j).at(i).update(Aeq, beq);
    }
  }
  for (int i = n; i < kMaxFootholds; i++) {
    for (int j = 0; j < nmodes_ - 1; j++) {
      foothold_constraints_.at(j).at(i).deactivate(*prog_);
      foothold_equality_constraints_.at(j).at(i).deactivate(*prog_);
    }
  }
  RowVectorXd int_sum = RowVectorXd::Zero(kMaxFootholds);
  int_sum.block(0, 0, 1, n) = RowVectorXd::Ones(n);
  for (int j = 0; j < nmodes_ - 1; j++) {
    integer_sum_constraints_.at(j).evaluator()->UpdateCoefficients(
        int_sum, VectorXd::Ones(1));
  }
}

void AlipMIQP::SolveOCProblemAsIs() {
  solve_time_.start_ = std::chrono::steady_clock::now();
  UpdateFootholdConstraints();
  auto result =  solver_.Solve(*prog_);
  if (result.is_success()) {
    solution_.first = result;
    solution_.second = ExtractDynamicsConstraintDual(result);
  } else {
    std::cout << "solve failed with code " << result.get_solution_result() << std::endl;
  }
  solve_time_.finish_ = std::chrono::steady_clock::now();
  solve_time_.solve_time_ = result.get_solver_details<GurobiSolver>().optimizer_time;
}

void AlipMIQP::UpdateInitialGuess(const Eigen::Vector3d &p0,
                                  const Eigen::Vector4d &x0) {
  // Update state initial guess
  vector<VectorXd> xg = xd_;

  // Set the initial guess for the current mode based on limited time
  VectorXd xx = VectorXd (nx_ * nknots_);
  xx.head<4>() = x0;
  Matrix4d Ad = alip_utils::CalcAd(H_, m_, tt_(0) / (nknots_ - 1));
  for (int i = 1; i < nknots_; i++) {
    GetStateAtKnot(xx, i) = Ad * GetStateAtKnot(xx, i-1);
  }
  xg.front() = xx;

  for (int n = 0; n < nmodes_; n++) {
    for (int k = 0; k < nknots_ ; k++) {
      prog_->SetInitialGuess(
          GetStateAtKnot(xx_.at(n), k),
          GetStateAtKnot(xg.at(n), k));
    }
  }
  Vector3d ptemp = p0;
  prog_->SetInitialGuess(pp_.front(), p0);
  for(int n = 1; n < nmodes_; n++) {
    Vector2d p1 = (xd_.at(n-1).tail<4>() - xd_.at(n).head<4>()).head<2>() + ptemp.head<2>();
    prog_->SetInitialGuess(pp_.at(n).head<2>(), p1);
    ptemp.head<2>() = p1;
  }
}

void AlipMIQP::UpdateInitialGuess() {
  UpdateInitialGuess(Vector3d::Zero(), Vector4d::Zero());
}

}