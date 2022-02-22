#pragma once
#include <vector>
#include <Eigen/Dense>
#include "solvers/c3.h"
#include "solvers/lcs.h"
#include "gurobi_c++.h"


namespace dairlib {
namespace solvers {

class C3MIQP:public C3 {
 public:
	/// Default constructor for time-varying LCS
	C3MIQP(const LCS& LCS, const std::vector<Eigen::MatrixXd>& Q, const std::vector<Eigen::MatrixXd>& R, const std::vector<Eigen::MatrixXd>& G, const std::vector<Eigen::MatrixXd>& U,
         const C3Options& options);

	/// Virtual method
    Eigen::VectorXd SolveSingleProjection(const Eigen::MatrixXd& U, const Eigen::VectorXd& delta_c, const Eigen::MatrixXd& E, const Eigen::MatrixXd& F, const Eigen::MatrixXd& H, const Eigen::VectorXd& c);

 private:

};

} // namespace dairlib
} // namespace solvers