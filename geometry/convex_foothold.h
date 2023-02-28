#pragma once
#include "Eigen/Dense"

namespace dairlib::geometry {

constexpr int kMaxFootholdFaces = 10;

/// Class representing a convex foothold consisting of a single
/// equality constraint ax = b defining the contact plane, and up to
/// kMaxFootholdFaces inequality constraints defining the extents of the
/// foothold. No effort is made to check feasibility or reasonableness of any
/// combination of constraints. No frame information is supplied, so
/// use responsibly
class ConvexFoothold {
 public:
  ConvexFoothold()= default;

  /*
   * Set the contact plane by supplying a normal and a point on the
   * contact plane
   */
  void SetContactPlane(Eigen::Vector3d normal, Eigen::Vector3d pt);

  /*
   * Add a constraint ax <= b to the convex foothold
   */
  void AddHalfspace(Eigen::Vector3d a, Eigen::VectorXd b);

  /*
   * Add a face with an outward facing normal which intersects with
   * pt
   */
  void AddFace(const Eigen::Vector3d& normal, const Eigen::Vector3d& pt);

  /*
   * Add a face by adding two vertices. v1 and v2 should be unique points
   * in the contact plane, and with the contact normal pointing toward the
   * observer, v2 should be counterclockwise from v1. These conditions are not
   * checked
   */
  void AddVertices(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);
  std::pair<Eigen::MatrixXd, Eigen::VectorXd>GetConstraintMatrices() const;
  std::pair<Eigen::MatrixXd, Eigen::VectorXd>GetEqualityConstraintMatrices() const;
  std::vector<Eigen::Vector3d> GetVertices();
  void ReExpressInNewFrame(const Eigen::Matrix3d& R_WF);

  static ConvexFoothold MakeFlatGround() {
    ConvexFoothold foothold;
    foothold.SetContactPlane(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero());
    foothold.AddFace(Eigen::Vector3d::UnitX(), 100 * Eigen::Vector3d::UnitX());
    foothold.AddFace(-Eigen::Vector3d::UnitX(), -100 * Eigen::Vector3d::UnitX());
    foothold.AddFace(Eigen::Vector3d::UnitY(), 100 * Eigen::Vector3d::UnitY());
    foothold.AddFace(-Eigen::Vector3d::UnitY(), -100 * Eigen::Vector3d::UnitY());
    return foothold;
  }

 private:
  Eigen::Vector3d SolveForVertexSharedByFaces(int i, int j);
  void SortFacesByYawAngle();
  Eigen::RowVector3d A_eq_;
  Eigen::VectorXd b_eq_;
  Eigen::MatrixXd A_ = Eigen::MatrixXd::Zero(0,0);
  Eigen::VectorXd b_ = Eigen::VectorXd::Zero(0);
};
}