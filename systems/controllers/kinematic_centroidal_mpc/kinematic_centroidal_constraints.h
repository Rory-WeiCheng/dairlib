#pragma  once

#include <drake/multibody/plant/multibody_plant.h>
#include "drake/common/drake_copyable.h"
#include "drake/common/symbolic.h"
#include "drake/solvers/constraint.h"
#include "drake/systems/trajectory_optimization/multiple_shooting.h"
#include "solvers/nonlinear_constraint.h"
#include "multibody/kinematic/kinematic_evaluator_set.h"

/*!
 * @brief Nonlinear constraint for enforcing the centroidal dynamics using euler integration
 */
template <typename T>
class CentroidalDynamicsConstraint : public dairlib::solvers::NonlinearConstraint<T> {

 public:
  /*!
   * @param plant used for calculating inertia tensor and mass
   * @param context
   * @param n_contact number of contact points
   * @param dt timestep for integration
   * @param knot_index
   */
  CentroidalDynamicsConstraint(const drake::multibody::MultibodyPlant<T>& plant,
                              drake::systems::Context<T>* context,
                              int n_contact, double dt, int knot_index);

 public:
  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>>& x,
                          drake::VectorX<T>* y) const override;

 private:
  /*!
   * @brief Calculates the time derivative of the centroidal state
   * @param context context used to calcualte inertia tensor and mass
   * @param xCent centroidal state = [quat, r, omega, dr]
   * @param contact_locations vector of the contact locations in world frame = [contact1x, contact1y, contact1z, ...]
   * @param contact_forces vector of the contact forces in world frame = [force1x, force1y, force1z, ...]
   * @return time derivate of centroidal state
   */
  drake::VectorX<T> CalcTimeDerivativesWithForce(
      drake::systems::Context<T>* context,
      const drake::VectorX<T>& xCent,
      const drake::VectorX<T>& contact_locations,
      const drake::VectorX<T>& contact_forces) const;

  const drake::multibody::MultibodyPlant<T>& plant_;
  drake::systems::Context<T>* context_;
  int n_x_;
  int n_u_;
  int n_contact_;
  int n_cent_ = 13;
  double dt_;
  const drake::VectorX<T> zero_control_;
};

/*!
 * @brief Nonlinear constraint on center of mass position matching centroidal state
 */
template <typename T>
class CenterofMassPositionConstraint : public dairlib::solvers::NonlinearConstraint<T> {

 public:
  CenterofMassPositionConstraint(const drake::multibody::MultibodyPlant<T>& plant,
                               drake::systems::Context<T>* context,
                               int knot_index);

 public:
  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>>& x,
                          drake::VectorX<T>* y) const override;

  const drake::multibody::MultibodyPlant<T>& plant_;
  drake::systems::Context<T>* context_;
  int n_x_;
  int n_u_;
  const drake::VectorX<T> zero_control_;
};

/*!
 * @brief Nonlinear constraint on center of mass velocity matching centroidal velocity
 */
template <typename T>
class CenterofMassVelocityConstraint : public dairlib::solvers::NonlinearConstraint<T> {

 public:
  CenterofMassVelocityConstraint(const drake::multibody::MultibodyPlant<T>& plant,
                                 drake::systems::Context<T>* context,
                                 int knot_index);

 public:
  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>>& x,
                          drake::VectorX<T>* y) const override;

  const drake::multibody::MultibodyPlant<T>& plant_;
  drake::systems::Context<T>* context_;
  int n_x_;
  int n_u_;
  const drake::VectorX<T> zero_control_;
};

/*!
 * @brief Nonlinear constraint on world frame angular velocity matching body frame angular velocity
 */
template <typename T>
class AngularVelocityConstraint : public dairlib::solvers::NonlinearConstraint<T> {

 public:
  AngularVelocityConstraint(int knot_index);

 public:
  void EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>>& x,
                          drake::VectorX<T>* y) const override;
};