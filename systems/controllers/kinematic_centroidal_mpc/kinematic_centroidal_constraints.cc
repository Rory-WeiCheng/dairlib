#include <drake/math/quaternion.h>
#include <iostream>
#include "kinematic_centroidal_constraints.h"
#include "multibody/multibody_utils.h"

template<typename T>
CentroidalDynamicsConstraint<T>::CentroidalDynamicsConstraint(const drake::multibody::MultibodyPlant<T> &plant,
                                                              drake::systems::Context<T> *context,
                                                              int n_contact,
                                                              double dt,
                                                              int knot_index): dairlib::solvers::NonlinearConstraint<T>(
    13,  2 * 13 + plant.num_positions() + plant.num_velocities() + 2 * 3 * n_contact,
    Eigen::VectorXd::Zero(13),
    Eigen::VectorXd::Zero(13),
    "centroidal_collocation[" +
        std::to_string(knot_index) + "]"),
                                                                               plant_(plant),
                                                                               context_(context),
                                                                               n_x_(plant.num_positions()
                                                                                        + plant.num_velocities()),
                                                                               n_u_(plant.num_actuators()),
                                                                               n_contact_(n_contact),
                                                                               dt_(dt),
                                                                               zero_control_(Eigen::VectorXd::Zero(n_u_)) {}


/// The format of the input to the eval() function is in the order
///   - xCent0, centroidal state at time k
///   - xCent1, centroidal state at time k+1
///   - x0, state at time k
///   - cj0, contact locations time k
///   - Fj0, contact forces at time k
template <typename T>
void CentroidalDynamicsConstraint<T>::EvaluateConstraint(
    const Eigen::Ref<const drake::VectorX<T>>& x, drake::VectorX<T>* y) const {
  // Extract decision variables
  const auto& xCent0 = x.segment(0, n_cent_);
  const auto& xCent1 = x.segment(n_cent_, n_cent_);
  const auto& x0 = x.segment(2 * n_cent_, n_x_);
  const auto& cj0 = x.segment(2 * n_cent_ + n_x_, 3 * n_contact_);
  const auto& Fj0 = x.segment(2 * n_cent_ + n_x_ + 3 * n_contact_, 3 * n_contact_);

  // Evaluate dynamics at k
  dairlib::multibody::SetContext<T>(plant_, x0, zero_control_, context_);
  const auto& xdot0Cent = CalcTimeDerivativesWithForce(context_, xCent0,cj0, Fj0);

  // Predict state and return error
  const auto x1Predict = xCent0 + xdot0Cent * dt_;
  *y = xCent1 - x1Predict;
}

template<typename T>
drake::VectorX<T> CentroidalDynamicsConstraint<T>::CalcTimeDerivativesWithForce(drake::systems::Context<T> *context,
                                                                                const drake::VectorX<T>& xCent,
                                                                                const drake::VectorX<T>& contact_locations,
                                                                                const drake::VectorX<T>& contact_forces) const {
  // We use this constructor since passing in a vector has the order [x,y,z,w], while xCent is [w,x,y,z]
  const Eigen::Quaternion<T> w_Q_b(xCent[0], xCent[1], xCent[2], xCent[3]);
  const auto b_Q_w = w_Q_b.inverse();
  const auto& r = xCent.segment(4, 3);
  const drake::Vector3<T>& omega_ewrt_b = xCent.segment(7, 3);
  const auto& d_r = xCent.segment(10, 3);

  const auto& body_frame = plant_.get_body(*(plant_.GetFloatingBaseBodies().begin())).body_frame();
  const drake::multibody::SpatialInertia< T >& spatial_inertia =
      plant_.CalcSpatialInertia(*context, body_frame, plant_.GetBodyIndices(drake::multibody::ModelInstanceIndex(2)));
  const auto& rotational_inertia = spatial_inertia.CalcRotationalInertia().CopyToFullMatrix3();
  const auto& mass = spatial_inertia.get_mass();

  drake::Vector3<T> sum_moments;
  drake::Vector3<T> sum_forces;
  for(int contact = 0; contact < n_contact_; contact ++){
    const drake::Vector3<T>& location = contact_locations.segment(contact * 3, 3);
    const drake::Vector3<T>& force = contact_forces.segment(contact * 3, 3);

    sum_moments = sum_forces + (location - r).cross(force);
    sum_forces = sum_forces + force;
  }

  // Working in body frame angular velocity
  const auto d_quat = drake::math::CalculateQuaternionDtFromAngularVelocityExpressedInB(w_Q_b, omega_ewrt_b);
  const auto d_omega_ewrt_b = rotational_inertia.transpose()* (b_Q_w * sum_moments - omega_ewrt_b.cross(rotational_inertia * omega_ewrt_b));
  const auto dd_r = sum_forces/mass - drake::Vector3<T>(0, 0, 9.81);

  drake::Vector<T, 13> rv;
  rv.head(4) = d_quat;
  rv.segment(4,3) = d_r;
  rv.segment(7,3) = d_omega_ewrt_b;
  rv.segment(10,3) = dd_r;
  return rv;
}

template<typename T>
CenterofMassPositionConstraint<T>::CenterofMassPositionConstraint(const drake::multibody::MultibodyPlant<T> &plant,
                                                                  drake::systems::Context<T> *context,
                                                                  int knot_index): dairlib::solvers::NonlinearConstraint<T>(
    3,  plant.num_positions()+ plant.num_velocities()+3,
    Eigen::VectorXd::Zero(3),
    Eigen::VectorXd::Zero(3),
    "center_of_mass_position[" +
        std::to_string(knot_index) + "]"),
                                                                                   plant_(plant),
                                                                                   context_(context),
                                                                                   n_x_(plant.num_positions()
                                                                                            + plant.num_velocities()),
                                                                                   n_u_(plant.num_actuators()),
                                                                                   zero_control_(Eigen::VectorXd::Zero(n_u_)) {}

/// The format of the input to the eval() function is in the order
///   - rCOM, location of the center of mass
///   - x0, state
template<typename T>
void CenterofMassPositionConstraint<T>::EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>> &x,
                                                           drake::VectorX<T> *y) const {
  const auto& rCom = x.segment(0, 3);
  const auto& x0 = x.segment(3, n_x_);

  dairlib::multibody::SetContext<T>(plant_, x0, zero_control_, context_);
  *y = rCom - plant_.CalcCenterOfMassPositionInWorld(*context_);
}

template<typename T>
CenterofMassVelocityConstraint<T>::CenterofMassVelocityConstraint(const drake::multibody::MultibodyPlant<T> &plant,
                                                                  drake::systems::Context<T> *context,
                                                                  int knot_index): dairlib::solvers::NonlinearConstraint<T>(
    3,  plant.num_positions()+ plant.num_velocities()+3,
    Eigen::VectorXd::Zero(3),
    Eigen::VectorXd::Zero(3),
    "center_of_mass_velocity[" +
        std::to_string(knot_index) + "]"),
                                                                                   plant_(plant),
                                                                                   context_(context),
                                                                                   n_x_(plant.num_positions()
                                                                                            + plant.num_velocities()),
                                                                                   n_u_(plant.num_actuators()),
                                                                                   zero_control_(Eigen::VectorXd::Zero(n_u_)) {}

/// The format of the input to the eval() function is in the order
///   - drCOM, location of the center of mass
///   - x0, state
template<typename T>
void CenterofMassVelocityConstraint<T>::EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>> &x,
                                                           drake::VectorX<T> *y) const{
  const auto& drCom = x.segment(0, 3);
  const auto& x0 = x.segment(3, n_x_);

  dairlib::multibody::SetContext<T>(plant_, x0, zero_control_, context_);
  *y = drCom - plant_.CalcCenterOfMassTranslationalVelocityInWorld(*context_);
}

template<typename T>
AngularVelocityConstraint<T>::AngularVelocityConstraint(int knot_index): dairlib::solvers::NonlinearConstraint<T>(
    3, 4 + 3 + 3,
    Eigen::VectorXd::Zero(3),
    Eigen::VectorXd::Zero(3),
    "angular_velocity_constraint[" +
        std::to_string(knot_index) + "]") {}

/// The format of the input to the eval() function is in the order
///   - w_Q_b, quaternion describing transform from body to world
///   - omega_ewrt_b, body frame angular velocity
///   - omega_ewrt_w, world frame angular velocity
template<typename T>
void AngularVelocityConstraint<T>::EvaluateConstraint(const Eigen::Ref<const drake::VectorX<T>> &x,
                                                      drake::VectorX<T> *y) const {
  // We use this constructor since passing in a vector has the order [x,y,z,w], while x is [w,x,y,z]
  const Eigen::Quaternion<T> w_Q_b(x[0], x[1], x[2], x[3]);
  const auto& omega_ewrt_b = x.segment(4, 3);
  const auto& omega_ewrt_w = x.segment(7, 3);
  *y = omega_ewrt_w - w_Q_b * omega_ewrt_b;
}

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS (class CentroidalDynamicsConstraint);
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS (class CenterofMassPositionConstraint);
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS (class CenterofMassVelocityConstraint);
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS (class AngularVelocityConstraint);