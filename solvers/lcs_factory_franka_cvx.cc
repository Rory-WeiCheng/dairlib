#include "solvers/lcs_factory_franka_cvx.h"

#include "multibody/geom_geom_collider.h"
#include "multibody/kinematic/kinematic_evaluator_set.h"
#include "drake/solvers/moby_lcp_solver.h"

#include "drake/math/autodiff_gradient.h"



namespace dairlib {
namespace solvers {

using std::vector;

using drake::AutoDiffVecXd;
using drake::AutoDiffXd;
using drake::MatrixX;
using drake::SortedPair;
using drake::geometry::GeometryId;
using drake::math::ExtractGradient;
using drake::math::ExtractValue;
using drake::multibody::MultibodyPlant;
using drake::systems::Context;

using Eigen::MatrixXd;
using Eigen::VectorXd;

std::pair<LCS,double> LCSFactoryFrankaConvex::LinearizePlantToLCS(
    const MultibodyPlant<double>& plant, const Context<double>& context,
    const MultibodyPlant<AutoDiffXd>& plant_ad,
    const Context<AutoDiffXd>& context_ad,
    const vector<SortedPair<GeometryId>>& contact_geoms,
    int num_friction_directions, double mu, float dt, LCS Res) {


  /// Use Anitescu's Convex Relaxation on contact model, the complementarity
  /// constraints are imposed by the velocity cone

  int n_pos = plant_ad.num_positions();
  int n_vel = plant_ad.num_velocities();
  int n_total = plant_ad.num_positions() + plant_ad.num_velocities();
  int n_input = plant_ad.num_actuators();

  // ------------------------------------------------------------------------ //
  /// First, calculate vdot from non-contact dynamics using AutoDiff Plant
  /// manipulator equation: M * vdot + C + G = Bu + J.T * F_ext (no J_c.T * F_contact)
  /// in Drake's notation convention: M * vdot + C = tau_g + tau_app

  AutoDiffVecXd C(n_vel);
  plant_ad.CalcBiasTerm(context_ad, &C);

  AutoDiffVecXd Bu = plant_ad.MakeActuationMatrix() *
                     plant_ad.get_actuation_input_port().Eval(context_ad);

  // tau_g = -G, see the above comments on drake notation
  AutoDiffVecXd tau_g = plant_ad.CalcGravityGeneralizedForces(context_ad);

  // f_app is a drake MultibodyForces object, not an actual sptial or generalized force
  // f_app.generalized_forces() = tau_app, tau_app = J.T * F_ext
  drake::multibody::MultibodyForces<AutoDiffXd> f_app(plant_ad);
  plant_ad.CalcForceElementsContribution(context_ad, &f_app);

  MatrixX<AutoDiffXd> M(n_vel, n_vel);
  plant_ad.CalcMassMatrix(context_ad, &M);

  // solve vdot_no_contact
  // If ldlt is slow, there are alternate formulations which avoid it
  AutoDiffVecXd vdot_no_contact =
      M.ldlt().solve(tau_g + f_app.generalized_forces() + Bu - C);

  // ------------------------------------------------------------------------ //
  /// Next, calculate qdot from non-contact dynamics and its derivatives (Jacobians) AB_q
  /// AB_q can be used to derive the mapping Nq
  /// Nq is the mapping from velocity to quaternion derivative qdot = Nq v

  // solve qdot_no_contact, can directly get from current plant_ad context
  AutoDiffVecXd qdot_no_contact(n_pos);
  AutoDiffVecXd state = plant_ad.get_state_output_port().Eval(context_ad);
  AutoDiffVecXd vel = state.tail(n_vel);
  plant_ad.MapVelocityToQDot(context_ad, vel, &qdot_no_contact);
  MatrixXd AB_q = ExtractGradient(qdot_no_contact);
  MatrixXd Nq = AB_q.block(0, n_pos, n_pos, n_vel);

  // ------------------------------------------------------------------------ //
  /// Then, from vdot_no_contact get the derivatives (Jacobians) AB_v
  /// Jacobians are named AB_v_q,AB_v_v,AB_v_u for q,v,u, respectively later
  /// Also calculate the dynamics constant term d_v that would be used later

  // Jacobian of vdot_no_contact
  MatrixXd AB_v = ExtractGradient(vdot_no_contact);

  // Constant term in dynamics, d_v = vdot_no_contact - AB_v * [q, v, u]
  VectorXd d_vv = ExtractValue(vdot_no_contact);
  VectorXd inp_dvv = plant.get_actuation_input_port().Eval(context);
  VectorXd x_dvv(n_pos + n_vel + n_input);
  x_dvv << plant.GetPositions(context), plant.GetVelocities(context), inp_dvv;
  VectorXd x_dvvcomp = AB_v * x_dvv;
  VectorXd d_v = d_vv - x_dvvcomp;

  // ------------------------------------------------------------------------ //
  /// Now, calculate the contact related terms, J_n and J_t means the contact
  /// Jacobians in normal and tangential directions, respectively
  /// Note that in Anitescu's convex formulation, the contact Jacobian is not
  /// decoupled in tangential and normal, but use a combination of the two
  /// i.e. J_c = E.T * J_n + mu * J_t

  VectorXd phi(contact_geoms.size());
  MatrixXd J_n(contact_geoms.size(), n_vel);
  MatrixXd J_t(2 * contact_geoms.size() * num_friction_directions, n_vel);

  // from GeomGeomCollider (collision dectection) get contact information
  for (int i = 0; i < contact_geoms.size(); i++) {
    multibody::GeomGeomCollider collider(
        plant, contact_geoms[i]);  // deleted num_fricton_directions (check with
                                   // Michael about changes in geomgeom)
    auto [phi_i, J_i] = collider.EvalPolytope(context, num_friction_directions);

    phi(i) = phi_i;

    J_n.row(i) = J_i.row(0);
    J_t.block(2 * i * num_friction_directions, 0, 2 * num_friction_directions, n_vel)
        = J_i.block(1, 0, 2 * num_friction_directions, n_vel);
  }

  // Define block diagonal E_t containing ones to combine the contact Jacobians
  MatrixXd E_t = MatrixXd::Zero(
      contact_geoms.size(), 2 * contact_geoms.size() * num_friction_directions);
  for (int i = 0; i < contact_geoms.size(); i++) {
    E_t.block(i, i * (2 * num_friction_directions), 1,
              2 * num_friction_directions) =
        MatrixXd::Ones(1, 2 * num_friction_directions);
  };

  // Contact Jacobian for Anitescu Model
  MatrixXd J_c = E_t.transpose() * J_n + mu * J_t;

  // Also calculate M^(-1)J_c.T that would be used in the future
  auto M_ldlt = ExtractValue(M).ldlt();
  MatrixXd MinvJ_c_T = M_ldlt.solve(J_c.transpose());

  // also note that now the n_contact should be smaller since no slack variable
  // and the complementarity variable lambda now means impulse component along
  // the extreme ray of the friction cone, so each contact is 2 * firction directions
  auto n_contact = 2 * contact_geoms.size() * num_friction_directions;

  // ------------------------------------------------------------------------ //
  /// Now, formulate the LCS matrices
  /// Dynamics equations
  /// q_{k+1} = [ q_k ] + [ dt * Nq * v_{k+1} ] = [ q_k ] + [ dt * Nq * v_k ]+ [ dt * dt * Nq * AB_v * [q_k; v_k; u_k] ] + [dt * Nq * Minv * J_c.T * lam] + [ dt * dt * Nq * d_v]
  /// v_{k+1} = [ v_k ] + [ dt * AB_v * [q_k; v_k; u_k] ] + [Minv * J_c.T * lam] + [ dt * d_v ]

  /// Matrix format
  /// [ q_{k+1}; v_{k+1}] = [ I + dt * dt * Nq * AB_v_q,  dt * Nq +  dt * dt * Nq * AB_v_v ] [q_k;v_k] +   [ dt * dt * Nq * AB_v_u ] [u_k] + [ dt * Nq * Minv * J_c.T ] [lam] + [ dt * dt * Nq * dv ]
  ///                       [ dt * AB_v_q              ,  I + dt * AB_v_v                  ]           +   [ dt * AB_v_u           ]       + [ Minv * J_c.T           ]       + [ dt * d_v          ]

  MatrixXd A(n_total, n_total);
  MatrixXd B(n_total, n_input);
  MatrixXd D(n_total, n_contact);
  VectorXd d(n_total);

  MatrixXd AB_v_q = AB_v.block(0, 0, n_vel, n_pos);
  MatrixXd AB_v_v = AB_v.block(0, n_pos, n_vel, n_vel);
  MatrixXd AB_v_u = AB_v.block(0, n_total, n_vel, n_input);

  A.block(0, 0, n_pos, n_pos) =
      MatrixXd::Identity(n_pos, n_pos) + dt * dt * Nq * AB_v_q;
  A.block(0, n_pos, n_pos, n_vel) = dt * Nq + dt * dt * Nq * AB_v_v;
  A.block(n_pos, 0, n_vel, n_pos) = dt * AB_v_q;
  A.block(n_pos, n_pos, n_vel, n_vel) =
       MatrixXd::Identity(n_vel, n_vel) + dt * AB_v_v;

  B.block(0, 0, n_pos, n_input) = dt * dt * Nq * AB_v_u;
  B.block(n_pos, 0, n_vel, n_input) = dt * AB_v_u;

  D = MatrixXd::Zero(n_total, n_contact);
  D.block(0, 0, n_pos, n_contact) = dt * Nq * MinvJ_c_T;
  D.block(n_pos, 0, n_vel, n_contact) = MinvJ_c_T;

  d.head(n_pos) = dt * dt * Nq * d_v;
  d.tail(n_vel) = dt * d_v;

//    std::cout<< "D" << std::endl;
//    std::cout<< D << std::endl;


  /// Complementarity equations
  /// [ 0 ] <= [ lambda ] (PERP) [ E_t.T * phi / dt + J_c * v_{k+1} ] >= 0
  /// [ 0 ] <= [ lambda ] (PERP) [ E_t.T * phi / dt + J_c * ([ v_k ] + [ dt * AB_v * [q_k; v_k; u_k] ] + [Minv * J_c.T * lam] + [ dt * d_v ]) ]

  /// Matrix format
  ///  [ 0 ] <= [lambda] (PERP) [ dt * J_c * AB_v_q,  J_c + dt * J_c * AB_v_v ] [q_k; v_k] + [ dt *  J_c * AB_v_u ] * [u_k] + [ J_c * Minv * J_c.T] * [lam_k] + [E_t.T * phi / dt + dt * J_c * d_v ] >= 0

  MatrixXd E(n_contact, n_total);
  MatrixXd F(n_contact, n_contact);
  MatrixXd H(n_contact, n_input);
  VectorXd c(n_contact);

  E.block(0, 0, n_contact, n_pos) = dt * J_c * AB_v_q;
  E.block(0, n_pos, n_contact, n_vel) = J_c + dt * J_c * AB_v_v;

  F = J_c * MinvJ_c_T;

  H = dt * J_c * AB_v_u;

  c = E_t.transpose() * phi / dt + dt * J_c * d_v;
//  std::cout<< "Bias Term" << std::endl;
//  std::cout<< dt * J_c * d_v << std::endl;

  // ------------------------------------------------------------------------ //
  /// Finally, consider residual lcs input and return the final lcs
  // add the residual part matrices, use the only learning velocity formulation
  MatrixXd Res_Av = Res.A_[0];
  MatrixXd Res_Bv = Res.B_[0];
  MatrixXd Res_Dv = Res.D_[0];
  VectorXd Res_dv = Res.d_[0];
  MatrixXd Res_E = Res.E_[0];
  MatrixXd Res_F = Res.F_[0];
  MatrixXd Res_H = Res.H_[0];
  VectorXd Res_c = Res.c_[0];

  // Assemble the position and learnt velocity part (for dynamics)
  MatrixXd Res_A(n_total, n_total);
  MatrixXd Res_B(n_total, n_input);
  MatrixXd Res_D(n_total, n_contact);
  VectorXd Res_d(n_total);

  Res_A.block(0, 0, n_pos, n_total) = dt * Nq * Res_Av;
  Res_A.block(n_pos, 0, n_vel, n_total) = Res_Av;
  Res_B.block(0, 0, n_pos, n_input) = dt * Nq * Res_Bv;
  Res_B.block(n_pos, 0, n_vel, n_input) = Res_Bv;
  Res_D.block(0, 0, n_pos, n_contact) = dt * Nq * Res_Dv;
  Res_D.block(n_pos, 0, n_vel, n_contact) = Res_Dv;
  Res_d.head(n_pos) = dt * Nq * Res_dv;
  Res_d.tail(n_vel) = Res_dv;

  // add the residual compensation to the nominal model
  A = A + Res_A;
  B = B + Res_B;
  D = D + Res_D;
  d = d + Res_d;
  E = E + Res_E;
//  F = F + Res_F + 0.01 * MatrixXd::Identity(n_contact, n_contact);
  F = F + Res_F;
  H = H + Res_H;
//  c = c + Res_c;
  c = c + Res_c;
  c.head(4) = c.head(4) +  0.002 * VectorXd::Ones(4);
//  c = c + Res_c +  0.15 * VectorXd::Ones(n_contact);

  // MPC horizon
  int N = 5;

  // Scaling fact
  auto Dn = D.squaredNorm();
  auto An = A.squaredNorm();
  auto AnDn = An / Dn;
//  auto AnDn = 0.00004;

  // return a list of matrices
  std::vector<MatrixXd> A_lcs(N, A);
  std::vector<MatrixXd> B_lcs(N, B);
  std::vector<MatrixXd> D_lcs(N, D * AnDn);
  std::vector<VectorXd> d_lcs(N, d );
  std::vector<MatrixXd> E_lcs(N, E / AnDn);
  std::vector<MatrixXd> F_lcs(N, F);
  std::vector<VectorXd> c_lcs(N, c / AnDn);
  std::vector<MatrixXd> H_lcs(N, H / AnDn);

  LCS system(A_lcs, B_lcs, D_lcs, d_lcs, E_lcs, F_lcs, H_lcs, c_lcs);

  std::pair <LCS, double> ret (system, AnDn);

  return ret;

}

}  // namespace solvers
}  // namespace dairlib
