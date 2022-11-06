#pragma once

#include "dairlib/lcmt_timestamped_saved_traj.hpp"
#include "dairlib/lcmt_trajectory_block.hpp"
#include "examples/goldilocks_models/goldilocks_utils.h"
#include "examples/goldilocks_models/planning/hybrid_rom_traj_opt.h"
#include "examples/goldilocks_models/reduced_order_models.h"
#include "examples/goldilocks_models/rom_walking_gains.h"
#include "lcm/hybrid_rom_planner_saved_trajectory.h"
#include "multibody/kinematic/kinematic_evaluator_set.h"
#include "multibody/kinematic/world_point_evaluator.h"
#include "multibody/multibody_utils.h"
#include "systems/framework/output_vector.h"

#include "drake/common/trajectories/exponential_plus_piecewise_polynomial.h"
#include "drake/common/trajectories/piecewise_polynomial.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/solvers/osqp_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solver_interface.h"
#include "drake/solvers/solver_options.h"
#include "drake/systems/framework/leaf_system.h"

namespace dairlib {
namespace goldilocks_models {

using multibody::KinematicEvaluatorSet;
using multibody::WorldPointEvaluator;
using std::cout;
using std::endl;

/// This is modified from `CassiePlannerWithMixedRomFom` and designed to
/// interface `HybridRomTrajOpt`

class CassiePlannerWithOnlyRom : public drake::systems::LeafSystem<double> {
 public:
  static const int ROBOT = 1;  // robot index for Cassie

  CassiePlannerWithOnlyRom(
      const drake::multibody::MultibodyPlant<double>& plant_control,
      double stride_period, const PlannerSetting& param,
      const std::set<int>& relax_index,
      const std::set<int>& set_init_state_from_prev_solution,
      const std::set<int>& idx_const_rom_vel_during_double_support,
      bool singel_eval_mode, bool log_data, int print_level = 1);

  const drake::systems::InputPort<double>& get_input_port_stance_foot() const {
    return this->get_input_port(stance_foot_port_);
  }
  const drake::systems::InputPort<double>& get_input_port_init_phase() const {
    return this->get_input_port(phase_port_);
  }
  const drake::systems::InputPort<double>& get_input_port_state() const {
    return this->get_input_port(state_port_);
  }
  const drake::systems::InputPort<double>& get_input_port_fsm_and_lo_time()
      const {
    return this->get_input_port(controller_signal_port_);
  }
  const drake::systems::InputPort<double>& get_input_port_quat_xyz_shift()
      const {
    return this->get_input_port(quat_xyz_shift_port_);
  }
  const drake::systems::InputPort<double>& get_input_port_planner_final_pos()
      const {
    return this->get_input_port(planner_final_pos_port_);
  }

  // Testing
  void completely_use_trajs_from_model_opt_as_target() {
    completely_use_trajs_from_model_opt_as_target_ = true;
  }

  // Made this function public for testing
  int DetermineNumberOfKnotPoints(
      double init_phase, double first_mode_duration,
      double remaining_single_support_duration) const;
  const std::vector<double>& min_dt() const { return min_dt_; }
  const std::vector<int>& num_time_samples() const { return num_time_samples_; }
  const std::vector<int>& num_time_samples_ds() const {
    return num_time_samples_ds_;
  }
  int knots_per_double_support() const { return knots_per_double_support_; }
  int knots_per_single_support() const { return knots_per_single_support_; }

 private:
  void SolveTrajOpt(const drake::systems::Context<double>& context,
                    dairlib::lcmt_timestamped_saved_traj* traj_msg) const;

  void CreateDesiredPelvisPosAndVel(
      int n_total_step, bool start_with_left_stance, double init_phase,
      const Eigen::VectorXd& final_position,
      std::vector<Eigen::VectorXd>* des_xy_pos,
      std::vector<Eigen::VectorXd>* des_xy_vel) const;
  void CreateDesiredComPosAndVel(
      int n_total_step, bool start_with_left_stance, double init_phase,
      const Eigen::VectorXd& final_position,
      std::vector<Eigen::VectorXd>* des_xy_pos,
      std::vector<Eigen::VectorXd>* des_xy_vel) const;
  void CreateDesiredBodyPosAndVel(
      bool pelvis_or_com, int n_total_step, bool start_with_left_stance,
      double init_phase, const Eigen::VectorXd& final_position,
      std::vector<Eigen::VectorXd>* des_xy_pos,
      std::vector<Eigen::VectorXd>* des_xy_vel) const;

  void RotateBetweenGlobalAndLocalFrame(bool rotate_from_global_to_local,
                                        const Eigen::VectorXd& quat_xyz_shift,
                                        const Eigen::MatrixXd& original_x_FOM,
                                        Eigen::MatrixXd* rotated_x_FOM) const;
  void RotatePosBetweenGlobalAndLocalFrame(
      bool rotate_from_global_to_local, bool position_only,
      const Eigen::VectorXd& quat_xyz_shift, const Eigen::MatrixXd& original_x,
      Eigen::MatrixXd* rotated_x) const;

  void BookKeeping(
      bool start_with_left_stance,
      const std::chrono::duration<double>& elapsed_lipm_mpc_and_ik,
      const std::chrono::duration<double>& elapsed_solve,
      const drake::solvers::MathematicalProgramResult& result) const;
  void WarmStartGuess(
      const Eigen::VectorXd& quat_xyz_shift,
      const Eigen::VectorXd& current_local_stance_foot_pos,
      const std::vector<Eigen::Vector2d>& reg_footstep,
      const int global_fsm_idx, int first_mode_knot_idx, double current_time,
      dairlib::goldilocks_models::HybridRomTrajOptCassie* trajopt) const;

  // Flags
  bool use_standing_pose_as_init_FOM_guess_ = false;
  // Although warm start helps most of the time, it could also make the solver
  // not able to find the optimal solution from time to time
  bool warm_start_with_previous_solution_ = true;
  bool use_lipm_mpc_and_ik_;

  // For sim eval (to remove trajopt-sim gap as much as possible)
  bool completely_use_trajs_from_model_opt_as_target_ = false;

  // Since sometimes the planner replan every 1ms in the beginning of the
  // simulation (e.g. at 0, 1, 2 ms), we use
  // min_time_difference_for_replanning_ to avoid replanning when the current
  // time is too close to the previous time. This method is just a workaround.
  // The best solution should be avoiding using old message in the lcm
  // subscriber. However, somehow there is still old message in the lcm
  // subscriber after I clear the listening channel. (it's gone and then come
  // back) Note that another workaround is to clear the lcm message twice in
  // the LcmDrivenLoop class.
  // double min_time_difference_for_replanning_ = 0.01;
  mutable double timestamp_of_previous_plan_ = -1;
  mutable dairlib::lcmt_timestamped_saved_traj previous_output_msg_;

  // Port indices
  int stance_foot_port_;
  int phase_port_;
  int state_port_;
  int controller_signal_port_;
  int quat_xyz_shift_port_;
  int planner_final_pos_port_;

  std::map<std::string, int> pos_map_;
  std::map<std::string, int> vel_map_;

  int nq_;
  int nv_;
  int nx_;
  int n_y_;
  int n_tau_;

  const drake::multibody::MultibodyPlant<double>& plant_control_;
  std::unique_ptr<drake::systems::Context<double>> context_plant_control_;
  double stride_period_;
  double single_support_duration_;
  double double_support_duration_;

  std::unique_ptr<ReducedOrderModel> rom_;
  StateMirror state_mirror_;
  std::vector<BodyPoint> left_contacts_;
  std::vector<BodyPoint> right_contacts_;
  BodyPoint left_origin_;
  BodyPoint right_origin_;
  BodyPoint left_mid_;
  BodyPoint right_mid_;
  std::vector<std::tuple<std::string, double, double>> joint_name_lb_ub_;
  mutable drake::solvers::SolverOptions solver_option_ipopt_;
  mutable drake::solvers::SolverOptions solver_option_snopt_;
  std::unique_ptr<drake::solvers::SolverInterface> solver_ipopt_;
  std::unique_ptr<drake::solvers::SolverInterface> solver_snopt_;

  // Parameters for traj opt
  //  const PlannerSetting& param_;
  mutable PlannerSetting param_;

  // Cost weight
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd R_;

  // Initial guesses
  Eigen::VectorXd h_guess_;
  Eigen::MatrixXd y_guess_;
  Eigen::MatrixXd dy_guess_;
  Eigen::MatrixXd tau_guess_;
  // Regularization terms
  Eigen::VectorXd h_reg_;
  Eigen::MatrixXd y_reg_;
  Eigen::MatrixXd dy_reg_;
  Eigen::MatrixXd tau_reg_;
  // Both initial guess and regularization
  Eigen::VectorXd x_guess_steppingleft_pre_;
  Eigen::VectorXd x_guess_steppingright_pre_;
  Eigen::VectorXd x_guess_steppingleft_post_;
  Eigen::VectorXd x_guess_steppingright_post_;
  // Regularization term for the footstep variables
  Eigen::Vector2d left_step_;
  Eigen::Vector2d right_step_;
  // Desired height
  double desired_com_height_;

  // Index of state which we initialize to 1 in order to avoid singularity
  // (which messes with gradient)
  int idx_state_init_to_1_;

  // For warm start with previous solution
  mutable int prev_global_fsm_idx_ = -1;
  mutable int prev_first_mode_knot_idx_ = -1;
  mutable std::vector<int> prev_mode_start_;
  // Previous solutions
  mutable Eigen::VectorXd z_;
  mutable HybridRomPlannerTrajectory lightweight_saved_traj_;
  mutable Eigen::VectorXd h_solutions_;
  mutable Eigen::MatrixXd input_at_knots_;
  // TODO: apart from local, we also store global for better warm-start.
  mutable Eigen::MatrixXd global_footstep_;
  mutable Eigen::MatrixXd global_y_end_of_last_mode_;
  mutable Eigen::VectorXd touchdown_foot_pos_;
  mutable Eigen::VectorXd eps_rom_;
  mutable Eigen::VectorXd local_predicted_com_vel_;
  mutable Eigen::VectorXd global_predicted_com_vel_;
  mutable Eigen::MatrixXd local_x_lipm_;
  mutable Eigen::MatrixXd local_u_lipm_;
  mutable Eigen::MatrixXd global_x_lipm_;
  mutable Eigen::MatrixXd global_u_lipm_;
  mutable Eigen::MatrixXd global_preprocess_x_lipm_;
  mutable Eigen::MatrixXd global_preprocess_u_lipm_;

  // Init state relaxation (relax the mapping function)
  std::set<int> relax_index_;

  // Initialize the MPC initial state with previous ROM traj solution
  std::set<int> set_init_state_from_prev_solution_;

  // Constant vel index for double support phase
  // Note that this is the index in state
  std::set<int> idx_const_rom_vel_during_double_support_;

  // Time limit
  bool fixed_time_limit_;
  double time_limit_for_first_loop_ = 60;
  double buffer_ = 0.02;  // time for lcm packing/traveling, data saving, etc
  // We don't want the solver to use all the remaining time. Otherwise, there
  // won't be enough time in the next loop.
  // I think min_solve_time_preserved_for_next_loop_ could be set to half of
  // the potentailly-shortest time horizon of the planner (reserve time
  // equally for the current and the next loop)
  double min_solve_time_preserved_for_next_loop_;

  // Default timestep and number of knot points
  mutable std::vector<int> num_time_samples_;
  mutable std::vector<int> num_time_samples_ds_;
  mutable std::vector<double> min_dt_;
  std::vector<double> max_dt_;
  int knots_per_single_support_;  // single support's number of kont points
  int knots_per_double_support_;  // double support's number of kont points
  double phase_from_ss_to_ds_;

  // Swing foot distance
  mutable std::vector<double> max_swing_distance_;

  // LIPM MPC and IK
  bool RunLipmMPC(bool start_with_left_stance, double init_phase,
                  double first_mode_duration,
                  const Eigen::VectorXd& final_position,
                  const Eigen::VectorXd& x_init,
                  Eigen::MatrixXd* local_preprocess_x_lipm,
                  Eigen::MatrixXd* local_preprocess_u_lipm) const;
  drake::solvers::OsqpSolver qp_solver_;
  bool GetDesiredFullStateFromLipmMPCSol(
      const Eigen::VectorXd& x_init, bool start_with_left_stance,
      const Eigen::MatrixXd& local_preprocess_x_lipm,
      const Eigen::MatrixXd& local_preprocess_u_lipm,
      Eigen::MatrixXd* desired_state) const;
  drake::solvers::SnoptSolver snopt_solver_;
  std::unique_ptr<WorldPointEvaluator<double>> left_foot_evaluator_;
  std::unique_ptr<WorldPointEvaluator<double>> right_foot_evaluator_;
  std::unique_ptr<KinematicEvaluatorSet<double>> left_foot_evaluators_;
  std::unique_ptr<KinematicEvaluatorSet<double>> right_foot_evaluators_;
  std::unique_ptr<WorldPointEvaluator<double>> left_toe_z_evaluator_;
  std::unique_ptr<WorldPointEvaluator<double>> left_heel_z_evaluator_;
  std::unique_ptr<WorldPointEvaluator<double>> right_toe_z_evaluator_;
  std::unique_ptr<WorldPointEvaluator<double>> right_heel_z_evaluator_;
  std::unique_ptr<KinematicEvaluatorSet<double>> contact_z_evaluators_;
  std::unique_ptr<WorldPointEvaluator<double>> left_toe_evaluator_;
  std::unique_ptr<WorldPointEvaluator<double>> left_heel_evaluator_;
  std::unique_ptr<WorldPointEvaluator<double>> right_toe_evaluator_;
  std::unique_ptr<WorldPointEvaluator<double>> right_heel_evaluator_;
  std::unique_ptr<KinematicEvaluatorSet<double>> contact_evaluators_;

  Eigen::VectorXd x_standing_fixed_spring_;

  // For debugging and data logging
  mutable int counter_ = 0;

  mutable double total_mpc_and_ik_solve_time_ = 0;
  mutable double max_mpc_and_ik_solve_time_ = 0;

  mutable double total_solve_time_ = 0;
  mutable double max_solve_time_ = 0;

  mutable bool past_is_left_stance_ = true;
  mutable int total_number_of_first_solve_of_the_mode_ = 0;
  mutable double total_solve_time_of_first_solve_of_the_mode_ = 0;
  mutable double max_solve_time_of_first_solve_of_the_mode_ = 0;

  mutable int num_failed_solve_ = 0;
  mutable int latest_failed_solve_idx_ = -1;

  bool single_eval_mode_;
  bool log_data_and_check_solution_;
  int print_level_ = 1;
  void PrintEssentialStatus(const std::string& msg) const {
    if (print_level_ > 0) std::cout << msg << std::endl;
  };
  void PrintStatus(const std::string& msg) const {
    if (print_level_ > 1) std::cout << msg << std::endl;
  };
  void SaveDataIntoFiles(
      double current_time, int global_fsm_idx, const Eigen::VectorXd& x_init,
      double init_phase, bool is_right_stance,
      const Eigen::VectorXd& quat_xyz_shift,
      const Eigen::VectorXd& current_local_stance_foot_pos,
      const Eigen::VectorXd& final_position,
      const Eigen::MatrixXd& global_regularization_x_FOM,
      const Eigen::MatrixXd& local_x0_FOM, const Eigen::MatrixXd& local_xf_FOM,
      const HybridRomTrajOptCassie& trajopt,
      const drake::solvers::MathematicalProgramResult& result,
      const std::string& dir_data, const std::string& prefix) const;
  void PrintCost(const HybridRomTrajOptCassie& trajopt,
                 const drake::solvers::MathematicalProgramResult& result) const;
  void PrintAllCostsAndConstraints(const HybridRomTrajOptCassie& trajopt) const;
  void ResolveWithAnotherSolver(
      const HybridRomTrajOptCassie& trajopt,
      const drake::solvers::MathematicalProgramResult& result,
      const std::string& prefix, double current_time,
      const Eigen::VectorXd& quat_xyz_shift,
      const Eigen::VectorXd& current_local_stance_foot_pos) const;
  void PrintTrajMsg(dairlib::lcmt_timestamped_saved_traj* traj_msg) const;
};

}  // namespace goldilocks_models
}  // namespace dairlib