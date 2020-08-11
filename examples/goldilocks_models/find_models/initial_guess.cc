//
// Created by jianshu on 3/25/20.
//
#include "examples/goldilocks_models/find_models/initial_guess.h"

using std::cout;
using std::endl;
using std::string;
using std::to_string;

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace dairlib::goldilocks_models {
// edited by Jianshu to try a new way of setting initial guess

VectorXd GetThetaScale(const ReducedOrderModel& rom) {
  // considering the scale for theta doesn't have a significant impact on
  // improving the quality of the initial guess,set them all ones.
  return VectorXd::Ones(rom.n_y() + rom.n_yddot());
}

// calculate the interpolation weight; update weight vector and solution matrix
void InterpolateAmongDifferentTasks(const string& dir, string prefix,
                                    const VectorXd& current_task,
                                    const VectorXd& task_scale,
                                    VectorXd& weight_vector,
                                    MatrixXd& solution_matrix) {
  // check if this sample is success
  int is_success = (readCSV(dir + prefix + string("_is_success.csv")))(0, 0);
  if (is_success == 1) {
    // extract past task
    VectorXd past_task = readCSV(dir + prefix + string("_task.csv"));
    double distance_task =  TaskDistanceCalculation(past_task,
        current_task,task_scale);

    // extract the solution
    VectorXd w_to_interpolate = readCSV(dir + prefix + string("_w.csv"));
    // concatenate the weight and solution for further calculation
    solution_matrix.conservativeResize(w_to_interpolate.rows(),
                                       solution_matrix.cols() + 1);
    solution_matrix.col(solution_matrix.cols() - 1) = w_to_interpolate;
    weight_vector.conservativeResize(weight_vector.rows() + 1);
    weight_vector(weight_vector.rows() - 1) = 1 / distance_task;
  }
}

// calculate interpolated initial guess using weight vector and solution matrix
VectorXd CalculateInterpolation(const VectorXd& weight_vector,
                                const MatrixXd& solution_matrix) {
  DRAKE_DEMAND(weight_vector.rows() > 0);
  // normalize the weight vector by L1 norm and interpolate
  VectorXd interpolated_solution = solution_matrix * weight_vector/weight_vector.sum();
  return interpolated_solution;
}


string SetInitialGuessByInterpolation(const string& directory, int iter,
                                      int sample,
                                      const TasksGenerator* task_gen,
                                      const Task& task,
                                      const ReducedOrderModel& rom,
                                      const ExpansionTasksGenerator& task_gen_expansion) {
  DRAKE_DEMAND(iter > 0);
  /* define some parameters used in interpolation
   * theta_range :decide the range of theta to use in interpolation
   * theta_sclae,task_scale :used to scale the theta and task in interpolation
   */
  double theta_range =
      0.004;  // this is tuned by robot_option=1,rom_option=2,3d task space
  int total_sample_num = task_gen->total_sample_number();

  VectorXd theta_scale = GetThetaScale(rom);
  VectorXd task_scale = task_gen->GetTaskScale();
  //    initialize variables used for setting initial guess
  VectorXd initial_guess;
  string initial_file_name;
  //    get theta of current iteration and task of current sample
  VectorXd current_theta = rom.theta();
  VectorXd current_task =
      Eigen::Map<const VectorXd>(task.get().data(), task.get().size());

  int past_iter;
  int sample_num;
  int num_sample_in_iteration;
  string prefix;
  if(iter==1){
    // In this iteration, the task space is gradually extended
    // Considering that the theta are all same for these iteration,
    // we only consider task in interpolation.

    if(task_gen_expansion.num_extending_task_space()==1){
      // we ca only use the solution in iteration 0
      past_iter = 0;
      num_sample_in_iteration = total_sample_num;
    }
    else{
      past_iter = 1;
      num_sample_in_iteration = total_sample_num*
          (task_gen_expansion.num_extending_task_space()-1);
    }

    // take out corresponding solution and store it in each column of w_task
    // calculate the interpolation weight and store it in weight_task
    MatrixXd w_task;
    VectorXd weight_task;

    /*
     * calculate the weighted sum of past solutions
     */
    for (sample_num = 0; sample_num <num_sample_in_iteration ; sample_num++) {
      prefix = to_string(past_iter) + string("_") + to_string(sample_num);
      InterpolateAmongDifferentTasks(directory, prefix, current_task,
                                     task_scale, weight_task, w_task);
    }
    // calculate the weighted sum of all past iterations
    initial_guess = CalculateInterpolation(weight_task, w_task);
    //    save initial guess and set init file
    initial_file_name = to_string(iter) + "_" +
        to_string(num_sample_in_iteration+sample)+string("_initial_guess.csv");
    writeCSV(directory + initial_file_name, initial_guess);
  }
  else{
    // There are two-stage interpolation here.
    // Get interpolated results using solutions of different tasks for each
    // theta. Then calculate interpolation using results from different theta.

    VectorXd weight_theta;  // each element corresponds to a weight
    MatrixXd w_theta;       // each column stores a interpolated result
    int iter_start = 1;

    for (past_iter = iter - 1; past_iter >= iter_start; past_iter--) {
      // find useful theta according to the difference between previous theta
      // and new theta
      VectorXd past_theta_s =
          readCSV(directory + to_string(past_iter) + string("_theta_y.csv"));
      VectorXd past_theta_sDDot = readCSV(directory + to_string(past_iter) +
          string("_theta_yddot.csv"));
      VectorXd past_theta(past_theta_s.rows() + past_theta_sDDot.rows());
      past_theta << past_theta_s, past_theta_sDDot;
      double theta_diff =
          (past_theta - current_theta).norm() / current_theta.norm();
      if ( (theta_diff < theta_range) ) {
        // take out corresponding solution and store it in each column of
        // w_task calculate the interpolation weight and store it in
        // weight_task
        MatrixXd w_task;
        VectorXd weight_task;
        VectorXd w_to_interpolate;
        if(past_iter==1)
        {
          // Considering that iteration 1 is used for expansion in which there
          // are enough samples,we find the closest sample and use the
          // solution of this sample instead of using the interpolated solution

          string prefix_cloest_task = to_string(past_iter) + string("_") + to_string(0);
          sample_num = 0;
          while(file_exist(directory+to_string(past_iter)+"_"+
          to_string(sample_num)+"_w.csv"))
          {
            prefix = to_string(past_iter) + string("_") +
                to_string(sample_num);
            prefix_cloest_task = CompareTwoTasks(directory,prefix_cloest_task,
                prefix,current_task,task_scale);
            sample_num++;
          }
          w_to_interpolate = readCSV(directory + prefix_cloest_task
                                         + string("_w.csv"));
        }
        else {
          // calculate the weighted sum of solutions from one iteration
          for (sample_num = 0; sample_num < total_sample_num; sample_num++) {
            prefix = to_string(past_iter) + string("_") + to_string(sample_num);
            InterpolateAmongDifferentTasks(directory, prefix, current_task,
                                           task_scale, weight_task, w_task);
          }
          // calculate the weighted sum of solutions for this theta
          w_to_interpolate =
              CalculateInterpolation(weight_task, w_task);
        }
        // calculate the weight for the result above using the difference
        // between past theta and current theta
        VectorXd dif_theta =
            (past_theta - current_theta).array().abs() * theta_scale.array();
        double distance_theta = (dif_theta.transpose() * dif_theta)(0, 0);
        // if theta in this iteration accidentally equals to current theta, no
        // need to interpolate and just use the solution from this iteration.
        if (distance_theta == 0) {
          w_theta.conservativeResize(w_to_interpolate.rows(), 1);
          w_theta << w_to_interpolate;
          weight_theta.conservativeResize(1);
          weight_theta << 1;
          break;
        }
          // else concatenate the weighted sum of this iteration and the weight
          // for it
        else {
          w_theta.conservativeResize(w_to_interpolate.rows(),
                                     w_theta.cols() + 1);
          w_theta.col(w_theta.cols() - 1) = w_to_interpolate;
          weight_theta.conservativeResize(weight_theta.rows() + 1);
          weight_theta(weight_theta.rows() - 1) = 1 / distance_theta;
        }
      }
    }
    initial_guess = CalculateInterpolation(weight_theta, w_theta);
    //    save initial guess and set init file
    initial_file_name = to_string(iter) + "_" + to_string(sample) +
        string("_initial_guess.csv");
    writeCSV(directory + initial_file_name, initial_guess);
  }

  return initial_file_name;
}

string ChooseInitialGuessFromMediateIteration(const string& directory, int iter,
    int sample,const TasksGenerator* task_gen,
    const Task& task,const ReducedOrderModel& rom,
    const MediateTasksGenerator& task_gen_mediate,
    const ExpansionTasksGenerator& task_gen_expansion){
  // this method is used to provide initial guess for the first failed sample
  // other samples should still use previous solution as initial guess

  int sample_num;
  int is_success = false;
  string prefix;
  string initial_file_name;
  int total_sample_number = task_gen->total_sample_number();
  if(sample==task_gen_mediate.sample_index_to_help()){
    //this is exactly the sample to help
    for (sample_num = task_gen_mediate.total_sample_number()-1+
        total_sample_number;sample_num >= total_sample_number;
         sample_num--){
      prefix = to_string(iter) + string("_") +
          to_string(sample_num);
      is_success = (readCSV(directory + prefix +string("_is_success.csv")))(0, 0);
      if(is_success==1)
      {
        break;
      }
    }
    initial_file_name = prefix+"_w.csv";
  }
  else{
    prefix = to_string(iter) + string("_") + to_string(sample);
    if(file_exist(directory+prefix+"_w.csv")){
      //we can use previous solutions
      initial_file_name = prefix+"_w.csv";
    }
    else{
      initial_file_name = SetInitialGuessByInterpolation(
          directory, iter, sample, task_gen, task, rom,task_gen_expansion);
    }
  }
  return initial_file_name;
}

// Use extrapolation to provide initial guesses while extending the task space
string SetInitialGuessByExtrapolation(const string& directory, int iter,
                                      int sample,
                                      const TasksGenerator* task_gen,
                                      const Task& task){
  // The algorithm here is based on the idea that the solution moves towards one
  // direction consistently.
  // S_n+1 = S_n+step_n
  // To predict the solution of iteration n+1, we need a estimation of step_n.
  // We use the results from previous iteration to estimate it.
  // step_n = w1(S1-S0)+w2(S2-S1)+w3(S3-S2)+...+w_n(Sn-Sn-1)
  // Si is the interpolated solution from iteration i
  DRAKE_DEMAND(iter > 0);
  VectorXd initial_guess;
  string initial_file_name;
  int past_iter;
  int sample_num;
  int total_sample_num = task_gen->total_sample_number();
  VectorXd current_task =
      Eigen::Map<const VectorXd>(task.get().data(), task.get().size());
  VectorXd task_scale = task_gen->GetTaskScale();
  // calculate the weighted sum of solutions from iteration 0
  MatrixXd w_task;
  VectorXd weight_task;
  string prefix;

  /*
   * use extrapolation
   */
  for (sample_num = 0; sample_num < total_sample_num; sample_num++) {
    prefix = to_string(0) + string("_") + to_string(sample_num);
    InterpolateAmongDifferentTasks(directory, prefix, current_task,
                                   task_scale, weight_task, w_task);
  }
  VectorXd interpolated_solution_0 =
      CalculateInterpolation(weight_task, w_task);
  if(iter==1){
    //for iteration 1 use the interpolated solution from iteration 0
    initial_file_name = to_string(iter) + "_" + to_string(sample) +
        string("_initial_guess.csv");
    writeCSV(directory + initial_file_name, interpolated_solution_0);
  }
  else{
    // Considering that we doesn't change the theta during the expansion,
    // we only extrapolate using the task.
    MatrixXd w_iteration = MatrixXd::Ones(interpolated_solution_0.rows(),iter);
    w_iteration.col(0) = interpolated_solution_0;
    MatrixXd w_step = MatrixXd::Ones(interpolated_solution_0.rows(),iter-1);
    VectorXd weight_step = VectorXd::Ones(iter-1);
    for (past_iter = iter - 1; past_iter > 0; past_iter--){
      MatrixXd w_task;
      VectorXd weight_task;
      for (sample_num = 0; sample_num < total_sample_num; sample_num++) {
        prefix = to_string(past_iter) + string("_") + to_string(sample_num);
        InterpolateAmongDifferentTasks(directory, prefix, current_task,
                                       task_scale, weight_task, w_task);
      }
      // calculate the weighted sum for this iteration
      VectorXd interpolated_solution_i =
          CalculateInterpolation(weight_task, w_task);
      w_iteration.col(past_iter) = interpolated_solution_i;
    }
    //calculate the estimation of step with past interpolated solutions
    for (int i  = 0; i<w_step.cols(); i++){
      //calculate the difference between two adjacent interpolated solutions
      w_step.col(i) = w_iteration.col(i+1)-w_iteration.col(i);
      weight_step(i) = i+1;
    }
    VectorXd step = CalculateInterpolation(weight_step, w_step);
    VectorXd solution_last_iter = w_iteration.col(w_iteration.cols()-1);
    initial_guess = solution_last_iter+step;
    initial_file_name = to_string(iter) + "_" + to_string(sample) +
        string("_initial_guess.csv");
    writeCSV(directory + initial_file_name, initial_guess);
  }

  return initial_file_name;
}

}  // namespace dairlib::goldilocks_models
