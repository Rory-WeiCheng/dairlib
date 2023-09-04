#pragma once

struct C3Options {
  // Hyperparameters
  int admm_iter = 2 ;    // total number of ADMM iterations 2 # 4 after issue fixed
  float rho = 0.1;       // inital value of the rho parameter
  float rho_scale = 2.5;  // scaling of rho parameter (/rho = rho_scale * /rho) 3 # 3 after issue fixed
  int num_threads = 0;   // 0 is dynamic, greater than 0 for a fixed count # 1 after issue fixed
  int delta_option = 1;  // different options for delta update
};