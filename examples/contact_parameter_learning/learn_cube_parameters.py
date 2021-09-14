import nevergrad as ng
import cube_sim
import drake_cube_sim
import mujoco_cube_sim
import bullet_cube_sim
import os
import sys
from json import dump, load
from random import random, sample, choice, seed
import numpy as np
from concurrent import futures
import time

####################################
## COMMON VALUES AND FUNCTIONS

# File paths
cube_data_folder = os.path.join(os.getcwd(), 'examples/contact_parameter_learning/cleaned_cube_trajectories/data')
drake_data_folder = os.path.join(os.getcwd(), 
    'examples/contact_parameter_learning/simulated_cube_trajectories/drake')
mujoco_data_folder = os.path.join(os.getcwd(), 
    'examples/contact_parameter_learning/simulated_cube_trajectories/mujoco')
bullet_data_folder = os.path.join(os.getcwd(), 
    'examples/contact_parameter_learning/simulated_cube_trajectories/bullet')

log_folder = os.path.join(os.getcwd(), 'examples/contact_parameter_learning/logs/cube')
model_folder = os.path.join(os.getcwd(), 'examples/contact_parameter_learning/learned_parameters/cube')

default_loss = cube_sim.LossWeights(pos=(2.0/cube_sim.BLOCK_HALF_WIDTH)*np.ones((3,)), vel=np.zeros((3,)), omega=np.zeros((3,)))
SIM_ERROR_LOSS = 100

# timestepping definitions
HIGH_RES_SUBSTEP = 100
MED_RES_SUBSTEP = 10
LOW_RES_SUBSTEP = 1

default_substep = MED_RES_SUBSTEP

# optimization parameters
batch_size = 550
num_workers = 1
num_trials = 550
num_train = 300

budget = 1000



# Make a list of train and test trials 
trial_idxs = range(num_trials)
training_idxs = trial_idxs #sample(trial_idxs, num_train)
test_idxs = [idx for idx in trial_idxs if not (idx in training_idxs)]

loss_over_time = []
params_over_time = []


def log_optimization(sim_name, test, loss, weights, params_over_time, optimal_params):
    datetime_str = sim_name + '_' + time.strftime("%Y_%m_%d_%H_%M") + '_' + str(default_substep)
    os.mkdir(os.path.join(log_folder, datetime_str))

    base_filename = os.path.join(log_folder, datetime_str)
    loss_log_name = os.path.join(base_filename, 'loss.npy')
    weights_file_name = os.path.join(base_filename, 'weights.json')
    test_idx_filename = os.path.join(base_filename, 'test_set.json') 

    params_filename = os.path.join(model_folder, datetime_str +'.json')
    with open(params_filename, 'w+') as fp:
        dump(optimal_params, fp)

    params_names = optimal_params.keys()
    for name in params_names:
        param_vals = []
        for i in range(len(params_over_time)):
            param_vals.append(params_over_time[i][name])
        filename = os.path.join(base_filename, name + '.npy')
        with open(filename, 'wb+') as fp:
            np.save(fp, np.array(param_vals))
    
    with open(test_idx_filename, 'w+') as fpt:
        dump(test, fpt)

    with open(loss_log_name, 'wb+') as fpl:
        np.save(fpl, loss)

    with open(weights_file_name, 'wb+') as fpw:
        weights.save(fpw)

####################################
## DRAKE FUNCTIONS

def get_drake_loss_mp(params):
    loss_sum = 0
    for i in range(batch_size):
        loss_sum += get_drake_loss(params, trial_num=training_idxs[i])
    print(loss_sum / batch_size)

    params_over_time.append(params)
    loss_over_time.append(loss_sum / batch_size)
    return loss_sum / batch_size

def get_drake_loss(params, trial_num=None):
    if (trial_num == None): trial_num = choice(training_idxs)
    weights = cube_sim.FastLossWeights(pos=(1.0/cube_sim.BLOCK_HALF_WIDTH)*np.ones((3,)))
    try:
        sim = drake_cube_sim.DrakeCubeSim(visualize=False, substeps=default_substep)
        loss = cube_sim.calculate_cubesim_loss(params, trial_num, cube_data_folder, sim, debug=False, weights=weights)
    except:
        loss = SIM_ERROR_LOSS
    return loss

def learn_drake_params():
    
    optimization_param = ng.p.Dict(
        mu = ng.p.Scalar(lower=0.01, upper=1.0), 
        stiffness = ng.p.Scalar(lower=1e2, upper=1e5),
        dissipation = ng.p.Scalar(lower=0, upper=2.0)
    )

    optimization_param.value=drake_cube_sim.default_drake_contact_params
    optimizer = ng.optimizers.NGOpt(parametrization=optimization_param, budget=budget)
    optimal_params = optimizer.minimize(get_drake_loss_mp)

    log_optimization('drake', test_idxs, loss_over_time, default_loss, params_over_time, optimal_params.value)

####################################
## MUJOCO FUNCTIONS

def get_mujoco_loss_mp(params):
    loss_sum = 0
    for i in range(batch_size):
        loss_sum += get_mujoco_loss(params, trial_num=training_idxs[i])
    print(loss_sum / batch_size)

    params_over_time.append(params)
    loss_over_time.append(loss_sum / batch_size)
    return loss_sum / batch_size

def get_mujoco_loss(params, trial_num=None):
    if (trial_num == None): trial_num = choice(training_idxs)
    weights = cube_sim.FastLossWeights(pos=(1.0/cube_sim.BLOCK_HALF_WIDTH)*np.ones((3,)))
    sim = mujoco_cube_sim.MujocoCubeSim(visualize=False, substeps=default_substep)
    return cube_sim.calculate_cubesim_loss(params, trial_num, cube_data_folder, sim, debug=False, weights=weights)

def learn_mujoco_params():
    optimization_param = ng.p.Dict(
        stiffness=ng.p.Scalar(lower=100, upper=10000),
        damping=ng.p.Scalar(lower=0, upper=1000),
        mu_tangent=ng.p.Scalar(lower=0.01, upper=1.0)
        # mu_torsion=ng.p.Scalar(lower=0.001, upper=1.0),
        # mu_rolling=ng.p.Log(lower=0.000001, upper=0.01)
    )
    optimization_param.value=mujoco_cube_sim.default_mujoco_contact_params
    optimizer = ng.optimizers.NGOpt(parametrization=optimization_param, budget=budget)
    optimal_params = optimizer.minimize(get_mujoco_loss_mp)

    log_optimization('mujoco', test_idxs, loss_over_time, default_loss, params_over_time, optimal_params.value)


####################################
## BULLET FUNCTIONS

def get_bullet_loss_mp(params):
    loss_sum = 0
    for i in range(batch_size):
        loss_sum += get_bullet_loss(params, trial_num=training_idxs[i])
    print(loss_sum / batch_size)

    params_over_time.append(params)
    loss_over_time.append(loss_sum / batch_size)
    return loss_sum / batch_size

def get_bullet_loss(params, trial_num=None):
    if (trial_num == None): trial_num = choice(training_idxs)
    weights = cube_sim.FastLossWeights(pos=(1.0/cube_sim.BLOCK_HALF_WIDTH)*np.ones((3,)), bullet=True)
    sim = bullet_cube_sim.BulletCubeSim(visualize=False, substeps=default_substep)
    return cube_sim.calculate_cubesim_loss(params, trial_num, cube_data_folder, sim, debug=False, weights=weights)

def learn_bullet_params():
    optimization_param = ng.p.Dict(
        stiffness=ng.p.Scalar(lower=100, upper=10000),
        damping=ng.p.Scalar(lower=0, upper=1000),
        mu_tangent=ng.p.Scalar(lower=0.01, upper=1.0),
        # restitution=ng.p.Scalar(lower=0.01, upper=0.3),
        # mu_torsion=ng.p.Scalar(lower=0.001, upper=1.0),
        # mu_rolling=ng.p.Log(lower=0.000001, upper=0.01)
    )
    optimization_param.value=bullet_cube_sim.default_bullet_contact_params
    optimizer = ng.optimizers.NGOpt(parametrization=optimization_param, budget=budget)
    optimal_params = optimizer.minimize(get_bullet_loss_mp)
    
    log_optimization('bullet', test_idxs, loss_over_time, default_loss, params_over_time, optimal_params.value)
    

if (__name__ == '__main__'):
    sim_choice = sys.argv[1]

    if (sim_choice == 'drake'):
        learn_drake_params()
    elif (sim_choice == 'mujoco'):
        learn_mujoco_params()
    elif (sim_choice == 'bullet'):
        learn_bullet_params()

    else:
        print('please pick a supported simulator')