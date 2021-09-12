import nevergrad as ng
import drake_cassie_sim
import mujoco_cassie_sim
import os
import pickle
from json import dump, load
import cassie_loss_utils
import numpy as np
import plot_styler
import matplotlib.pyplot as plt
from random import sample, choice
import time
import sys

SIM_ERROR_LOSS = 400

drake_sim = drake_cassie_sim.DrakeCassieSim(drake_sim_dt=8e-5, loss_filename='2021_09_07_weights')
mujoco_sim = mujoco_cassie_sim.LearningMujocoCassieSim(loss_filename='2021_09_07_weights')
# drake_sim = drake_cassie_sim.DrakeCassieSim(drake_sim_dt=8e-5, loss_filename=time.strftime("%Y_%m_%d") + '_weights')
# mujoco_sim = mujoco_cassie_sim.LearningMujocoCassieSim(loss_filename=time.strftime("%Y_%m_%d") + '_weights')
loss_over_time = []
stiffness_over_time = []
params_over_time = []
log_num = 'training'
budget = 2500
# budget = 5000
# budget = 5000
# budget = 25000

# batch_size = 22
all_logs = drake_sim.log_nums_real
batch_size = len(all_logs)
num_train = int(0.9 * len(all_logs))
training_idxs = sample(all_logs, num_train)
test_idxs = [idx for idx in all_logs if not (idx in training_idxs)]

ps = plot_styler.PlotStyler()
ps.set_default_styling(
  directory='/home/yangwill/Documents/research/projects/impact_uncertainty/figures/learning_parameters')


def get_drake_loss(params, log_num=None, plot=False):
  if (log_num == None):
    log_num = choice(training_idxs)
  # print(log_num)
  sim_id = drake_sim.run(params, log_num)
  if sim_id == '-1':
    print('initial state was infeasible')
    loss = SIM_ERROR_LOSS
  else:
    loss = drake_sim.compute_loss(log_num, sim_id, params, plot)
  loss_over_time.append(loss)
  stiffness_over_time.append(params['stiffness'])
  return loss

def get_drake_loss_mp(params):
  loss_sum = 0
  # for i in range(batch_size):
  for i in all_logs:
    loss_sum += get_drake_loss(params, i)
  print(loss_sum / batch_size)

  params_over_time.append(params)
  loss_over_time.append(loss_sum / batch_size)
  return loss_sum / batch_size

def get_mujoco_loss(params, log_num=None):
  if (log_num == None):
    log_num = choice(training_idxs)
  sim_id = mujoco_sim.run(params, log_num)
  loss = mujoco_sim.compute_loss(log_num, sim_id, params)
  # sim_id = mujoco_sim.run(params, log_num)
  # loss = mujoco_sim.compute_loss(log_num, sim_id, params)
  loss_over_time.append(loss)
  return loss

def get_mujoco_loss_mp(params):
  loss_sum = 0
  # for i in range(batch_size):
  for i in all_logs:
    loss_sum += get_mujoco_loss(params, i)
  print(loss_sum / batch_size)

  params_over_time.append(params)
  loss_over_time.append(loss_sum / batch_size)
  return loss_sum / batch_size

def print_loss_weights(loss_filename):
  new_loss_filename = time.strftime("%Y_%m_%d") + '_weights'
  loss_weights = cassie_loss_utils.CassieLoss(loss_filename)
  loss_weights.weights.vel[0:3, 0:3] = 1e2 * np.eye(3)
  loss_weights.weights.vel[3, 3] = 1e-2  # hip roll joints
  loss_weights.weights.vel[4, 4] = 1e-2
  loss_weights.weights.vel[11, 11] = 1e-2  # knee spring joints
  loss_weights.weights.vel[12, 12] = 1e-2
  loss_weights.weights.vel[15, 15] = 0  # ankle spring joint
  loss_weights.weights.vel[17, 17] = 0
  loss_weights.weights.vel[16, 16] = 1e-2  # toe joints
  loss_weights.weights.vel[18, 18] = 1e-2
  loss_weights.weights.omega = 5 * np.eye(3)
  loss_weights.weights.impulse_weight = 1e-7
  loss_weights.weights.save(new_loss_filename)
  loss_weights.print_weights()

def learn_drake_cassie_params(batch=False):
  optimization_param = ng.p.Dict(
    mu=ng.p.Scalar(lower=0.01, upper=1.0),
    # mu_ratio=ng.p.Scalar(lower=0.001, upper=1.0),
    stiffness=ng.p.Log(lower=1e3, upper=1e6),
    dissipation=ng.p.Scalar(lower=0.0, upper=3.0),
    # stiction_tol=ng.p.Log(lower=1e-4, upper=1e-2),
    # vel_offset=ng.p.Array(shape=(3,)).set_bounds(lower=-0.5, upper=0.5),
    # z_offset=ng.p.Array(shape=(1,)).set_bounds(lower=-0.05, upper=0.05)
    # vel_offset=ng.p.Array(shape=(len(all_logs) * 3,)).set_bounds(lower=-1, upper=1),
    # z_offset=ng.p.Array(shape=(len(all_logs),)).set_bounds(lower=-0.05, upper=0.05)
  )

  optimization_param.value = drake_sim.default_drake_contact_params
  # optimization_param.value=drake_sim.load_params('15' + '_optimized_params_5000').value
  optimizer = ng.optimizers.NGOpt(parametrization=optimization_param, budget=budget)
  if batch:
    params = optimizer.minimize(get_drake_loss_mp)
  else:
    params = optimizer.minimize(get_drake_loss)
  loss = np.array(loss_over_time)
  stiffness = np.array(stiffness_over_time)
  np.save(drake_sim.params_folder + log_num + '_loss_trajectory_' + str(budget), loss)
  np.save(drake_sim.params_folder + log_num + '_stiffness_trajectory_' + str(budget), stiffness)
  drake_sim.save_params(params, '_training_' + str(budget))
  # drake_sim.save_params(params, log_num + '_x_offsets_' + str(budget))
  print('optimal params:')
  print(params)


def learn_mujoco_cassie_params():
  # optimization_param = ng.p.Dict(
  #   timeconst=ng.p.Log(lower=1e-4, upper=1e-2),
  #   dampratio=ng.p.Scalar(lower=1e-2, upper=1e1),
  #   ground_mu_tangent=ng.p.Scalar(lower=0.01, upper=1.0),
  #   mu_torsion=ng.p.Scalar(lower=0.00001, upper=1.0),
  #   mu_rolling=ng.p.Log(lower=0.000001, upper=0.01)
  # )
  optimization_param = ng.p.Dict(
    stiffness=ng.p.Scalar(lower=1e3, upper=1e6),
    damping=ng.p.Scalar(lower=0, upper=1000),
    mu_tangent=ng.p.Scalar(lower=0.01, upper=1.0)
    # mu_torsion=ng.p.Scalar(lower=0.001, upper=1.0),
    # mu_rolling=ng.p.Log(lower=0.000001, upper=0.01)
  )

  optimization_param.value = mujoco_sim.default_mujoco_contact_params
  optimizer = ng.optimizers.NGOpt(parametrization=optimization_param, budget=budget)
  params = optimizer.minimize(get_mujoco_loss_mp)
  loss = np.array(loss_over_time)
  stiffness = np.array(stiffness_over_time)
  np.save(drake_sim.params_folder + log_num + '_loss_trajectory_' + str(budget), loss)
  np.save(drake_sim.params_folder + log_num + '_stiffness_trajectory_' + str(budget), stiffness)
  mujoco_sim.save_params(params, '_training_' + str(budget))


def plot_loss_trajectory():
  loss_t = np.load(drake_sim.params_folder + log_num + '_loss_trajectory_' + str(budget) + '.npy')
  # stiffness_t = np.load(drake_sim.params_folder + log_num + '_stiffness_trajectory_' + str(budget) + '.npy')
  stiffness_t = np.load(drake_sim.params_folder + 'training_stiffness_trajectory_5000.npy')
  import pdb; pdb.set_trace()
  # ps.scatter(stiffness_t, loss_t, xlabel='penetration_allowance (m)', ylabel='loss')
  ps.plot(np.arange(0, loss_t.shape[0]), stiffness_t, xlabel='iter', ylabel='stiffness')
  plt.show()


def print_drake_cassie_params(single_log_num, plot=False):
  # optimal_params = drake_sim.load_params('drake_2021_09_01_10_training_5000').value
  # optimal_params = drake_sim.load_params('drake_2021_09_07_16_training_1000').value
  # optimal_params = drake_sim.load_params('drake_2021_08_28_22_26' + single_log_num + '_x_offsets_5000').value

  # optimal_params = drake_sim.load_params('drake_2021_09_07_18_training_5000').value
  optimal_params = drake_sim.load_params('drake_2021_09_10_17_training_2500').value
  loss = get_drake_loss(optimal_params, single_log_num, plot)
  print(loss)
  # stiffness = optimal_params['stiffness']
  # dissipation = optimal_params['dissipation']
  # print('stiffness')
  # print(stiffness)
  # print('dissipation')
  # print(dissipation)
  return loss


def print_mujoco_cassie_params(single_log_num):
  # optimal_params = drake_sim.load_params(log_num + '_optimized_params_' + str(budget))
  # optimal_params = mujoco_sim.load_params('all' + '_optimized_params_' + str(budget))
  # optimal_params = mujoco_sim.load_params('mujoco_2021_09_06_22_training_5000').value
  # optimal_params = mujoco_sim.load_params('mujoco_2021_09_07_17_training_1000').value
  # optimal_params = mujoco_sim.load_params('mujoco_2021_09_07_18_training_5000').value
  optimal_params = mujoco_sim.load_params('mujoco_2021_09_09_15_training_5000').value
  loss = get_mujoco_loss(optimal_params, single_log_num)
  print(loss)

  stiffness = optimal_params['stiffness']
  damping = optimal_params['damping']
  # print('stiffness')
  # print(stiffness)
  # print('damping')
  # print(damping)
  return loss


def plot_per_log_loss_drake():
  log_nums = []
  losses = []
  for log_num_ in all_logs:
    print(log_num_)
    losses.append(print_drake_cassie_params(log_num_))
    log_nums.append((log_num_))
  ps.plot(log_nums, losses, xlabel='log number', ylabel='best loss')
  losses = np.array(losses)
  print('average_loss: ' + str(np.average(losses)))
  plt.show()


def plot_per_log_loss_mujoco():
  log_nums = []
  losses = []
  for log_num_ in all_logs:
    print(log_num_)
    losses.append(print_mujoco_cassie_params(log_num_))
    log_nums.append((log_num_))
  ps.plot(log_nums, losses, xlabel='log number', ylabel='best loss')
  losses = np.array(losses)
  print('average_loss: ' + str(np.average(losses)))
  plt.show()


def save_x_offsets():
  z_offsets = {}
  vel_offsets = {}
  for i in all_logs:
    log_num = i
    print_drake_cassie_params(log_num)
  # with open(drake_sim.params_folder + 'optimized_z_offsets.pkl', 'wb') as file:
  #   pickle.dump(z_offsets, file, protocol=pickle.HIGHEST_PROTOCOL)
  # with open(drake_sim.params_folder + 'optimized_vel_offsets.pkl', 'wb') as file:
  #   pickle.dump(vel_offsets, file, protocol=pickle.HIGHEST_PROTOCOL)

def print_params():
  global log_num
  for i in all_logs:
    print(i)
    log_num = i
    print_drake_cassie_params(log_num)

def learn_x_offsets():
  global training_idxs
  global log_num
  for i in all_logs:
    print(i)
    log_num = i
    training_idxs = [log_num]
    learn_drake_cassie_params()

def print_drake_optimal():
  print('drake_optimal')
  # optimal_params = drake_sim.load_params('drake_2021_09_07_18_training_5000').value
  # optimal_params = drake_sim.load_params('drake_2021_09_08_16_training_5000').value
  # optimal_params = drake_sim.load_params('drake_2021_09_09_15_training_5000').value
  optimal_params = drake_sim.load_params('drake_2021_09_10_17_training_2500').value
  print('stiffness')
  print(optimal_params['stiffness'])
  print('dissipation')
  print(optimal_params['dissipation'])
  print('mu')
  print(optimal_params['mu'])

def print_mujoco_optimal():
  print('mujoco_optimal')
  # optimal_params = mujoco_sim.load_params('mujoco_2021_09_07_18_training_5000').value
  # optimal_params = mujoco_sim.load_params('mujoco_2021_09_08_17_training_5000').value
  # optimal_params = mujoco_sim.load_params('mujoco_2021_09_09_15_training_5000').value
  optimal_params = mujoco_sim.load_params('mujoco_2021_09_11_18_training_500').value
  print('stiffness')
  print(optimal_params['stiffness'])
  print('damping')
  print(optimal_params['damping'])
  print('mu_tangent')
  print(optimal_params['mu_tangent'])

if (__name__ == '__main__'):

  print_drake_optimal()
  print_mujoco_optimal()
  # print_loss_weights('pos_loss_weights')
  # learn_x_offsets()
  # save_x_offsets()
  # print_drake_cassie_params()
  # learn_drake_cassie_params(batch=True)
  # learn_mujoco_cassie_params()
  # plot_per_log_loss_drake()
  # plot_per_log_loss_mujoco()
  # print_mujoco_cassie_params()
  # log_num = '33'
  # print_drake_cassie_params(str(sys.argv[1]), True)
  # plot_loss_trajectory()
  pass