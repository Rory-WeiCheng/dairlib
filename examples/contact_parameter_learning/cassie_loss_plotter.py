import numpy as np
import lcm
from scipy.integrate import trapz
import pickle
from pydairlib.common import FindResourceOrThrow
from bindings.pydairlib.common.plot_styler import PlotStyler
from pydrake.trajectories import PiecewisePolynomial
from pydairlib.lcm import lcm_trajectory
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydairlib.cassie.cassie_utils import *
from pydairlib.cassie.kinematics_helper import KinematicsHelper
import pydairlib.multibody
from process_lcm_log import process_log
from cassie_impact_data import CassieImpactData
import scipy.linalg as linalg
from scipy import interpolate
import matplotlib.pyplot as plt


def plot_error_bands(impact_data):

  data_range = np.arange(28, 34, 1)
  # data_range = np.concatenate((np.arange(12, 17, 1), data_range))
  # v_all = np.empty()
  v_sim = []
  v_hardware = []
  # vproj_all = []
  n_samples = 10000
  # joint_idx = 6

  # t_master = np.load(data_directory + 't_28.npy')[:n_samples]
  t_master = impact_data.t_x_hardware['15'][:, 0]
  for i in data_range:
    # t = np.load(data_directory + 't_' + '%.2d.npy' % i)
    # v_hardware = np.load(data_directory + 'v_' + '%.2d.npy' % i)
    # vproj = np.load(data_directory + 'vproj_' + '%.2d.npy' % i)
    # import pdb; pdb.set_trace()
    v_sim_interp = interpolate.interp1d(impact_data.t_x_sim['%.2d' % i], impact_data.x_trajs_sim['%.2d' % i], axis=0, bounds_error=False)
    v_hardware_interp = interpolate.interp1d(impact_data.t_x_hardware['%.2d' % i][:,0], impact_data.x_trajs_hardware['%.2d' % i], axis=0, bounds_error=False)
    # vproj_interp = interpolate.interp1d(t, vproj, axis=0, bounds_error=False)
    # v_all.append(v[:n_samples, :])
    # vproj_all.append(vproj[:n_samples, :])


    # v_sim.append(v_sim_interp(t_master))
    # v_hardware.append(v_hardware_interp(t_master))
    v_sim.append(v_sim_interp(impact_data.t_x_sim['%.2d' % i]))
    v_hardware.append(v_hardware_interp(impact_data.t_x_sim['%.2d' % i]))


    # vproj_all.append(vproj_interp(t_master))
    # ps.plot(t[:n_samples], v[:n_samples, joint_idx], color='b')
    # ps.plot(t[:n_samples], vproj[:n_samples, joint_idx], color='r')
    # import pdb; pdb.set_trace()
  # plt.xlim([-10, 30])
  # plt.show()
  import pdb; pdb.set_trace()
  v_all = np.stack(v_sim, axis=-1)
  vproj_all = np.stack(v_hardware, axis=-1)


  v_std = np.std(v_all, axis=2)
  v_mean = np.mean(v_all, axis=2)
  vproj_std = np.std(vproj_all, axis=2)
  vproj_mean = np.mean(vproj_all, axis=2)
  plt.figure('joint velocities')
  for i in range(12):
    ps.plot(t_master, v_mean[:, i], color=ps.cmap(i))
    ps.plot_bands(t_master, t_master, (v_mean - v_std)[:, i], (v_mean + v_std)[:, i], color=ps.cmap(i))
  # plt.xlim([-10, 50])
  # plt.ylim([-10, 10])
  plt.title('Joint Velocities')
  plt.xlabel('Time since Start of Impact (ms)')
  plt.ylabel('Velocity (rad/s)')
  ps.save_fig('joint_velocities_w_dev.png')
  plt.figure('projected joint velocities')
  for i in range(12):
    ps.plot(t_master, vproj_mean[:, i], color=ps.cmap(i))
    ps.plot_bands(t_master, t_master, (vproj_mean - vproj_std)[:, i], (vproj_mean + vproj_std)[:, i], color=ps.cmap(i))
  # plt.xlim([-10, 50])
  # plt.ylim([-3, 1])
  plt.title('Projected Joint Velocities')
  plt.xlabel('Time since Start of Impact (ms)')
  plt.ylabel('Velocity (rad/s)')
  ps.save_fig('projected_joint_velocities_w_dev.png')

  plt.show()

def get_window_around_contact_event(x_traj, t_x, start_time, end_time):
  # start_idx = np.argwhere(np.isclose(t_x, self.start_time, atol=5e-4))[0][0]
  # end_idx = np.argwhere(np.isclose(t_x, self.end_time, atol=5e-4))[0][0]
  start_idx = np.argwhere(np.isclose(t_x, start_time, atol=5e-4))[1][0]
  end_idx = np.argwhere(np.isclose(t_x, end_time, atol=5e-4))[1][0]
  window = slice(start_idx, end_idx)
  return t_x[window], x_traj[window]

def plot_velocity_trajectory(impact_data, log_num, indices):
  t_hardware = impact_data.t_x_hardware[log_num]
  x_hardware = impact_data.x_trajs_hardware[log_num]
  t_sim = impact_data.t_x_sim[log_num]
  x_sim = impact_data.x_trajs_sim[log_num]

  start_time = impact_data.start_times[log_num]
  end_time = start_time + 0.05

  t_hardware, x_hardware = get_window_around_contact_event(x_hardware, t_hardware, start_time, end_time)
  t_sim, x_sim = get_window_around_contact_event(x_sim, t_sim, start_time, end_time)
  for i in indices:
    # plt.figure(x_datatypes[i] + ': ' + str(i))
    plt.figure(log_num + '_' + x_datatypes[i] + ': ' + str(i))
    ps.plot(t_hardware, x_hardware[:, i])
    ps.plot(t_sim, x_sim[:, i])

def plot_centroidal_trajectory(impact_data, log_num, use_center_of_mass=False):
  t_hardware = impact_data.t_x_hardware[log_num]
  x_hardware = impact_data.x_trajs_hardware[log_num]
  t_sim = impact_data.t_x_sim[log_num]
  x_sim = impact_data.x_trajs_sim[log_num]

  start_time = impact_data.start_times[log_num]
  end_time = start_time + 0.05
  t_hardware, x_hardware = get_window_around_contact_event(x_hardware, t_hardware, start_time, end_time)
  t_sim, x_sim = get_window_around_contact_event(x_sim, t_sim, start_time, end_time)

  com_pos_sim = np.empty((t_sim.shape[0], 3))
  com_vel_sim = np.empty((t_sim.shape[0], 3))
  com_pos_hardware = np.empty((t_hardware.shape[0], 3))
  com_vel_hardware = np.empty((t_hardware.shape[0], 3))
  for i in range(t_sim.shape[0]):
    x_sim_i = x_sim[i]
    if use_center_of_mass:
      com_pos_sim[i] = kinematics_calculator.compute_center_of_mass_pos(x_sim_i)
      com_vel_sim[i] = kinematics_calculator.compute_center_of_mass_vel(x_sim_i)
    else:
      com_pos_sim[i] = x_sim_i[4:7]
      com_vel_sim[i] = x_sim_i[26:29]
    #
  for i in range(t_hardware.shape[0]):
    x_i = x_hardware[i]
    if use_center_of_mass:
      com_pos_hardware[i] = kinematics_calculator.compute_center_of_mass_pos(x_i)
      com_vel_hardware[i] = kinematics_calculator.compute_center_of_mass_vel(x_i)
    else:
      com_pos_hardware[i] = x_i[4:7]
      com_vel_hardware[i] = x_i[26:29]
    #
  # ps.plot(t_sim, com_pos_sim)
  # ps.plot(t_hardware, com_pos_hardware)
  plt.figure('x' + log_num)
  ps.plot(t_sim, com_vel_sim[:, 0])
  ps.plot(t_hardware, com_vel_hardware[:, 0])
  # plt.figure('y' + log_num)
  ps.plot(t_sim, com_vel_sim[:, 1])
  ps.plot(t_hardware, com_vel_hardware[:, 1])
  # plt.figure('z' + log_num)
  ps.plot(t_sim, com_vel_sim[:, 2])
  ps.plot(t_hardware, com_vel_hardware[:, 2])

  return

def grf_single_log(impact_data, log_num):
  lambda_hardware = impact_data.contact_forces_hardware[log_num]
  lambda_sim = impact_data.contact_forces_sim[log_num]
  # import pdb; pdb.set_trace()
  t_hardware = impact_data.t_x_hardware[log_num]
  t_sim = impact_data.t_x_sim[log_num]
  ps.plot(t_hardware, lambda_hardware[0, :, 2])
  ps.plot(t_hardware, lambda_hardware[2, :, 2])
  ps.plot(t_sim, lambda_sim[0, :, 2])
  ps.plot(t_sim, lambda_sim[1, :, 2])
  ps.plot(t_sim, lambda_sim[2, :, 2])
  ps.plot(t_sim, lambda_sim[3, :, 2])


def main():
  global ps
  global nominal_impact_time
  global impact_time
  global figure_directory
  global data_directory
  global sim_data_directory
  global terrain_heights
  global perturbations
  global penetration_allowances
  global threshold_durations
  global x_datatypes
  # global start_time
  # global end_time
  global kinematics_calculator

  # start_time = 30.64
  # end_time = start_time + 0.05
  # data_directory = '/home/yangwill/Documents/research/projects/impact_uncertainty/data/'
  data_directory = '/home/yangwill/Documents/research/projects/invariant_impacts/data/'
  sim_data_directory = '/home/yangwill/workspace/dairlib/examples/contact_parameter_learning/cassie_sim_data/'
  figure_directory = '/home/yangwill/Documents/research/projects/invariant_impacts/figures/'
  ps = PlotStyler()
  ps.set_default_styling(directory=figure_directory)

  with open("x_datatypes", "rb") as fp:
    x_datatypes = pickle.load(fp)

  impact_data = CassieImpactData()
  kinematics_calculator = KinematicsHelper()

  joint_vel_indices = range(29, 45)
  hip_joints_indices = range(29, 35)
  fb_vel_indices = range(23, 29)
  # joint_pos_indices = range(7, 23)
  # hip_index = range(29,30)
  # knee_index = range(29,30)

  joint_vel_indices = range(35, 37)

  # load all the data used for plotting
  for log_num in ['08', '15', '24']:
    # plt.figure(log_num)
    # grf_single_log(impact_data, log_num)
    # plot_velocity_trajectory(impact_data, log_num, joint_vel_indices)
    plot_centroidal_trajectory(impact_data, log_num)

  # plot_velocity_trajectory(impact_data, '08', hip_joints_indices)
  # plot_velocity_trajectory(impact_data, '21', joint_vel_indices)
  # plot_error_bands(impact_data)
  ps.show_fig()


if __name__ == '__main__':
  main()