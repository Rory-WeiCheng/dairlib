import numpy as np
import os
import pickle
from json import dump, load
import concurrent.futures as futures
import nevergrad as ng

from pydairlib.cassie.gym_envs.cassie_gym import *
# from cassie_utils import *
from pydairlib.cassie.controllers import OSCRunningControllerFactory
from pydairlib.cassie.controllers import OSCWalkingControllerFactory
from pydairlib.cassie.simulators import CassieSimDiagram
from pydairlib.cassie.gym_envs.reward_osudrl import RewardOSUDRL
from pydrake.common.yaml import yaml_load, yaml_dump
import time
import matplotlib.pyplot as plt
import yaml


class OSCGainsOptimizer():

    def __init__(self, budget, reward_function):
        self.budget = budget
        self.total_loss = 0
        self.reward_function = reward_function
        self.end_time = 5.0
        self.gym_env = None
        self.sim = None
        self.controller = None

        self.urdf = 'examples/Cassie/urdf/cassie_v2.urdf'
        self.default_osc_running_gains_filename = 'examples/Cassie/osc_run/osc_running_gains.yaml'
        self.osc_running_gains_filename = 'examples/Cassie/osc_run/learned_osc_running_gains.yaml'
        self.osqp_settings = 'examples/Cassie/osc_run/osc_running_qp_settings.yaml'

        self.drake_params_folder = "bindings/pydairlib/cassie/optimal_gains/"
        self.date_prefix = time.strftime("%Y_%m_%d_%H")
        self.loss_over_time = []

        self.default_osc_gains = {
            'SwingFootKp': np.array([125, 80, 50]),
            'SwingFootKd': np.array([5, 5, 1]),
            'FootstepKd': np.array([0.2, 0.45, 0]),
            'center_line_offset': 0.03,
        }

    def save_params(self, folder, params, budget):
        with open(folder + self.date_prefix + '_' + str(budget) + '.pkl', 'wb') as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    def load_params(self, param_file, folder):
        with open(folder + param_file + '.pkl', 'rb') as f:
            return pickle.load(f)

    def write_params(self, params):
        gains = yaml_load(filename=self.default_osc_running_gains_filename, private=True)
        new_gains = gains.copy()
        for key in params:
            if hasattr(params[key], "__len__"):
                new_gains[key] = np.diag(params[key]).flatten().tolist()
            else:
                new_gains[key] = params[key]
        yaml_dump(new_gains, filename=self.osc_running_gains_filename)

    def get_single_loss(self, params):
        self.write_params(params)
        controller_plant = MultibodyPlant(8e-5)
        addCassieMultibody(controller_plant, None, True, self.urdf, False, False)
        controller_plant.Finalize()
        self.controller = OSCRunningControllerFactory(controller_plant, self.osc_running_gains_filename,
                                                      self.osqp_settings)
        self.gym_env.make(self.controller, self.urdf)
        # rollout a trajectory and compute the loss
        cumulative_reward = 0
        while self.gym_env.current_time < 7.5 and not self.gym_env.terminated:
            state, reward = self.gym_env.step(np.zeros(18))
            cumulative_reward += reward
        self.loss_over_time.append(cumulative_reward)
        print(-cumulative_reward)
        return -cumulative_reward

    def learn_drake_params(self, batch=True):
        self.loss_over_time = []
        self.default_params = ng.p.Dict(
            SwingFootKp=ng.p.Array(lower=0., upper=150., shape=(3,)),
            SwingFootKd=ng.p.Array(lower=0., upper=15., shape=(3,)),
            FootstepKd=ng.p.Array(lower=0., upper=1., shape=(3,)),
            center_line_offset=ng.p.Scalar(lower=0.01, upper=0.1),
        )
        self.gym_env = CassieGym(self.reward_function, visualize=True)
        self.default_params.value = self.default_osc_gains
        optimizer = ng.optimizers.NGOpt(parametrization=self.default_params, budget=self.budget)
        # with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        params = optimizer.minimize(self.get_single_loss)
        loss_samples = np.array(self.loss_over_time)
        np.save(self.drake_params_folder + 'loss_trajectory_' + str(self.budget), loss_samples)
        self.save_params(self.drake_params_folder, params, budget)


if __name__ == '__main__':
    # budget = 2000
    budget = 1000

    reward_function = RewardOSUDRL()

    optimizer = OSCGainsOptimizer(budget, reward_function)
    optimizer.learn_drake_params()

    # optimal_params = optimizer.load_params('2022_03_22_11_100', optimizer.drake_params_folder).value
    # optimizer.write_params(optimal_params)
    # reward_over_time = np.load('bindings/pydairlib/cassie/optimal_gains/loss_trajectory_100.npy')
    # plt.plot(reward_over_time)
    # plt.show()
