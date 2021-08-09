from cassiemujoco import *
import time
import numpy as np
from mujoco_lcm_utils import *
# import lcm
from pydrake.lcm import DrakeLcm, Subscriber
import dairlib
import signal
from pydrake.trajectories import PiecewisePolynomial


class MujocoCassieSim():

  def __init__(self, sim_dt=5e-5, loss_filename='default_loss_weights', publish_state=True):
    self.actuator_index_map = {'hip_roll_left_motor': 0,
                               'hip_yaw_left_motor': 1,
                               'hip_pitch_left_motor': 2,
                               'knee_left_motor': 3,
                               'toe_left_motor': 4,
                               'hip_roll_right_motor': 5,
                               'hip_yaw_right_motor': 6,
                               'hip_pitch_right_motor': 7,
                               'knee_right_motor': 8,
                               'toe_right_motor': 9}
    self.realtime_rate = 1.0
    self.cycle_usec = 1000000 / (2000 * self.realtime_rate)
    self.lcm = DrakeLcm()
    self.model_file = '/home/yangwill/workspace/cassie-mujoco-sim/model/cassie.xml'
    self.robot_output = init_robot_output()
    self.cassie_out = init_cassie_out()
    self.u_pd = pd_in_t()
    self.cassie_in = cassie_user_in_t()
    self.cassie_env = CassieSim(self.model_file)
    self.input_sub = Subscriber(lcm=self.lcm, channel="CASSIE_INPUT", lcm_type=dairlib.lcmt_robot_input)
    self.publish_state = publish_state

  def pack_input(self, cassie_in, u):
    # Set control parameters
    for i in range(10):
      cassie_in.torque[i] = u[i]
    return cassie_in

  def pack_cassie_out(self):
    return self.cassie_out

  def unpack_robot_in(self, robot_in):
    u = np.zeros(10)
    for i in range(10):
      j = self.actuator_index_map[robot_in.effort_names[i]]
      u[j] = robot_in.efforts[i]
    return u

  def pack_input_pd(self):
    self.u_pd.leftLeg.motorPd.torque[3] = 0  # Feedforward torque
    self.u_pd.leftLeg.motorPd.pTarget[3] = -2
    self.u_pd.leftLeg.motorPd.pGain[3] = 100
    self.u_pd.leftLeg.motorPd.dTarget[3] = -2
    self.u_pd.leftLeg.motorPd.dGain[3] = 10
    self.u_pd.rightLeg.motorPd = u_pd.leftLeg.motorPd
    return self.u_pd

  def set_contact_params(self):
    return

  def convert_drake_state_to_mujoco_state(self, x):

    return x_full

  def sim_step(self, u):
    action = self.pack_input(self.cassie_in, u)
    self.cassie_env.step(action)
    t = self.cassie_env.time()
    q = self.cassie_env.qpos()
    v = self.cassie_env.qvel()
    return q, v, u, t

  def set_state(self, x_init):
    mujoco_state = self.convert_drake_state_to_mujoco_state(x_init)
    self.cassie_env.set_state(mujoco_state)

  def run_sim_playback(self, x_init, start_time, end_time, input_traj=PiecewisePolynomial(np.zeros(10))):
    self.cassie_env.set_time(start_time)
    self.set_state(x_init)
    # u = np.zeros(10)
    x_traj = []
    u_traj = []
    t_traj = []
    t = start_time
    while self.cassie_env.time() < end_time:
      now = time.time()  # get the time
      u = input_traj.value(t)
      q, v, u, t = self.sim_step(u)
      if (self.publish_state):
        robot_output = pack_robot_output(self.robot_output, q, v, u, t)
        self.lcm.Publish('CASSIE_STATE_SIMULATION', self.robot_output.encode())
      x_traj.append([q, v])
      u_traj.append(u)
      t_traj.append(t)
      elapsed = time.time() - now
      time.sleep(self.cycle_usec * 1e-6 - elapsed)

    return x_traj, u_traj, t_traj

  def run_sim(self, start_time, end_time, x_init=np.zeros(0)):
    self.cassie_env.set_time(start_time)
    if x_init.size != 0:
      self.set_state(x_init)
    u = np.zeros(10)
    x_traj = []
    u_traj = []
    t_traj = []
    t = start_time
    while self.cassie_env.time() < end_time:
      now = time.time()  # get the time
      self.lcm.HandleSubscriptions(1)
      if len(self.input_sub.message.efforts) != 0:
        u = self.unpack_robot_in(self.input_sub.message)
      q, v, u, t = self.sim_step(u)
      if (self.publish_state):
        robot_output = pack_robot_output(self.robot_output, q, v, u, t)
        self.lcm.Publish('CASSIE_STATE_SIMULATION', self.robot_output.encode())
      x_traj.append([q, v])
      u_traj.append(u)
      t_traj.append(t)
      elapsed = time.time() - now
      if(elapsed < self.cycle_usec * 1e-6):
        time.sleep(self.cycle_usec * 1e-6 - elapsed)

    return x_traj, u_traj, t_traj
    # cassie_env.step_pd(u_pd)

    # import pdb; pdb.set_trace()

    # print(state)
