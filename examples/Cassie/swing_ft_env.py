import gym
from gym import spaces
import os
import lcm
import queue
import threading
from drake import lcmt_scope
import subprocess as sp
import time 
import numpy as np

from dairlib import lcmt_swing_foot_spline_params, lcmt_robot_output, lcmt_radio_out


class LCMInterface:
    def __init__(self, channels, message_types, address=None):
        assert len(channels) == len(message_types)
        self.channels = channels 
        self.message_types = {channels[i]:message_types[i] for i in range(len(channels))}
        self.lc = lcm.LCM()
        # self.queues = {channel:queue.LifoQueue() for channel in self.channels}
        # self.values = {channel:None for channel in self.channels}
        # self.locks = {channel:threading.Lock() for channel in self.channels}
        self.stop_listener = threading.Event()
        for c in channels:
            self.lc.subscribe(c, self.handler)

    def listener_loop(self):
        while True:
            self.lc.handle() 
            if self.stop_listener.is_set():
                break
            time.sleep(0.001)
 
    def start_listening(self):
        self.stop_listener.clear()
        self.listener_thread = threading.Thread(target=self.listener_loop)
        self.listener_thread.start()
        return

    def stop_reset_listening(self):
        self.stop_listener.set()
        self.queues = {channel:queue.LifoQueue() for channel in self.channels}

    def handler(self, channel, data):
        self.queues[channel].put(self.message_types[channel].decode(data))

    def get_latest(self, channel):
        return self.queues[channel].get(block=True)

    def publish(self, channel, message):
        self.lc.publish(channel, message.encode())


class CassieSwingFootEnv:
    """
        Class that provides and interface to run_osc_walking_controller_alip
        simulation in Drake, with an action space defined by the
        lcmt_swing_foot_spline_params structure.
    """
    def __init__(self, default_swing,
                 lcm_address=None,
                 use_radio=False,
                 sim=True,
                 viz=True):

        self.reward_channel = "SWING_FOOT_REWARD"
        self.fsm_channel = "FINITE_STATE_MACHINE"
        self.action_channel = "SWING_FOOT_PARAMS"
        if sim:
            self.state_channel = "CASSIE_STATE_SIMULATION"
        else:
            self.state_channel = "CASSIE_STATE_DISPATCHER"
        self.radio_channel = "CASSIE_VIRTUAL_RADIO"
        self.using_sim = sim
        self.use_radio = use_radio
        self.single_support_states = [0, 1]

        self.lcm_interface = LCMInterface(
            [self.reward_channel, self.fsm_channel, self.state_channel],
            [lcmt_scope, lcmt_scope, lcmt_robot_output])

        # TODO: define the observation, action spaces if we want
        #  to use something like StableBaselines
        self.default_swing = default_swing

        # Infrastructure definitions
        self.ctrlr, self.sim = None, None
        self.bin_dir = "./bazel-bin/examples/Cassie/"
        self.controller_p = "run_osc_walking_controller_alip"
        self.sim_p = "multibody_sim"
        self.ctrlr_options = ["--learn_swing_foot_path=true"]
        if self.use_radio:
            self.ctrlr_options.append("--use_radio=true")
        # self.ctrlr_options = []
        self.sim_options = ["--publish_rate=2000", "--init_height=0.95",
                            "--target_realtime_rate=1.0"]
        self.viz = viz
        if self.viz:
            self.viz_options = \
                ["--floating_base=true", "--channel="+self.state_channel]
            self.visualizer = sp.Popen(["bazel-bin/examples/Cassie/visualizer"]
                                       + self.viz_options)

    def kill_procs(self):
        if self.sim is not None:
            self.sim.terminate()
            self.sim = None
        if self.ctrlr is not None:
            self.ctrlr.terminate()
            self.ctrlr = None
        
    def kill_director(self):
        if self.drake_director is not None:
            self.drake_director.terminate()
        if self.visualizer is not None:
            self.visualizer.terminate()

    def fill_action_message(self, action):
        action_msg = lcmt_swing_foot_spline_params()
        action_msg.n_knot = int(action[0])
        lst = np.zeros((int(action[0]), 3))
        for k in range(int(action[0])):
            for n in range(3):
                lst[k][n] = action[1 + 3 * k + n]
        action_msg.knot_xyz = lst.tolist()
        action_msg.swing_foot_vel_initial = action[-6:-3]
        action_msg.swing_foot_vel_final = action[-3:]
        return action_msg

    def fill_radio_message(self, radio):
        radio_msg = lcmt_radio_out()
        radio_msg.channel[0:4] = radio 
        return radio_msg

    def step(self, action, radio=None):
        """ Sends swing foot spline action to Drake, and receives back
        the new state and reward. Runs SYNCHRNOUSLY with sim so is 
        runtime-sensitive.

        param action: 1 + 3*action[0] + 3 + 3 array
        """
        assert (len(action) == 1+3*action[0] + 6), \
            "action length must match # of knot points!"

        # publish action, radio messages
        action_msg = self.fill_action_message(action)
        self.lcm_interface.publish(self.action_channel, action_msg)
        if self.use_radio and radio is not None:
            radio_msg = self.fill_radio_message(radio)
            self.lcm_interface.publish(self.radio_channel, radio_msg)

        cur_fsm_state = self.lcm_interface.get_latest(self.fsm_channel).value[0]
        val = cur_fsm_state
        start_t = time.time()
        while int(cur_fsm_state) not in self.single_support_states:
            cur_fsm_state = self.lcm_interface.get_latest(self.fsm_channel).value[0]
            time.sleep(.001)
            if time.time() - start_t > 2:
                self.state = self.select_states(self.lcm_interface.get_latest(self.state_channel))
                r = self.lcm_interface.get_latest(self.reward_channel).value[0]
                return self.state, r, True 

        start_t = time.time()
        while val == cur_fsm_state:
            val = self.lcm_interface.get_latest(self.fsm_channel).value[0]
            time.sleep(0.001)
            if time.time() - start_t > 2:
                self.state = self.select_states(self.lcm_interface.get_latest(self.state_channel))
                r = self.lcm_interface.get_latest(self.reward_channel).value[0]
                return self.state, r, True 

        reward = self.lcm_interface.get_latest(self.reward_channel).value[0]
        self.state = self.select_states(self.lcm_interface.get_latest(self.state_channel))
        # check failure on the state (has the robot fallen over?)
        failed = self.check_failure(self.state)
        return self.state, reward, failed 

    def reset(self):
        """
        Restarts sim & controller, returns state
        """

        # Kill controller and sim if running
        self.kill_procs()

        # reset LCM buffers
        self.lcm_interface.stop_reset_listening()

        #  Start walking controller process
        self.ctrlr = sp.Popen(
            [os.path.join(self.bin_dir, self.controller_p)] + self.ctrlr_options)

        # Wait for controller diagram to build before
        # sending default swing foot params message
        time.sleep(0.1)
        self.lcm_interface.publish(
            self.action_channel, self.fill_action_message(self.default_swing))

        # start simulation
        if self.using_sim:
            self.sim = sp.Popen(
                [os.path.join(self.bin_dir, self.sim_p)] + self.sim_options)
            # need to let the initial conditions solve, initialize
            time.sleep(1.5)
        self.lcm_interface.start_listening()
        self.state = self.select_states(
            self.lcm_interface.get_latest(self.state_channel))
        return self.state


    def select_states(self, full_cassie_state):
        """
        Selects reduced states from the full robot_out
        LCM message. For now selecting only CoM pos + vel.
        """
        return list(full_cassie_state.position)[0:7] + list(full_cassie_state.velocity)[0:6]


    def check_failure(self, state):
        """ Checks if the robot has fallen down
        """
        cond = False
        if self.ctrlr is not None:
            cond = self.ctrlr.poll() is not None
        if self.sim is not None:
            cond = self.sim.poll() is not None
        return state[6] < 0.6 or self.ctrlr is None or self.sim is None or cond


def main():
    try:
        env = CassieSwingFootEnv() 
        s = env.reset()
    except KeyboardInterrupt:
        env.kill_procs()

if __name__ == "__main__":
    main()