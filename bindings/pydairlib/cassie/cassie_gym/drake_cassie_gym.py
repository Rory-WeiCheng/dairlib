import numpy as np
from dataclasses import dataclass

from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder, Context, InputPort
from pydrake.multibody.plant import *
from pydrake.systems.analysis import Simulator
from pydrake.systems.sensors import ImageToLcmImageArrayT, PixelType
from pydrake.systems.lcm import LcmPublisherSystem
from pydrake.lcm import DrakeLcm
from drake import lcmt_image_array

from pydairlib.common import FindResourceOrThrow
from pydairlib.cassie.cassie_utils import *
from pydairlib.multibody import *
from pydairlib.systems.primitives import *
from pydairlib.systems.robot_lcm_systems import RobotOutputSender
from dairlib import lcmt_radio_out
from pydairlib.cassie.simulators import CassieVisionSimDiagram
from pydairlib.cassie.cassie_gym.cassie_env_state import CassieEnvState


@dataclass
class CassieGymParams:
    """
        Container class for the parameters which could
        be randomized for simulation
    """
    terrain_normal: np.ndarray = np.array([0.0, 0.0, 1.0])
    x_init: np.ndarray = np.array(
        [1, 0, 0, 0, 0, 0, 0.85,
         -0.0358, 0, 0.674, -1.588, -0.0458, 1.909, -0.0382, -1.823,
          0.0358, 0, 0.674, -1.588, -0.0458, 1.909, -0.0382, -1.823,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    map_yaw: float = 0
    mu: float = 0.8

    @staticmethod
    def make_random(ic_file_path):
        ics = np.load(ic_file_path)
        x = ics[np.random.choice(ics.shape[0], size=1, replace=False)].ravel()
        normal = np.random.uniform(
            low=[-0.1, -0.1, 1.0],
            high=[0.1, 0.1, 1.0],
        )
        map_yaw = 2 * np.random.random() - 1
        mu = 0.5 * np.random.random() + 0.5
        return CassieGymParams(
            terrain_normal=normal,
            x_init=x,
            map_yaw=map_yaw,
            mu=mu
        )


@dataclass
class FixedVectorInputPort:
    input_port: InputPort = None
    context: Context = None
    value: np.ndarray = None


class DrakeCassieGym:
    def __init__(self, visualize=False, params=CassieGymParams()):
        self.params = params
        self.sim_dt = 1e-2
        self.visualize = visualize
        self.start_time = 0.00
        self.current_time = 0.00
        self.controller = None
        self.terminated = False
        self.initialized = False
        self.cassie_state = None
        self.prev_cassie_state = None
        self.traj = None

        # Simulation objects
        self.builder = None
        self.plant = None
        self.cassie_sim = None
        self.sim_plant = None
        self.diagram = None
        self.drake_simulator = None
        self.cassie_sim_context = None
        self.controller_context = None
        self.plant_context = None
        self.controller_output_port = None

    def make(self, controller, urdf='examples/Cassie/urdf/cassie_v2.urdf'):
        self.builder = DiagramBuilder()
        self.plant = MultibodyPlant(8e-5)
        self.controller = controller
        self.cassie_sim = CassieVisionSimDiagram(
            plant=self.plant,
            urdf=urdf,
            visualize=self.visualize,
            mu=self.params.mu,
            map_yaw=self.params.map_yaw,
            normal=self.params.terrain_normal)
        self.sim_plant = self.cassie_sim.get_plant()
        self.builder.AddSystem(self.controller)
        self.builder.AddSystem(self.cassie_sim)

        if self.visualize:
            self.lcm = DrakeLcm()
            self.image_array_sender = self.builder.AddSystem(
                ImageToLcmImageArrayT())
            self.image_array_sender.DeclareImageInputPort[PixelType.kDepth32F]("depth")
            self.image_array_publisher = self.builder.AddSystem(
                LcmPublisherSystem.Make(
                    "DRAKE_RGBD_CAMERA_IMAGES",
                    lcm_type=lcmt_image_array,
                    lcm=self.lcm,
                    publish_period=0.1,
                    use_cpp_serializer=True))

            self.builder.Connect(self.cassie_sim.get_camera_out_output_port(),
                                 self.image_array_sender.get_input_port())
            self.builder.Connect(self.image_array_sender.get_output_port(),
                                 self.image_array_publisher.get_input_port())

        self.builder.Connect(self.controller.get_control_output_port(),
                             self.cassie_sim.get_actuation_input_port())
        self.builder.Connect(self.cassie_sim.get_state_output_port(),
                             self.controller.get_state_input_port())

        self.diagram = self.builder.Build()
        self.drake_simulator = Simulator(self.diagram)
        self.set_context_members(self.drake_simulator.get_mutable_context())
        self.controller_output_port = self.controller.get_torque_output_port()
        self.drake_simulator.get_mutable_context().SetTime(self.start_time)
        self.reset()
        self.initialized = True

    def set_context_members(self, diagram_context):
        self.cassie_sim_context = \
            self.diagram.GetMutableSubsystemContext(
                self.cassie_sim, diagram_context)
        self.controller_context = \
            self.diagram.GetMutableSubsystemContext(
                self.controller, diagram_context)
        self.plant_context = \
            self.diagram.GetMutableSubsystemContext(
                self.sim_plant, diagram_context)

    def reset(self):
        new_context = self.diagram.CreateDefaultContext()
        self.drake_simulator.reset_context(new_context)
        self.set_context_members(
            self.drake_simulator.get_mutable_context()
        )
        self.sim_plant.SetPositionsAndVelocities(self.plant_context, self.params.x_init)
        self.drake_simulator.get_mutable_context().SetTime(self.start_time)
        x = self.sim_plant.GetPositionsAndVelocities(self.plant_context)
        u = np.zeros(10)
        self.drake_simulator.Initialize()
        self.current_time = self.start_time
        self.prev_cassie_state = CassieEnvState(self.current_time, x, u)
        self.cassie_state = CassieEnvState(self.current_time, x, u)
        self.terminated = False
        return

    def advance_to(self, time):

        while self.current_time < time and not self.terminated:
            self.step()
        return

    def check_termination(self):
        left_foot_pos = self.sim_plant.CalcPointsPositions(
            context=self.plant_context,
            frame_B=self.sim_plant.GetBodyByName("toe_left").body_frame(),
            p_BQi=np.zeros(3).T,
            frame_A=self.sim_plant.world_frame()
        ).flatten()[2]
        right_foot_pos = self.sim_plant.CalcPointsPositions(
            context=self.plant_context,
            frame_B=self.sim_plant.GetBodyByName("toe_right").body_frame(),
            p_BQi=np.zeros(3).T,
            frame_A=self.sim_plant.world_frame()
        ).flatten()[2]
        pelvis_z = self.cassie_state.get_fb_positions()[2]
        return pelvis_z < 0.4 or right_foot_pos > pelvis_z - 0.3 or \
               left_foot_pos > pelvis_z - 0.3

    def step(self, radio=np.zeros(18), fixed_ports=None, time=None):
        if not self.initialized:
            print("Call make() before calling step() or advance()")

        # Calculate next timestep
        next_timestep = self.drake_simulator.get_context().get_time() + \
                        (self.sim_dt if time is None else time)

        # Set simulator inputs and advance simulator
        self.cassie_sim.get_radio_input_port().FixValue(
            context=self.cassie_sim_context,
            value=radio)
        self.controller.get_radio_input_port().FixValue(
            context=self.controller_context,
            value=radio)

        if fixed_ports is not None:
            for port in fixed_ports:
                port.input_port.FixValue(
                    context=port.context,
                    value=port.value
                )
        self.drake_simulator.AdvanceTo(next_timestep)
        self.current_time = self.drake_simulator.get_context().get_time()

        # Get the state
        x = self.sim_plant.GetPositionsAndVelocities(self.plant_context)
        u = self.controller_output_port.Eval(
            self.controller_context)[:-1]  # remove the timestamp
        self.cassie_state = CassieEnvState(self.current_time, x, u)
        self.terminated = self.check_termination()
        self.prev_cassie_state = self.cassie_state
        return self.cassie_state

    def get_traj(self):
        return self.traj

    # Some simulators for Cassie require cleanup
    def free_sim(self):
        return
