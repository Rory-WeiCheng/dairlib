from pyexpat.model import XML_CQUANT_NONE
# pydrake
from pydrake.common.yaml import yaml_load
from pydrake.all import (AbstractValue, DiagramBuilder, DrakeLcm, LeafSystem,
                         MultibodyPlant, Parser, RigidTransform, Subscriber,
                         LcmPublisherSystem, TriggerType, AddMultibodyPlantSceneGraph,
                         LcmInterfaceSystem)



# pydairlib
from dairlib import (lcmt_robot_output, lcmt_robot_input, lcmt_lcs, lcmt_c3)
from pydairlib.systems import (RobotLCSSender,RobotC3Receiver,
                               RobotC3Sender, ResidualLearner,
                               RobotOutputReceiver, RobotOutputSender,
                               LcmOutputDrivenLoop, OutputVector,
                               TimestampedVector,
                               AddActuationRecieverAndStateSenderLcm)
from pydairlib.multibody import (makeNameToPositionsMap, makeNameToVelocitiesMap)
import pydairlib.common

# residual learner
# from residual_learner import ResidualLearner

# others
import numpy as np
import math


lcm = DrakeLcm()
builder = DiagramBuilder()

builder_franka = DiagramBuilder()
sim_dt = 1e-4
plant_franka, scene_graph_franka = AddMultibodyPlantSceneGraph(builder_franka, sim_dt)
# The package addition here seems necessary due to how the URDF is defined
parser_franka = Parser(plant_franka)
parser_franka.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/franka_trajectory_following/robot_properties_fingers/urdf/franka_box.urdf"))
parser_franka.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/franka_trajectory_following/robot_properties_fingers/urdf/sphere_model.urdf"))
# Fix the base of the finger to the world
X_WI_franka = RigidTransform.Identity()
plant_franka.WeldFrames(plant_franka.world_frame(), plant_franka.GetFrameByName("panda_link0"), X_WI_franka)
plant_franka.Finalize()
context_franka = plant_franka.CreateDefaultContext()



state_receiver = builder.AddSystem(RobotOutputReceiver(plant_franka))
learner = builder.AddSystem(ResidualLearner())
lcs_sender = builder.AddSystem(RobotLCSSender())

builder.Connect(state_receiver.get_output_port(0), learner.get_input_port(0))
builder.Connect(learner.get_output_port(0), lcs_sender.get_input_port(0))
import pdb; pdb.set_trace()
lcs_publisher = builder.AddSystem(LcmPublisherSystem.Make(
    channel="RESIDUAL_LCS", lcm_type=lcmt_lcs, lcm=lcm,
    publish_triggers={TriggerType.kForced},
    publish_period=0.0, use_cpp_serializer=True))
builder.Connect(lcs_sender.get_output_port(),lcs_publisher.get_input_port())

diagram = builder.Build()
context_d = diagram.CreateDefaultContext()
# receiver_context = diagram.GetMutableSubsystemContext(state_receiver, context_d)

loop = LcmOutputDrivenLoop(drake_lcm=lcm, diagram=diagram,
                          lcm_parser=state_receiver,
                          input_channel="FRANKA_STATE_ESTIMATE",
                          is_forced_publish=True)

loop.Simulate(math.inf)
