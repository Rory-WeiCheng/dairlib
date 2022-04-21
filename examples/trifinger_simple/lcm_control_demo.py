from dairlib import (lcmt_robot_output, lcmt_robot_input)

import pydairlib.common
import pydairlib.lcm
from pydairlib.systems import (RobotCommandSender, RobotOutputReceiver,
                               LcmOutputDrivenLoop, OutputVector,
                               TimestampedVector)

from pydrake.all import (AbstractValue, DiagramBuilder, DrakeLcm, LeafSystem,
                         MultibodyPlant, Parser, RigidTransform, Subscriber,
                         LcmPublisherSystem, TriggerType, AddMultibodyPlantSceneGraph)
#import pydairlib.common


from pydairlib.multibody import (addFlatTerrain, makeNameToPositionsMap)
import pydairlib.common
from pydairlib.systems.controllers import C3Controller

import numpy as np


lcm = DrakeLcm()

plant = MultibodyPlant(0.0)


#The package addition here seems necessary due to how the URDF is defined
parser = Parser(plant)
parser.package_map().Add("robot_properties_fingers",
                         "examples/trifinger_simple/robot_properties_fingers")
parser.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/trifinger_simple/robot_properties_fingers/urdf/trifinger_minimal_collision_2.urdf"))
# parser.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
#     "examples/trifinger_simple/robot_properties_fingers/cube/cube_v2.urdf"))
parser.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/trifinger_simple/robot_properties_fingers/urdf/sphere.urdf"))


#Fix the base of the finger to the world
X_WI = RigidTransform.Identity()
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link"), X_WI)
plant.Finalize()

builder = DiagramBuilder()

state_receiver = builder.AddSystem(RobotOutputReceiver(plant))

#############################################################################################


builder_f = DiagramBuilder()
sim_dt = 1e-4
plant_f, scene_graph = AddMultibodyPlantSceneGraph(builder_f, 0.0)
# addFlatTerrain(plant=plant_f, scene_graph=scene_graph, mu_static=1.0,
#                mu_kinetic=1.0)

# The package addition here seems necessary due to how the URDF is defined
parser_f = Parser(plant_f)
parser_f.package_map().Add("robot_properties_fingers",
                         "examples/trifinger_simple/robot_properties_fingers")
parser_f.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/trifinger_simple/robot_properties_fingers/urdf/trifinger_minimal_collision_2.urdf"))
# parser_f.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
#     "examples/trifinger_simple/robot_properties_fingers/cube/cube_v2.urdf"))
parser_f.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/trifinger_simple/robot_properties_fingers/urdf/sphere.urdf"))


# Fix the base of the finger to the world
X_WI_f = RigidTransform.Identity()
plant_f.WeldFrames(plant_f.world_frame(), plant_f.GetFrameByName("base_link"), X_WI_f)
plant_f.Finalize()


#context_f = plant_f.CreateDefaultContext()

plant_ad_f = plant_f.ToAutoDiffXd()

context_ad_f = plant_ad_f.CreateDefaultContext()

diagram_f = builder_f.Build()

diagram_context = diagram_f.CreateDefaultContext()

context_f = diagram_f.GetMutableSubsystemContext(plant_f, diagram_context)

#############################################################################################

# Set the initial state
#q = [0.0594048, -0.663454, -0.925483, 0.0594048, -0.663454, -0.925483, 0.0594048, -0.663454,-0.925483, 0.999988, 0, 0, -0.00480895, 0, 4.16845e-07, 0.0324052]
#q = np.reshape(q, (16,1))


nq = plant.num_positions()
nv = plant.num_velocities()
nu = plant.num_actuators()
nc = 2

print(nq)

q = np.zeros((nq,1))
q_map = makeNameToPositionsMap(plant)
q[0] = 0
q[1] = -0.1
q[2] = 0.02
# q[3] = 0.2
# q[4] = 0
# q[5] = 0.02
# q[6] = 0
# q[7] = 0.1
# q[8] = 0.02
q[q_map['base_qw']] = 1
q[q_map['base_qx']] = 0
q[q_map['base_qz']] = 0
q[q_map['base_x']] = 1
q[q_map['base_y']] = 0
q[q_map['base_z']] = 0

mu = 1.0




Qinit = 0*np.eye(nq+nv)
Qinit[7:9,7:9] = 1*np.eye(2)
#print(Qinit)

# Qinit[0:9,0:9] = 0*np.eye(9) #trifinger
#
# #Qinit[2] = 10
# #Qinit[5] = 10
# #Qinit[8] = 10
# Qinit[9:16,9:16] = 100*np.eye(7) #cube
# Qinit[16:31,16:31] = 0*np.eye(15) #velocities
# Qinit[16:25,16:25] = 1*np.eye(9) #vel
# Qinit[15,15] = 1000
#print(Qinit)
Rinit = 0.01*np.eye(nu) #torques
#admm_params
Ginit = 0.01*np.eye(nq+nv+nu+6*nc)
Ginit[0:3,0:3] = 0*np.eye(3)

#Ginit[0:9,0:9] = 0*np.eye(9)
#Ginit[16:31,16:31] = 0*np.eye(15)
# Ginit[9:16,9:16] = 0*np.eye(7)
# Ginit[31:49,31:49] = 0*np.eye(18)
Uinit = 1*np.eye(nq+nv+nu+6*nc)
Uinit[0:nq,0:nq] = 100*np.eye(nq)

#Uinit[9:16,9:16] = 100*np.eye(7)
#Uinit[0:31,0:31] = 100*np.eye(31)
#Uinit[55:64,55:64] = 1*np.eye(9)
xdesiredinit = np.zeros((nq+nv,1))
xdesiredinit[:nq] = q

#Qinit[14,14] = 100

# Ginit = 0.1*np.eye(46)
# Uinit = 0.1*np.eye(46)


# Qinit[0,0] = 10000
# Qinit[3,3] = 10000
# Qinit[6,6] = 10000



num_friction_directions = 2
N = 5
Q = []
R = []
G = []
U = []
xdesired = []

for i in range(N):
    Q.append(Qinit)
    R.append(Rinit)
    G.append(Ginit)
    U.append(Uinit)
    xdesired.append(xdesiredinit)
Q.append(Qinit)
xdesired.append(xdesiredinit)

finger_lower_link_0_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("tip_link_1_real"))[0]
#finger_lower_link_120_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("tip_link_2_real"))[0]
#finger_lower_link_240_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("tip_link_3_real"))[0]
#cube_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("cube"))[0]
sphere_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("sphere"))[0]
ground_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("box"))[0]

contact_geoms = [finger_lower_link_0_geoms, sphere_geoms, ground_geoms] #finger_lower_link_120_geoms, finger_lower_link_240_geoms,


plant_ad = plant.ToAutoDiffXd()

context = plant.CreateDefaultContext()

context_ad = plant_ad.CreateDefaultContext()

controller = builder.AddSystem(
    C3Controller(plant, plant_f, context, context_f, plant_ad, plant_ad_f, context_ad, context_ad_f, scene_graph, diagram_f, contact_geoms, num_friction_directions, mu, Q, R, G, U, xdesired))


#controller = builder.AddSystem(TrifingerDemoController(plant))


builder.Connect(state_receiver.get_output_port(0), controller.get_input_port(0))

control_sender = builder.AddSystem(RobotCommandSender(plant))
builder.Connect(controller.get_output_port(), control_sender.get_input_port(0))

control_publisher = builder.AddSystem(LcmPublisherSystem.Make(
    channel="TRIFINGER_INPUT", lcm_type=lcmt_robot_input, lcm=lcm,
    publish_triggers={TriggerType.kForced},
    publish_period=0.0, use_cpp_serializer=True))
builder.Connect(control_sender.get_output_port(),
    control_publisher.get_input_port())

diagram = builder.Build()

context_d = diagram.CreateDefaultContext()
receiver_context = diagram.GetMutableSubsystemContext(state_receiver, context_d)

loop = LcmOutputDrivenLoop(drake_lcm=lcm, diagram=diagram,
                          lcm_parser=state_receiver,
                          input_channel="TRIFINGER_OUTPUT",
                          is_forced_publish=True)

loop.Simulate(20)