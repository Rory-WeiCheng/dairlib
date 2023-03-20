from pydrake.common.yaml import yaml_load
from pydrake.all import (AbstractValue, DiagramBuilder, DrakeLcm, LeafSystem,
                         MultibodyPlant, Parser, RigidTransform, Subscriber,
                         LcmPublisherSystem, TriggerType, AddMultibodyPlantSceneGraph,
                         LcmInterfaceSystem)
from pydairlib.systems import (lcs,lcs_factory_franka_new)

# newly added
from pydrake.multibody.tree import JacobianWrtVariable,MultibodyForces_
from pydrake.autodiffutils import InitializeAutoDiff,AutoDiffXd,ExtractGradient,ExtractValue

import pydairlib.common
import numpy as np
from examples.franka_trajectory_following.scripts.franka_logging_utils_test import get_most_recent_logs
import time


start_time = time.time()
# load parameters
param = yaml_load(
    filename="examples/franka_trajectory_following/parameters.yaml")

####################################### plant_f， simplified model ###########################################
# initialize the diagram builder: builder_f
builder_f = DiagramBuilder()
sim_dt = 1e-4

# Adds both MultibodyPlant: plant_f and the SceneGraph: scene_graph, and wires them together.
plant_f, scene_graph = AddMultibodyPlantSceneGraph(builder_f, 0.0)

# The same as above, package addition
parser_f = Parser(plant_f)
parser_f.package_map().Add("robot_properties_fingers",
                         "examples/franka_trajectory_following/robot_properties_fingers")
# load the end effector (finger only) urdf
parser_f.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/franka_trajectory_following/robot_properties_fingers/urdf/trifinger_minimal_collision_2.urdf"))
# load the ball urdf
parser_f.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/franka_trajectory_following/robot_properties_fingers/urdf/sphere.urdf"))

# Fix the base of the finger to the world
X_WI_f = RigidTransform.Identity()
plant_f.WeldFrames(plant_f.world_frame(), plant_f.GetFrameByName("base_link"), X_WI_f)
plant_f.Finalize()

#
# create autodiff plant corresponding to plant_f to use the autodiff scalar type, with a dynamic-sized vector of partial derivatives: plant_ad_f
plant_ad_f = plant_f.ToAutoDiffXd()
context_ad_f = plant_ad_f.CreateDefaultContext()

# remember that AddMultibodyPlantSceneGraph actually adds both plant_f and scene_graph, and wires them together. So we need to build the diagram
diagram_f = builder_f.Build()
diagram_context = diagram_f.CreateDefaultContext()

# define the whole diagram's context, returns a mutable reference to the subcontext that corresponds to the contained System subsystem.
context_f = diagram_f.GetMutableSubsystemContext(plant_f, diagram_context)

##################################### plant_franka, full model  ###############################################

# initialize the diagram builder: builder_franka
builder_franka = DiagramBuilder()
sim_dt = 1e-4
sample_dt = 1e-2

# Adds both MultibodyPlant: plant_franka and the SceneGraph: scene_graph, and wires them together.
# this plant_franka mainly is used contain the whole franka instead of simple end-effector
plant_franka, scene_graph_franka = AddMultibodyPlantSceneGraph(builder_franka, sim_dt)

# The same as above, package addition
parser_franka = Parser(plant_franka)
parser_franka.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/franka_trajectory_following/robot_properties_fingers/urdf/franka_box.urdf"))
parser_franka.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/franka_trajectory_following/robot_properties_fingers/urdf/sphere.urdf"))

# Fix the base of the franka robot to the world
X_WI_franka = RigidTransform.Identity()
plant_franka.WeldFrames(plant_franka.world_frame(), plant_franka.GetFrameByName("panda_link0"), X_WI_franka)
plant_franka.Finalize()

# remember that AddMultibodyPlantSceneGraph actually adds both plant_f and scene_graph, and wires them together.
# We need to build the diagram for the later use to get generalized acceleration
diagram_franka = builder_franka.Build()
diagram_context_franka = diagram_franka.CreateDefaultContext()
context_franka = diagram_franka.GetMutableSubsystemContext(plant_franka, diagram_context_franka)

##################################### all plants definition ends ###############################################

#### Start from here, doing FK simulation and get data #####
# This python script is used to get the robot-related properties from drake so that we can form the matrices for learning, mainly refer to lcs_factory_franka.cc

# load the data
logdir, log_num = get_most_recent_logs()
stateinput_file = "{}/{}/State_Input-{}.npz".format(logdir, log_num, log_num)
contact_file = "{}/{}/Contact_Info-{}.npz".format(logdir, log_num, log_num)

# data_stateinput: 'q': Franka joint angle; 'R_b': ball orientation quaternion; 'p_b': ball position;
# 'q_dot': Frank joint velocity; 'w_b': ball angular velocity; 'v_b': ball velocity; 'u': joint effort (torque)'
data_stateinput = np.load(stateinput_file)
data_stateinput_names = data_stateinput.files
# print(data_stateinput_names)

# data_contact: 'f_eeball': contact force between end effector and the ball, 'f_ballg': contact force between the ball and the ground
data_contact = np.load(contact_file)
data_contact_names = data_contact.files
# print(data_contact_names)

timestamp_state = data_stateinput['timestamp_state']

# create empty lists for saving the data
A_list = []
B_list = []
D_list = []
d_list = []
E_list = []
F_list = []
H_list = []
c_list = []
N_list = []
x_list = []
x_model_list = []

# sample dt is 0.01, so for sim_dt=1e-4, roughly sample every 100 points
# can be further improved to get the actually 0.01 interval using timestamp
for i in range(1,len(timestamp_state),100):
    # get position and velocities, in the future, fix the data logging in the future to get shorter and clearer codes
    position = []
    velocity = []
    position.append(data_stateinput['q'][:,i])
    position.append(data_stateinput['R_b'][:,i])
    position.append(data_stateinput['p_b'][:,i])
    position = np.concatenate(position)
    velocity.append(data_stateinput['q_dot'][:,i])
    velocity.append(data_stateinput['w_b'][:, i])
    velocity.append(data_stateinput['v_b'][:, i])
    velocity = np.concatenate(velocity)
    effort = data_stateinput['u'][:,i]

    # In order to get the LCS model, we need to know the position of the end-effector
    # So we need to update the franka context to do the FK
    plant_franka.SetPositions(context_franka,position)
    plant_franka.SetVelocities(context_franka,velocity)
    # write my own set input function (equivalent to SetInputsIfNew in c++)
    if (not plant_franka.get_actuation_input_port().HasValue(context_franka)) or (not (effort == plant_franka.get_actuation_input_port().Eval(context_franka)).all()):
        plant_franka.get_actuation_input_port().FixValue(context_franka, effort)

    # end effector offset
    ee_offset = np.array(param["EE_offset"])

    # get H_mat and get end effector position (p_ee) using ee_offset
    H_mat = plant_franka.EvalBodyPoseInWorld(context_franka, plant_franka.GetBodyByName("panda_link10"))
    R_current = H_mat.rotation()
    p_ee = H_mat.translation() + R_current @ ee_offset

    # get jacobian at end-effector to calculate velocity at v_ee
    ee_frame = plant_franka.GetBodyByName("panda_link10").body_frame()
    world_frame = plant_franka.world_frame()
    J_wee = plant_franka.CalcJacobianSpatialVelocity(context_franka,JacobianWrtVariable.kV,ee_frame,ee_offset,world_frame,world_frame)
    # extract the franka part since the whole model also includes the ball
    J_franka = J_wee[:6,:7]
    v_ee = (J_franka @ velocity[0:7])[-3:]

    # calculate the end-effector velocity a_ee, first we need to get the generalized acceleration vdot and spatical acceleration bias Jdotv
    # extract the franka part since the whole model also includes the ball
    vdot = plant_franka.get_generalized_acceleration_output_port().Eval(context_franka)
    vdot_franka = vdot[0:7]
    Jdotv_ee = plant_franka.CalcBiasSpatialAcceleration(context_franka,JacobianWrtVariable.kV,ee_frame,ee_offset,world_frame,world_frame)
    a_ee = (J_franka @ vdot_franka)[-3:] + Jdotv_ee.translational()

    # get ball pose and (angular) velocity
    q_b = position[-7:]
    v_b = velocity[-6:]

    # assemble end effector info and ball info to form the configuration q and velocity v (state) for the simplified model, also, initialize the input u to be 0
    q = np.concatenate([p_ee, q_b])
    v = np.concatenate([v_ee, v_b])
    state = np.concatenate([q,v])
    # mass of the end-effector, originally is 0.01, should be tuned.
    m_ee = 0.01
    # the input force should be u = F = m_ee*a_ee
    u = a_ee * m_ee
    xu = np.concatenate([state,u])

    # set the state and input for the simplified model: xu to autodiff type and set simplified model's context
    # note that for simplified model, only end-effector and ball, so num_position = 10 (3T for ee and 4R+3T for ball), num_velocity = 9 (3T for ee and 3R+3T for ball)
    xu_ad = InitializeAutoDiff(xu)
    plant_ad_f.SetPositionsAndVelocities(context_ad_f,xu_ad[0:plant_ad_f.num_positions()+plant_ad_f.num_velocities()])

    # write my own set input function (equivalent to SetInputsIfNew<AutoDiffXd>)
    if (not plant_ad_f.get_actuation_input_port().HasValue(context_ad_f)) or (not (xu_ad[-3:] == plant_ad_f.get_actuation_input_port().Eval(context_ad_f)).all()):
        plant_ad_f.get_actuation_input_port().FixValue(context_ad_f, xu_ad[-3:])

    # Also update the not Autodiff simplified model
    plant_f.SetPositions(context_f, q)
    plant_f.SetVelocities(context_f, v)
    if (not plant_f.get_actuation_input_port().HasValue(context_f)) or (not (u == plant_f.get_actuation_input_port().Eval(context_f)).all()):
        plant_f.get_actuation_input_port().FixValue(context_f, u)

    # set robot and contact geometries
    finger_lower_link_0_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("tip_link_1_real"))[0]
    sphere_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("sphere"))[0]
    ground_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("box"))[0]
    contact_geoms = [finger_lower_link_0_geoms, sphere_geoms, ground_geoms]
    # frictional contact parameters
    num_friction_directions = 2
    mu = float(param["mu"])

    # The return of the lcs_franka_new includes the system matrices and scaling (which is not used for now)
    System, Scaling = lcs_factory_franka_new.LinearizePlantToLCS(plant_f,context_f,plant_ad_f,context_ad_f,contact_geoms,num_friction_directions,mu,sample_dt)

    # record the system matrices
    A = System.A[0]
    B = System.B[0]
    D = System.D[0]
    d = System.d[0]
    E = System.E[0]
    F = System.F[0]
    H = System.H[0]
    c = System.c[0]

    A_list.append(A)
    B_list.append(B)
    D_list.append(D)
    d_list.append(d)
    E_list.append(E)
    F_list.append(F)
    H_list.append(H)
    c_list.append(c)

    # simulate the lcs and record the next state predicted by the model, called x_model
    x_model = System.Simulate(state,u)
    x_list.append(state)
    x_model_list.append(x_model)

x_model_all = np.array(x_model_list).T
x_all = np.array(x_list).T
# calculate residual, except for the head and end
residual = x_all[:,1:]-x_model_all[:,:-1]

# record the time of executing the code
end_time = time.time()
print("time used: {:.2f} s".format(end_time - start_time))

# save the data
print("creating matrices npz files")
mdic_contact ={"A_lcs":A_list,"B_lcs":B_list,"D_lcs":D_list,"d_lcs":d_list,"E_lcs":E_list,"F_lcs":F_list,"H_lcs":H_list,"c_lcs":c_list}
npz_file = "{}/{}/LCS_Matrices-{}.npz".format(logdir, log_num, log_num)
np.savez(npz_file, **mdic_contact)
print("creating state and residual npz files")
mdic_contact ={"state_plant":x_all[:,1:],"state_model":x_model_all[:,:-1],"residual":residual}
npz_file = "{}/{}/State_Residual-{}.npz".format(logdir, log_num, log_num)
np.savez(npz_file, **mdic_contact)


# import matplotlib.pyplot as plt
# # briefly check the result
# for i in range(residual.shape[0]):
#     plt.figure
#     plt.plot(residual[i,:])
#     plt.show()

    # # for single step validation before doing lcs add this block in the loop
    # if i == 5:
    #     # # validation print out the body names
    #     # link0 = plant_franka.GetBodyByName('panda_link10')
    #     # print(link0)
    #     # ball = plant_franka.GetBodyByName('sphere')
    #     # print(ball)
    #     # table = plant_franka.GetBodyByName('box')
    #     # print(table)
    #
    #     # # validate position and velocity
    #     # print(position)
    #     # print(velocity)
    #
    #     # # validate updating context
    #     # print(plant_franka.GetPositions(context_franka))
    #     # print(plant_franka.GetVelocities(context_franka))
    #
    #     # # validate ee offset derivation
    #     # print(ee_offset.shape)
    #
    #     # # validate H_mat
    #     # print(H_mat)
    #     # print(R_current)
    #     # print(ee_offset)
    #
    #     # # validate Jacobian
    #     # print(J_wee.shape)
    #     # print(J_franka.shape)
    #     # print(v_ee.shape)
    #
    #     # # validate configuration
    #     # print(q)
    #     # print(v)
    #     # print(state)
    #     # print(u)
    #     # print(xu)
    #
    #     # # validate autodiff
    #     # print(xu_ad)
    #     # print(plant_ad_f.num_positions())
    #     # print(plant_ad_f.num_velocities())
    #     # print(plant_ad_f.GetPositions(context_ad_f))
    #     # print(plant_ad_f.GetVelocities(context_ad_f))
    #
    #     # # validate simplified model plant
    #     # print(plant_f.num_positions())
    #     # print(plant_f.num_velocities())
    #     # print(plant_f.num_actuators())
    #     # print(q)
    #     # print(v)
    #     # print(plant_f.GetPositions(context_f))
    #     # print(plant_f.GetVelocities(context_f))
    #     print("plant check pass")
    #     break