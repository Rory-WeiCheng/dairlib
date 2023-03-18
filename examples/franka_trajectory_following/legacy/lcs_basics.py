from pyexpat.model import XML_CQUANT_NONE
from dairlib import (lcmt_robot_output, lcmt_robot_input, lcmt_c3)
from pydrake.common.yaml import yaml_load
import pydairlib.common
import pydairlib.lcm
from pydairlib.systems import (RobotC3Receiver, 
                               RobotC3Sender,
                               RobotOutputReceiver, RobotOutputSender,
                               LcmOutputDrivenLoop, OutputVector,
                               TimestampedVector,
                               AddActuationRecieverAndStateSenderLcm)
from pydrake.all import (AbstractValue, DiagramBuilder, DrakeLcm, LeafSystem,
                         MultibodyPlant, Parser, RigidTransform, Subscriber,
                         LcmPublisherSystem, TriggerType, AddMultibodyPlantSceneGraph,
                         LcmInterfaceSystem)
#import pydairlib.common
from pydairlib.multibody import (addFlatTerrain, makeNameToPositionsMap, makeNameToVelocitiesMap)

# newly added
from pydrake.multibody.tree import JacobianWrtVariable,MultibodyForces_
from pydrake.autodiffutils import InitializeAutoDiff,AutoDiffXd,ExtractGradient,ExtractValue
from scipy.linalg import lu_factor, lu_solve

import pydairlib.common
from pydairlib.systems.controllers_franka import C3Controller_franka
import numpy as np
from pydrake.trajectories import PiecewisePolynomial
import math
from examples.franka_trajectory_following.scripts.franka_logging_utils_test import get_most_recent_logs

# load parameters
param = yaml_load(
    filename="examples/franka_trajectory_following/parameters.yaml")

# initialize drake lcm
lcm = DrakeLcm()

##################################### basic plant ###################################################
# initialize Multibody Plant: plant
plant = MultibodyPlant(0.0)

# The package addition here seems necessary due to how the URDF is defined
# Parser the robot file into plant: parser
parser = Parser(plant)
parser.package_map().Add("robot_properties_fingers",
                         "examples/franka_trajectory_following/robot_properties_fingers")
# load the end effector (finger only) urdf
parser.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/franka_trajectory_following/robot_properties_fingers/urdf/trifinger_minimal_collision_2.urdf"))
# load the ball urdf
parser.AddModelFromFile(pydairlib.common.FindResourceOrThrow(
    "examples/franka_trajectory_following/robot_properties_fingers/urdf/sphere.urdf"))

# Fix the base of the finger to the world and finalize the plant
X_WI = RigidTransform.Identity()
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base_link"), X_WI)
plant.Finalize()

# initialize the diagram builder: builder
builder = DiagramBuilder()

####################################### scene(?) plant ###########################################
# initialize the diagram builder: builder_f
builder_f = DiagramBuilder()
sim_dt = 1e-4

# Adds both MultibodyPlant: plant_f and the SceneGraph: scene_graph, and wires them together.
# this plant_f mainly is used to do scene graph
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

##################################### fanka(?) plant ###############################################

# initialize the diagram builder: builder_franka
builder_franka = DiagramBuilder()
sim_dt = 1e-4
output_dt = 1e-4

# Adds both MultibodyPlant: plant_franka and the SceneGraph: scene_graph, and wires them together.
# this plant_f mainly is used contain the whole franka instead of simple end-effector
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

context_franka = plant_franka.CreateDefaultContext()

##################################### all plants definition ends ###############################################

#### Start from here #####
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
A_list = []
B_list = []
for i in range(len(timestamp_state)):
    # get position and velocities, in the future, fix the data logging to get shorter codes
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
    J_franka = J_wee[:6,:7]
    v_ee = (J_franka @ velocity[0:7])[-3:]

    # get ball pose and (angular) velocity + position and translation velocity
    q_b = position[-7:]
    v_b = velocity[-6:]
    p_b = position[-3:]
    v_bT = position[-3:]

    # assemble end effector info and ball info to form the configuration q and velocity v needed for lcs, also, initialize the input u to be 0
    q = np.concatenate([p_ee, q_b])
    v = np.concatenate([v_ee, v_b])
    state = np.concatenate([q,v])
    u = np.zeros(3)
    xu = np.concatenate([state,u])

    # set the state and input: xu to autodiff type and set simple model's context
    # note that for simple model, only end-effector and ball, so num_position = 10 (3T for ee and 4R+3T for ball), num_velocity = 9 (3T for ee and 3R+3T for ball)
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


    '''
    ## old methhod, now switch to using the c++ bindings
    ## Start doing LCS
    dt = 0.1
    # extract the bias term C(q,v)*v
    Cv = plant_ad_f.CalcBiasTerm(context_ad_f)
    # derive the input term B*u
    Bu = plant_ad_f.MakeActuationMatrix() @ plant_ad_f.get_actuation_input_port().Eval(context_ad_f)

    # number of states, n_pos = n_state in c++, n_state = n_total in c++
    n_pos = plant_ad_f.num_positions()
    n_vel = plant_ad_f.num_velocities()
    n_state = n_pos + n_vel
    n_input = plant_ad_f.num_actuators()

    # derive the gravity generalized force
    tau_g = plant_ad_f.CalcGravityGeneralizedForces(context_ad_f)
    # derive rigid body forces and mass matrix
    f_app = MultibodyForces_(plant_ad_f)
    plant_ad_f.CalcForceElementsContribution(context_ad_f, f_app)
    M = plant_ad_f.CalcMassMatrix(context_ad_f)
    # calculate vdot_no_contact solve M*vdot_no_contact = tau_g + f_app.generalized_forces() + Bu - Cv, simply use 
    # pydrake.autodiffutils.inv since scipy pakages LU solver can't deal with autodiff arguments (c++ code originally use ldlT solver)
    vdot_no_contact = AutoDiffXd.inv(M) @ (tau_g + f_app.generalized_forces() + Bu - Cv)
    # calculate qdot_no_contact using velocity to qdot
    vel = plant_ad_f.get_state_output_port().Eval(context_ad_f)[-n_vel:]
    qdot_no_contact = plant_ad_f.MapVelocityToQDot(context_ad_f,vel)
    d_q = ExtractValue(qdot_no_contact)

    ## get A,B matrices
    # initialization
    A = np.zeros((n_state,n_state))
    B = np.zeros((n_state,n_input))
    # intermediate blocks AB_q and AB_v
    AB_q = ExtractGradient(qdot_no_contact)
    Nq = AB_q[0:n_pos,n_pos:n_state]
    AB_v = ExtractGradient(vdot_no_contact)
    AB_v_q = AB_v[0:n_vel,0:n_pos ]
    AB_v_v = AB_v[0:n_vel,n_pos:n_state]
    AB_v_u = AB_v[0:n_vel,n_state:n_state+n_input]

    # Derive A,B
    A[0:n_pos, 0:n_pos] = np.eye(n_pos) + dt * dt * Nq @ AB_v_q
    A[0:n_pos, n_pos:n_state] = dt * Nq + dt * dt * Nq @ AB_v_v
    A[n_pos:n_state, 0:n_pos] = dt * AB_v_q
    A[n_pos:n_state, n_pos:n_state] = dt * AB_v_v + np.eye(n_vel)

    B[0:n_pos, 0:n_input] = dt * dt * Nq @ AB_v_u
    B[n_pos:n_state, 0:n_input] = dt * AB_v_u

    A_list.append(A)
    B_list.append(B)
    '''

# print(A_list.shape)
# print(B_list.shape)
print("creating matrices npz files")
mdic_contact ={"A_lcs":A,"B_lcs":B}
npz_file = "{}/{}/Matrices-{}.npz".format(logdir, log_num, log_num)
np.savez(npz_file, **mdic_contact)

    # # for single step validation before doing lcs
    # if i == 2:
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
    #
    # # for single step validation when doing lcs
    # if i == 2:
    #     ## check basic variables dimension
    #     # print(contact_geoms)
    #     # print(Cv)
    #     # print(Bu)
    #     # print(tau_g)
    #     # print(vel)
    #     # print(f_app)
    #     # print(M)
    #     # print(vdot_no_contact)
    #     # print(qdot_no_contact)
    #
    #     ## check matrix dimension
    #     # print(AB_q.shape)
    #     # print(AB_v_u.shape)
    #     # print(Nq.shape)
    #     break
    #
    # if i == 5:
    #     break


'''
# define constants, from plants (simple model, only ee and ball)
# nq = 10 (ee: 3T, ball: 4R+3T), nv = 9 (ee: 3T, ball: 3R+3T), nu = 3 (ee acceleration)
nq = plant.num_positions()
nv = plant.num_velocities()
nu = plant.num_actuators()
nc = 2      # number of contacts

# create name to position map, for simple model
q = np.zeros((nq,1))
q_map = makeNameToPositionsMap(plant)
v_map = makeNameToVelocitiesMap(plant)

# set finger initial configuration
finger_init = param["q_init_finger"]
q[0] = finger_init[0]
q[1] = finger_init[1]
q[2] = finger_init[2]

# set ball initial condition
ball_init = param["q_init_ball_c3"]
q[q_map['base_qw']] = ball_init[0]
q[q_map['base_qx']] = ball_init[1]
q[q_map['base_qy']] = ball_init[2]
q[q_map['base_qz']] = ball_init[3]
q[q_map['base_x']] = ball_init[4]
q[q_map['base_y']] = ball_init[5]
q[q_map['base_z']] = ball_init[6]
mu = param["mu"]
'''
'''
# set weight matrix initial condition
Qinit = param["Q_default"] * np.eye(nq+nv)
Qinit[0,0] = param["Q_finger"]
Qinit[1,1] = param["Q_finger"]
Qinit[2,2] = param["Q_finger"]
Qinit[7,7] = param["Q_ball_x"]
Qinit[8,8] = param["Q_ball_y"]
Qinit[10:10+nv,10:10+nv] = param["Q_ball_vel"] * np.eye(nv)
Qinit[10:13,10:13] = param["Q_finger_vel"]*np.eye(3) #10
# Qinit[13:16, 13:16] = 0*np.eye(3) # zero out cost on rotational velocity
Rinit = param["R"] * np.eye(nu) #torques


# set ADMM parameters
Ginit = param["G"] * np.eye(nq+nv+nu+6*nc)
Uinit = param["U_default"] * np.eye(nq+nv+nu+6*nc)
Uinit[0:nq+nv,0:nq+nv] = param["U_pos_vel"] * np.eye(nq+nv) 
Uinit[nq+nv+6*nc:nq+nv+nu+6*nc, nq+nv+6*nc:nq+nv+nu+6*nc] = param["U_u"] * np.eye(nu)

# set desired state initial condition
xdesiredinit = np.zeros((nq+nv,1))
xdesiredinit[:nq] = q

# set desired trak=jectory parameters
r = param["traj_radius"]
x_c = param["x_c"]
y_c = param["y_c"]

# set rolling related parameters
degree_increment = param["degree_increment"]
if param["hold_order"] == 0:
    theta = np.arange(degree_increment, 400 + degree_increment, degree_increment)
elif param["hold_order"] == 1:
    theta = np.arange(0, 400, degree_increment)
xtraj = []
for i in theta:
    x = r * np.sin(math.radians(i+param["phase"]))
    y = r * np.cos(math.radians(i+param["phase"]))
    # x = r * np.sin(math.radians(i+param["phase"]+90)) * np.cos(math.radians(i+param["phase"]+90))
    # y = r * np.sin(math.radians(i+param["phase"]+90))
    q[q_map['base_x']] = x + x_c
    q[q_map['base_y']] = y + y_c
    q[q_map['base_z']] = param["ball_radius"] + param["table_offset"]
    xtraj_hold = np.zeros((nq+nv,1))
    xtraj_hold[:nq] = q
    xtraj.append(xtraj_hold)

time_increment = 1.0*param["time_increment"]  # also try just moving in a straight line maybe
delay = param["stabilize_time1"] + param["move_time"] + param["stabilize_time2"]
timings = np.arange(delay, time_increment*len(xtraj) + delay, time_increment)

if param["hold_order"] == 0:
    pp = PiecewisePolynomial.ZeroOrderHold(timings, xtraj)
elif param["hold_order"] == 1:
    pp = PiecewisePolynomial.FirstOrderHold(timings, xtraj)


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

#Qinit[nv-3:nv,nv-3:nv] = 1*np.eye(3) #penalize final velocities
Q.append(Qinit)
xdesired.append(xdesiredinit)
'''
'''
# set robot and contact geometries
finger_lower_link_0_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("tip_link_1_real"))[0]
sphere_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("sphere"))[0]
ground_geoms = plant_f.GetCollisionGeometriesForBody(plant_f.GetBodyByName("box"))[0]

contact_geoms = [finger_lower_link_0_geoms, sphere_geoms, ground_geoms] #finger_lower_link_120_geoms, finger_lower_link_240_geoms,

# create autodiff plant corresponding to plant to use the autodiff scalar type, with a dynamic-sized vector of partial derivatives: plant_ad
plant_ad = plant.ToAutoDiffXd()
context = plant.CreateDefaultContext()
context_ad = plant_ad.CreateDefaultContext()

# Given: every timestamps' x(q,v), u, find the corresponding

controller = builder.AddSystem(
    C3Controller_franka(plant, plant_f, plant_franka, context, context_f, context_franka, plant_ad, plant_ad_f, context_ad, context_ad_f, scene_graph, diagram_f, contact_geoms, num_friction_directions, mu, Q, R, G, U, xdesired, pp))
'''

'''
# create state message receiver (check corresponding lcm type): from frank plant (full model)
state_receiver = builder.AddSystem(RobotOutputReceiver(plant_franka))

# create c3 controller (planning using simple model)
controller = builder.AddSystem(
    C3Controller_franka(plant, plant_f, plant_franka, context, context_f, context_franka, plant_ad, plant_ad_f, context_ad, context_ad_f, scene_graph, diagram_f, contact_geoms, num_friction_directions, mu, Q, R, G, U, xdesired, pp))

# create control command sender (planning using simple model)
state_force_sender = builder.AddSystem(RobotC3Sender(10, 9, 6, 9))

# build the diagram: this builder is to connect all the subsystem, while builder_f build the plant_f and scene graph and builder_franka build the plant_franka and scenegraph_franka

# controller receive the state from the franka robot
builder.Connect(state_receiver.get_output_port(0), controller.get_input_port(0))

# controller send control command to the state force sender
builder.Connect(controller.get_output_port(), state_force_sender.get_input_port(0))

# create control publisher, receive message from state force sender and publish it out
control_publisher = builder.AddSystem(LcmPublisherSystem.Make(
    channel="CONTROLLER_INPUT", lcm_type=lcmt_c3, lcm=lcm,
    publish_triggers={TriggerType.kForced},
    publish_period=0.0, use_cpp_serializer=True))
builder.Connect(state_force_sender.get_output_port(),
    control_publisher.get_input_port())

# finish building the diagram
diagram = builder.Build()

# the context for the whole diagram
context_d = diagram.CreateDefaultContext()

# the context of the state receiver
receiver_context = diagram.GetMutableSubsystemContext(state_receiver, context_d)

# loop driven by lcm message
loop = LcmOutputDrivenLoop(drake_lcm=lcm, diagram=diagram,
                          lcm_parser=state_receiver,
                          input_channel="FRANKA_OUTPUT",
                          is_forced_publish=True)

# simulate for 200s
loop.Simulate(200)
'''