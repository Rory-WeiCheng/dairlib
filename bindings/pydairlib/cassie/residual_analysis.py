import sys
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pydairlib
import scipy.linalg
import scipy.io
import cvxpy as cp
from scipy.spatial.transform import Rotation
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydairlib.cassie.cassie_utils import *
from pydairlib.multibody import kinematic
from pydrake.multibody.tree import JacobianWrtVariable, MultibodyForces
from pydairlib.multibody import multibody

class CassieSystem():
    def __init__(self):

        self.urdf = "examples/Cassie/urdf/cassie_v2.urdf"

        # Initialize for drake
        self.builder = DiagramBuilder()
        self.drake_sim_dt = 5e-5
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, self.drake_sim_dt)
        addCassieMultibody(self.plant, self.scene_graph, True, 
                            self.urdf, True, True)
        self.plant.Finalize()
        self.world = self.plant.world_frame()
        self.base_frame = self.plant.GetBodyByName("pelvis").body_frame()
        self.context = self.plant.CreateDefaultContext()
        
        left_rod_on_thigh = LeftRodOnThigh(self.plant)
        left_rod_on_heel = LeftRodOnHeel(self.plant)
        self.left_loop = kinematic.DistanceEvaluator(
                        self.plant, left_rod_on_heel[0], left_rod_on_heel[1], left_rod_on_thigh[0],
                        left_rod_on_thigh[1], 0.5012)
        right_rod_on_thigh = RightRodOnThigh(self.plant)
        right_rod_on_heel = RightRodOnHeel(self.plant)
        self.right_loop = kinematic.DistanceEvaluator(
                        self.plant, right_rod_on_heel[0], right_rod_on_heel[1], right_rod_on_thigh[0],
                        right_rod_on_thigh[1], 0.5012)
        # Initial other parameters
        self.K = np.zeros((22,23))
        self.C = np.zeros((22,22))

        # Visualizer
        self.visualizer = multibody.MultiposeVisualizer(self.urdf, 1, '')

        self.init_intermediate_variable_for_plot()

    def init_intermediate_variable_for_plot(self):
        self.r = []
        self.effect_e_J_c = []
        self.effect_e_J_h = []
        self.r_spring = []
        self.r_damping = []

    def drawPose(self, q):
        self.visualizer.DrawPoses(q)

    def get_spring_stiffness(self, K):
        self.K = K
    
    def get_damping(self, C):
        self.C = C

    def calc_vdot(self, t, q, v, u, is_contact=None, lambda_c_gt=None, lambda_c_position=None, v_dot_gt=None, is_soft_constraints=False):
        """
            The unknown term are \dot v, contact force \lambda_c_active and constraints forces \lambda_h.

            M * \dot v + bias + Cv + Kq == Bu + gravity + J_c_active^T*\lambda_c + J_h^T*\lambda_h     (1)
            \dot J_h * v + J_h * \dot v == 0    (2)
            \dot J_c_active * v + J_c_active * \dot v == 0 (3)

            ==>

            [M, -J_c_active^T, -J_h^T      [ \dot v             [ Bu + gravity - bias - Cv -Kq
            J_h, 0, 0           *     \lambda_c     =      -\dot J_h * v
            J_c_active, 0, 0          \lambda_h            -\dot J_c_active * v 
            ]                       ]                     ]
        """
        # Set configuration
        self.plant.SetPositionsAndVelocities(self.context, np.hstack((q,v)))

        # Get mass matrix
        M = self.plant.CalcMassMatrix(self.context)

        # Get bias term
        bias = self.plant.CalcBiasTerm(self.context)

        # Get the B matrix
        B = self.plant.MakeActuationMatrix()

        # Get the gravity term

        gravity = self.plant.CalcGravityGeneralizedForces(self.context)

        # Get the general damping and spring force 
        damping_and_spring_force = MultibodyForces(self.plant)
        self.plant.CalcForceElementsContribution(self.context, damping_and_spring_force)
        damping_and_spring_force = damping_and_spring_force.generalized_forces()

        # Get the J_h term
        J_h = np.zeros((2, 22))
        J_h[0, :] = self.left_loop.EvalFullJacobian(self.context)
        J_h[1, :] = self.right_loop.EvalFullJacobian(self.context)

        # Get the JdotTimesV term
        J_h_dot_times_v = np.zeros(2,)
        J_h_dot_times_v[0] = self.left_loop.EvalFullJacobianDotTimesV(self.context)
        J_h_dot_times_v[1] = self.right_loop.EvalFullJacobianDotTimesV(self.context)

        # Get the J_c_active term
        # For simplicity test, now directly give the lambda_c ground truth from simulation to determine if contact happen
        J_c_active, J_c_active_dot_v, num_contact_unknown, get_force_at_point_matrix = self.get_J_c_ative_and_J_c_active_dot_v(is_contact=is_contact, lambda_c_gt=lambda_c_gt, lambda_c_position=lambda_c_position)
        z_direction_selection_matrix = np.zeros((4,12))
        z_direction_selection_matrix[0,2] = 1;  z_direction_selection_matrix[1,5] = 1
        z_direction_selection_matrix[2,8] = 1;  z_direction_selection_matrix[3,11] = 1

        # Construct Linear Equation Matrices
        # Solving the exact linear equations:
        if num_contact_unknown > 0:
            A = np.vstack((
                    np.hstack((M, -J_c_active.T, -J_h.T)),
                    np.hstack((J_h, np.zeros((2,num_contact_unknown)),np.zeros((2, 2)))),
                    np.hstack((J_c_active, np.zeros((num_contact_unknown, num_contact_unknown)), np.zeros((num_contact_unknown,2))))
            ))

            b = np.hstack((
                B @ u + gravity - bias + damping_and_spring_force,
                -J_h_dot_times_v,
                -J_c_active_dot_v
            ))
        else:
            A = np.vstack((
                np.hstack((M, -J_h.T)),
                np.hstack((J_h, np.zeros((2, 2)))),
                ))

            b = np.hstack((
                B @ u + gravity - bias + damping_and_spring_force,
                -J_h_dot_times_v,
            ))

        if not is_soft_constraints or num_contact_unknown == 0:
            # Solve minimum norm solution, since J_c may be not full row rank
            solution = A.T @ np.linalg.inv(A @ A.T) @ b
        else:
            solution = cp.Variable(22+num_contact_unknown+2)
            epislon = cp.Variable(22+num_contact_unknown+2)
            obj = cp.Minimize(cp.norm(epislon))
            constraints = [A @ solution == b + epislon]
            constraints.append(z_direction_selection_matrix @ get_force_at_point_matrix @ solution[22:22+num_contact_unknown] >= 0)
            prob = cp.Problem(obj, constraints)
            prob.solve()
            solution = solution.value

        v_dot = solution[:22]
        if num_contact_unknown > 0:
            lambda_c = get_force_at_point_matrix @ solution[22:22+num_contact_unknown]
        else:
            lambda_c = np.zeros(6,)
        lambda_h = solution[-2:]
        
        v_map, v_map_inverse = self.make_velocity_map()
        r = (v_dot - v_dot_gt)
        self.r.append(r)

        self.r_spring.append((A.T @ np.linalg.inv(A @ A.T) @ np.hstack((-self.K @ q, np.zeros(2 + num_contact_unknown))))[:22])
        self.r_damping.append((A.T @ np.linalg.inv(A @ A.T) @ np.hstack((-self.C @ v, np.zeros(2 + num_contact_unknown))))[:22])

        if num_contact_unknown > 0:
            e_J_c = J_c_active @ v_dot_gt + J_c_active_dot_v
            effect_e_J_c = (np.linalg.inv(M) @ J_c_active.T @ np.linalg.inv(J_c_active @ np.linalg.inv(M) @ J_c_active.T) @ e_J_c)
            self.effect_e_J_c.append(effect_e_J_c)
        else:
            self.effect_e_J_c.append(np.zeros(22))

        e_J_h = J_h @ v_dot_gt + J_h_dot_times_v
        effect_e_J_h = (np.linalg.inv(M) @ J_h.T @ np.linalg.inv( J_h @ np.linalg.inv(M) @ J_h.T) @ e_J_h)
        self.effect_e_J_h.append(effect_e_J_h)

        # if abs(r) > 150:
        #     print("residual:", r)
        #     if num_contact_unknown > 0:
        #         print("residual of J_c:", e_J_c)
        #         print("effect of residual of J_c", effect_e_J_c)
        #     print("residual pf J_h", e_J_h)
        #     print("effect of residual of J_h", effect_e_J_h)

        return v_dot, lambda_c, lambda_h

    def get_J_c_ative_and_J_c_active_dot_v(self, is_contact, lambda_c_gt, lambda_c_position):
        J_c_active = None; J_c_active_dot_v = None; num_contact_unknown = 0
        left_front_index = None; left_rear_index = None
        right_front_index = None; right_rear_index = None
        left_index = None; right_index = None
        if lambda_c_gt is not None:
            if np.linalg.norm(lambda_c_gt[:3]) > 0:
                point_on_foot, foot_frame = LeftToeFront(self.plant)
                if lambda_c_position is not None:
                    point_on_foot_on_world = lambda_c_position[:3]
                    point_on_foot = self.plant.CalcPointsPositions(self.context, self.world, point_on_foot_on_world, foot_frame)
                J_c_left_front = self.plant.CalcJacobianTranslationalVelocity(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world)
                J_c_left_front_dot_v = self.plant.CalcBiasTranslationalAcceleration(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world).squeeze()
                J_c_active = J_c_left_front
                J_c_active_dot_v = J_c_left_front_dot_v
                left_front_index = num_contact_unknown
                num_contact_unknown += 3
            if np.linalg.norm(lambda_c_gt[3:6]) > 0:
                point_on_foot, foot_frame = LeftToeRear(self.plant)
                if lambda_c_position is not None:
                    point_on_foot_on_world = lambda_c_position[3:6]
                    point_on_foot = self.plant.CalcPointsPositions(self.context, self.world, point_on_foot_on_world, foot_frame)
                J_c_left_rear = self.plant.CalcJacobianTranslationalVelocity(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world)
                J_c_left_rear_dot_v = self.plant.CalcBiasTranslationalAcceleration(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world).squeeze()
                if J_c_active is None:
                    J_c_active = J_c_left_rear
                    J_c_active_dot_v = J_c_left_rear_dot_v
                else:
                    J_c_active = np.vstack((J_c_active, J_c_left_rear))
                    J_c_active_dot_v = np.hstack((J_c_active_dot_v, J_c_left_rear_dot_v))
                left_rear_index = num_contact_unknown
                num_contact_unknown += 3
            if np.linalg.norm(lambda_c_gt[6:9]) > 0:
                point_on_foot, foot_frame = RightToeFront(self.plant)
                if lambda_c_position is not None:
                    point_on_foot_on_world = lambda_c_position[6:9]
                    point_on_foot = self.plant.CalcPointsPositions(self.context, self.world, point_on_foot_on_world, foot_frame)
                J_c_right_front = self.plant.CalcJacobianTranslationalVelocity(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world)
                J_c_right_front_dot_v = self.plant.CalcBiasTranslationalAcceleration(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world).squeeze()
                if J_c_active is None:
                    J_c_active = J_c_right_front
                    J_c_active_dot_v = J_c_right_front_dot_v
                else:
                    J_c_active = np.vstack((J_c_active, J_c_right_front))
                    J_c_active_dot_v = np.hstack((J_c_active_dot_v, J_c_right_front_dot_v))
                right_front_index = num_contact_unknown
                num_contact_unknown += 3
            if np.linalg.norm(lambda_c_gt[9:12]) > 0:
                point_on_foot, foot_frame = RightToeRear(self.plant)
                if lambda_c_position is not None:
                    point_on_foot_on_world = lambda_c_position[9:12]
                    point_on_foot = self.plant.CalcPointsPositions(self.context, self.world, point_on_foot_on_world, foot_frame)
                J_c_right_rear = self.plant.CalcJacobianTranslationalVelocity(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world)
                J_c_right_rear_dot_v = self.plant.CalcBiasTranslationalAcceleration(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world).squeeze()
                if J_c_active is None:
                    J_c_active = J_c_right_rear
                    J_c_active_dot_v = J_c_right_rear_dot_v
                else:
                    J_c_active = np.vstack((J_c_active, J_c_right_rear))
                    J_c_active_dot_v = np.hstack((J_c_active_dot_v, J_c_right_rear_dot_v))
                right_rear_index = num_contact_unknown
                num_contact_unknown += 3

            get_force_at_point_matrix = np.zeros((12, num_contact_unknown))
            if left_front_index is not None:
                get_force_at_point_matrix[0,left_front_index] = 1
                get_force_at_point_matrix[1,left_front_index+1] = 1
                get_force_at_point_matrix[2,left_front_index+2] = 1
            if left_rear_index is not None:
                get_force_at_point_matrix[3,left_rear_index] = 1
                get_force_at_point_matrix[4,left_rear_index+1] = 1
                get_force_at_point_matrix[5,left_rear_index+2] = 1
            if right_front_index is not None:
                get_force_at_point_matrix[6,right_front_index] = 1
                get_force_at_point_matrix[7,right_front_index+1] = 1
                get_force_at_point_matrix[8,right_front_index+2] = 1
            if right_rear_index is not None:
                get_force_at_point_matrix[9,right_rear_index] = 1
                get_force_at_point_matrix[10,right_rear_index+1] = 1
                get_force_at_point_matrix[11,right_rear_index+2] = 1
        else:
            if is_contact[0] > 0:
                toe_front, foot_frame = LeftToeFront(self.plant)
                toe_rear, foot_frame = LeftToeRear(self.plant)
                point_on_foot = (toe_front + toe_rear)/2
                J_c_left = self.plant.CalcJacobianTranslationalVelocity(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world)
                J_c_left_dot_v = self.plant.CalcBiasTranslationalAcceleration(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world).squeeze()
                J_c_active = J_c_left
                J_c_active_dot_v = J_c_left_dot_v
                left_index = num_contact_unknown
                num_contact_unknown += 3
            if is_contact[1] > 0:
                toe_front, foot_frame = RightToeFront(self.plant)
                toe_rear, foot_frame = RightToeRear(self.plant)
                point_on_foot = (toe_front + toe_rear)/2
                J_c_right = self.plant.CalcJacobianTranslationalVelocity(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world)
                J_c_right_dot_v = self.plant.CalcBiasTranslationalAcceleration(self.context, JacobianWrtVariable.kV, foot_frame, point_on_foot, self.world, self.world).squeeze()
                if J_c_active is None:
                    J_c_active = J_c_right
                    J_c_active_dot_v = J_c_right_dot_v    
                else:
                    J_c_active = np.vstack((J_c_active, J_c_right))
                    J_c_active_dot_v = np.hstack(J_c_active_dot_v, J_c_right_dot_v)

                right_index = num_contact_unknown
                num_contact_unknown += 3
            
            get_force_at_point_matrix = np.zeros((6, num_contact_unknown))
            if left_index is not None:
                get_force_at_point_matrix[0,left_index] = 1
                get_force_at_point_matrix[1,left_index+1] = 1
                get_force_at_point_matrix[2,left_index+2] = 1
            if right_index is not None:
                get_force_at_point_matrix[3, right_index] = 1
                get_force_at_point_matrix[4, right_index+1] = 1
                get_force_at_point_matrix[5, right_index+2] = 1

        return J_c_active, J_c_active_dot_v, num_contact_unknown, get_force_at_point_matrix      

    def make_position_map(self):

        pos_map = pydairlib.multibody.makeNameToPositionsMap(self.plant)
        pos_map_inverse = {value:key for (key, value) in pos_map.items()}
        
        return pos_map, pos_map_inverse
    
    def make_velocity_map(self):

        vel_map = pydairlib.multibody.makeNameToVelocitiesMap(self.plant)
        vel_map_inverse = {value:key for (key, value) in vel_map.items()}

        return vel_map, vel_map_inverse

    def make_actuator_map(self):

        act_map = pydairlib.multibody.makeNameToActuatorsMap(self.plant)
        act_map_inverse = {value:key for (key, value) in act_map.items()}

        return act_map, act_map_inverse

    def cal_damping_and_spring_force(self, q, v):
        self.plant.SetPositionsAndVelocities(self.context, np.hstack((q,v)))
        damping_and_spring_force = MultibodyForces(self.plant)
        self.plant.CalcForceElementsContribution(self.context, damping_and_spring_force)
        damping_and_spring_force = damping_and_spring_force.generalized_forces()

        return damping_and_spring_force

class CaaiseSystemTest():
    def __init__(self, data_path, start_time, end_time):
        
        self.data_path = data_path
        self.start_time = start_time
        self.end_time = end_time
        
        self.cassie = CassieSystem()
        
        self.pos_map, self.pos_map_inverse = self.cassie.make_position_map()
        self.vel_map, self.vel_map_inverse = self.cassie.make_velocity_map()
        self.act_map, self.act_map_inverse = self.cassie.make_actuator_map()
        self.lambda_c_map, self.lambda_c_map_inverse = self.make_lambda_c_map(is_four_point=False)

    def make_lambda_c_map(self, is_four_point=True):
        
        if is_four_point:
            lambda_c_map = {"left_front_x":0, "left_front_y":1, "left_front_z":2,
                            "left_rear_x":3, "left_rear_y":4, "left_rear_z":5,
                            "right_front_x":6, "right_front_y":7, "right_front_z":8,
                            "right_rear_x":9, "right_rear_y":10, "right_rear_z":11
                        }
        else:
            lambda_c_map = {"left_x":0, "left_y":1, "left_z":2,
                            "right_x":3, "right_y":4, "right_z":5,}
        lambda_c_map_inverse = {value:key for (key,value) in lambda_c_map.items()}
        
        return lambda_c_map, lambda_c_map_inverse

    def get_residual(self, vdot_gt, vdot_est, cutting_f = 100):
        residuals = vdot_gt - vdot_est
        residuals_smoothed = self.first_order_filter(residuals, cutting_f=cutting_f)
        return residuals_smoothed

    def first_order_filter(self, orginal_signals, cutting_f=100, Ts=0.0005):
        a = np.exp(-2*np.pi*cutting_f*Ts)
        filtered_signals = [orginal_signals[0]]
        for i in range(1, orginal_signals.shape[0]):
            filtered_signals.append(a * filtered_signals[-1] + (1-a) * orginal_signals[i])
        filtered_signals = np.array(filtered_signals)
        return filtered_signals 

    def hardware_test(self):
        raw_data = scipy.io.loadmat(self.data_path)
        process_data = self.process_hardware_data(raw_data, self.start_time, self.end_time)
        t = process_data["t"]; q = process_data["q"]; v = process_data["v"]; v_dot_gt = process_data["v_dot"]; 
        u = process_data["u"]; is_contact = process_data["is_contact"]; v_dot_osc = process_data["v_dot_osc"]; u_osc = process_data["u_osc"]
        K = process_data["spring_stiffness"]; C = process_data["damping_ratio"]
        self.cassie.get_spring_stiffness(K); self.cassie.get_damping(C)

        t_list = []
        v_dot_est_list = []
        lambda_c_est_list = []
        lambda_h_est_list = []
        v_dot_gt_list = []

        for i in range(0, t.shape[0], ):
            t_list.append(t[i])
            v_dot_est, lambda_c_est, lambda_h_est= self.cassie.calc_vdot(t[i], q[i,:], v[i,:], u[i,:], is_contact=is_contact[i,:], v_dot_gt=v_dot_gt[i,:])
            v_dot_est_list.append(v_dot_est)
            lambda_c_est_list.append(lambda_c_est)
            lambda_h_est_list.append(lambda_h_est)
            v_dot_gt_list.append(v_dot_gt[i,:])
        
        t_list = np.array(t_list)
        v_dot_est_list = np.array(v_dot_est_list)
        lambda_c_est_list = np.array(lambda_c_est_list)
        lambda_h_est_list = np.array(lambda_h_est_list)
        v_dot_gt_list = np.array(v_dot_gt_list)

        self.cassie.r = np.array(self.cassie.r)
        self.cassie.effect_e_J_c = np.array(self.cassie.effect_e_J_c)
        self.cassie.effect_e_J_h = np.array(self.cassie.effect_e_J_h)
        self.cassie.r_spring = np.array(self.cassie.r_spring)
        self.cassie.r_damping = np.array(self.cassie.r_damping)

        residuals = self.get_residual(v_dot_gt_list, v_dot_est_list)
        self.draw_residual(q[:,self.pos_map["knee_joint_left"]], residuals, "bindings/pydairlib/cassie/residual_analysis/hardware_test/residual/residual_vs_knee_joint_left")

        for i in range(22):
            plt.cla()
            plt.plot(t_list, self.cassie.r[:,i], label="total residual")
            plt.plot(t_list, self.cassie.effect_e_J_c[:,i], label="effect of violate J_c")
            plt.plot(t_list, self.cassie.effect_e_J_h[:,i], label="effect of violate J_h")
            plt.plot(t_list, self.cassie.r_spring[:,i], label="residual due to spring force")
            plt.plot(t_list, self.cassie.r_damping[:,i], label="residual due to damping force")
            plt.legend()
            plt.title(self.vel_map_inverse[i])
            plt.savefig("bindings/pydairlib/cassie/residual_analysis/hardware_test/v_dot_residual/{}.png".format(self.vel_map_inverse[i]))
        
        for i in range(22):
            plt.cla()
            plt.plot(t_list, v_dot_est_list[:,i], 'r', label='est')
            plt.plot(t_list, v_dot_gt_list[:,i], 'g', label='gt')
            plt.legend()
            plt.title(self.vel_map_inverse[i])
            plt.savefig("bindings/pydairlib/cassie/residual_analysis/hardware_test/v_dot/{}.png".format(self.vel_map_inverse[i]))

        for i in range(6):
            plt.cla()
            plt.plot(t_list, lambda_c_est_list[:,i], 'r', label="est")
            plt.legend()
            plt.title(self.lambda_c_map_inverse[i])
            plt.savefig("bindings/pydairlib/cassie/residual_analysis/hardware_test/lambda_c/{}.png".format(self.lambda_c_map_inverse[i]))
        
        import pdb; pdb.set_trace()

    def simulation_test(self):
        raw_data = scipy.io.loadmat(self.data_path)
        process_data = self.process_sim_data(raw_data, self.start_time, self.end_time)
        t = process_data["t"]; q = process_data["q"]; v = process_data["v"]; v_dot_gt = process_data["v_dot"]; 
        u = process_data["u"]; lambda_c_gt = process_data["lambda_c"]; lambda_c_position = process_data["lambda_c_position"]
        v_dot_osc = process_data["v_dot_osc"]; #u_osc = process_data["u_osc"]
        damping_ratio = process_data["damping_ratio"]; spring_stiffness = process_data["spring_stiffness"]

        self.cassie.get_damping(damping_ratio)
        self.cassie.get_spring_stiffness(spring_stiffness)

        t_list = []
        v_dot_est_list = []
        lambda_c_est_list = []
        lambda_h_est_list = []
        v_dot_gt_list = []
        lambda_c_gt_list = []
        # v_dot_osc_list = []

        for i in range(t.shape[0]):
            t_list.append(t[i])
            v_dot_est, lambda_c_est, lambda_h_est= self.cassie.calc_vdot(t[i], q[i,:], v[i,:], u[i,:], lambda_c_gt=lambda_c_gt[i,:], lambda_c_position=lambda_c_position[i,:], v_dot_gt=v_dot_gt[i,:])
            v_dot_est_list.append(v_dot_est)
            lambda_c_est_list.append(lambda_c_est)
            lambda_h_est_list.append(lambda_h_est)
            v_dot_gt_list.append(v_dot_gt[i,:])
            lambda_c_gt_list.append(lambda_c_gt[i,:])
            # v_dot_osc_list.append(v_dot_osc[i,:])

        t_list = np.array(t_list)
        v_dot_est_list = np.array(v_dot_est_list)        
        lambda_c_est_list = np.array(lambda_c_est_list)
        lambda_h_est_list = np.array(lambda_h_est_list)
        lambda_c_gt_list = np.array(lambda_c_gt_list)
        v_dot_gt_list = np.array(v_dot_gt_list)
        # v_dot_osc_list = np.array(v_dot_osc_list)

        # Save v_dot pictures
        for i in range(22):
            plt.cla()
            # plt.plot(t_list, v_dot_osc_list[:,i], 'b', label='osc')
            plt.plot(t_list, v_dot_est_list[:,i], 'r', label='est')
            plt.plot(t_list, v_dot_gt_list[:,i], 'g', label='gt')
            plt.legend()
            plt.title(self.vel_map_inverse[i])
            plt.ylim(-100,100)
            plt.savefig("bindings/pydairlib/cassie/residual_analysis/simulation_test/v_dot/{}.png".format(self.vel_map_inverse[i]))

        # Save residual pictures
        # self.draw_residual(lambda_c_gt[:,2]+lambda_c_gt[:,5]+lambda_c_gt[:,8]+lambda_c_gt[:,11], residual_list, 
        #                     "bindings/pydairlib/cassie/residual_analysis/simulation_test/residuals/total_contact_force_at_z_direction")
        
        # Save lambda_c
        for i in range(12):
            plt.cla()
            plt.plot(t_list, lambda_c_est_list[:,i], 'r', label='est')
            plt.plot(t_list, lambda_c_gt_list[:,i], 'g', label='gt')
            plt.legend()
            plt.title(self.lambda_c_map_inverse[i])
            plt.ylim(-200,200)
            plt.savefig("bindings/pydairlib/cassie/residual_analysis/simulation_test/lambda_c/{}.png".format(self.lambda_c_map_inverse[i]))
        
        import pdb; pdb.set_trace()

    def draw_residual(self, x_axis, residuals, dir):

        if not os.path.exists(dir):
            os.makedirs(dir)

        for i in range(22):

            n = x_axis.shape[0]
            x_axis_sorted = np.sort(x_axis)
            residuals_sorted = np.sort(residuals[:,i])
            x_min=x_axis_sorted[int(n*0.01)]; x_max=x_axis_sorted[int(n*(1-0.01))]
            y_min=residuals_sorted[int(n*0.01)]; y_max=residuals_sorted[int(n*(1-0.01))]

            corrcoef = np.corrcoef(np.vstack((x_axis, residuals[:,i])))

            A = np.hstack((x_axis[:,None], np.ones((n,1))))
            sol = np.linalg.pinv(A) @ residuals[:,i]

            plt.cla()
            plt.plot(x_axis, residuals[:, i], 'ro', markersize=1)
            plt.plot([x_min, x_max], [x_min*sol[0]+sol[1], x_max*sol[0]+sol[1]], 'b')
            plt.title(self.vel_map_inverse[i] + " corrcoef:" + "{:.2f}".format(corrcoef[1,0]))
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.savefig(dir + "/{}.png".format(self.vel_map_inverse[i]))

    def interpolation(self, t,t1,t2,v1,v2):
        ratio = (t - t1)/(t2-t1)
        v = v1 + ratio * (v2 -v1)
        return v

    def process_hardware_data(self, raw_data, start_time, end_time):
        
        # processing robot output
        robot_output = raw_data['robot_output']
        t_robot_output = robot_output["t_x"][0][0][0]
        start_index = np.argwhere(t_robot_output > start_time)[0][0]
        end_index = np.argwhere(t_robot_output < end_time)[-1][0]
        t_robot_output = t_robot_output[start_index:end_index]
        q = robot_output['q'][0][0][start_index:end_index,:]
        v = robot_output['v'][0][0][start_index:end_index,:]
        u = robot_output['u'][0][0][start_index:end_index,:]
        # obtained v_dot by finite diff
        smooth_window = 10
        v_dot = (robot_output['v'][0][0][start_index+smooth_window:end_index+smooth_window,:] - robot_output['v'][0][0][start_index-smooth_window:end_index-smooth_window,:])\
            /(robot_output["t_x"][0][0][0][start_index+smooth_window:end_index+smooth_window] - robot_output["t_x"][0][0][0][start_index-smooth_window:end_index-smooth_window])[:,None]
        
        v_dot = self.first_order_filter(v_dot)

        # processing contact force
        contact_output = raw_data['contact_output']
        t_contact = contact_output['t_contact'][0][0][0]
        start_index = np.argwhere(t_contact > start_time - 0.01)[0][0] # make sure contact force period range cover the output states
        end_index = np.argwhere(t_contact < end_time + 0.01)[-1][0]
        t_contact = t_contact[start_index: end_index]
        is_contact = contact_output['is_contact'][0][0][start_index:end_index,:]
        is_contact_processed = []
        pointer = 0
        for t in t_robot_output:
            while not (t_contact[pointer] <= t and t_contact[pointer+1] >=t):
                pointer+=1
            if abs(t_contact[pointer] - t) < abs(t_contact[pointer+1] - t):
                is_contact_processed.append(is_contact[pointer,:])
            else:
                is_contact_processed.append(is_contact[pointer+1,:])
        is_contact_processed = np.array(is_contact_processed)

        # Get the osc_output

        osc_output = raw_data["osc_output"]
        t_osc = osc_output["t_osc"][0][0][0]
        start_index = np.argwhere(t_osc > start_time - 0.01)[0][0]
        end_index = np.argwhere(t_osc < end_time + 0.01)[-1][0]
        t_osc = t_osc[start_index: end_index]
        u_osc = osc_output["u_sol"][0][0][start_index:end_index,:]
        u_osc_processed = []
        v_dot_osc = osc_output["dv_sol"][0][0][start_index:end_index,:]
        v_dot_osc_proccessed = []
        pointer = 0
        for t in t_robot_output:
            while not (t_osc[pointer] <= t and t_osc[pointer+1] >=t):
                pointer+=1
            u_osc_interpolated = self.interpolation(t, t_osc[pointer], t_osc[pointer+1],
                                                    u_osc[pointer,:], u_osc[pointer+1, :])
            v_dot_osc_interpolated = self.interpolation(t, t_osc[pointer], t_osc[pointer+1],
                                                    v_dot_osc[pointer,:], v_dot_osc[pointer+1, :])
            u_osc_processed.append(u_osc_interpolated)
            v_dot_osc_proccessed.append(v_dot_osc_interpolated)
        u_osc_processed = np.array(u_osc_processed)
        v_dot_osc_proccessed = np.array(v_dot_osc_proccessed)

        # Get damping ratio
        damping_ratio = raw_data['damping_ratio']

        # Get spring stiffness
        spring_stiffness = raw_data['spring_stiffness']

        processed_data = {
            't':t_robot_output,
            'q':q,
            'v':v,
            'v_dot':v_dot,
            'u':u,
            'is_contact':is_contact_processed,
            'u_osc': u_osc_processed,
            'v_dot_osc': v_dot_osc_proccessed,
            'damping_ratio':damping_ratio,
            'spring_stiffness':spring_stiffness
        }

        return processed_data

    def process_sim_data(self, raw_data, start_time, end_time):

        # processing robot output
        robot_output = raw_data['robot_output']
        t_robot_output = robot_output["t_x"][0][0][0]
        start_index = np.argwhere(t_robot_output > start_time)[0][0]
        end_index = np.argwhere(t_robot_output < end_time)[-1][0]
        t_robot_output = t_robot_output[start_index:end_index]
        q = robot_output['q'][0][0][start_index:end_index,:]
        v = robot_output['v'][0][0][start_index:end_index,:]
        u = robot_output['u'][0][0][start_index:end_index,:]
        # obtained v_dot by finite diff
        smooth_window = 10
        v_dot = (robot_output['v'][0][0][start_index+smooth_window:end_index+smooth_window,:] - robot_output['v'][0][0][start_index-smooth_window:end_index-smooth_window,:])\
            /(robot_output["t_x"][0][0][0][start_index+smooth_window:end_index+smooth_window] - robot_output["t_x"][0][0][0][start_index-smooth_window:end_index-smooth_window])[:,None]
        
        v_dot = self.first_order_filter(v_dot)

        # processing contact force
        contact_output = raw_data['contact_output']
        t_contact = contact_output['t_lambda'][0][0][0]
        start_index = np.argwhere(t_contact > start_time - 0.01)[0][0] # make sure contact force period range cover the output states
        end_index = np.argwhere(t_contact < end_time + 0.01)[-1][0]
        t_contact = t_contact[start_index: end_index]
        contact_force = contact_output['lambda_c'][0][0]
        contact_position = contact_output['p_lambda_c'][0][0]
        left_toe_front_force = contact_force[0,start_index:end_index,:]
        left_toe_front_position = contact_position[0,start_index:end_index,:]
        left_toe_rear_force = contact_force[1,start_index:end_index,:]
        left_toe_rear_position = contact_position[1,start_index:end_index,:]
        right_toe_front_force = contact_force[2,start_index:end_index,:]
        right_toe_front_position = contact_position[2,start_index:end_index,:]
        right_toe_rear_force = contact_force[3,start_index:end_index,:]
        right_toe_rear_position = contact_position[3,start_index:end_index,:]
        contact_force_processed = []
        contact_position_processed = []
        pointer = 0
        for t in t_robot_output:
            while not (t_contact[pointer] <= t and t_contact[pointer+1] >=t):
                pointer+=1
            left_toe_front_force_interpolated = self.interpolation(t, t_contact[pointer], t_contact[pointer+1], 
                                                        left_toe_front_force[pointer,:], left_toe_front_force[pointer+1,:])
            left_toe_front_position_interpolated = self.interpolation(t, t_contact[pointer], t_contact[pointer+1],
                                                        left_toe_front_position[pointer,:], left_toe_front_position[pointer+1,:])
            left_toe_rear_force_interpolated = self.interpolation(t, t_contact[pointer], t_contact[pointer+1],
                                                        left_toe_rear_force[pointer,:], left_toe_rear_force[pointer+1,:])
            left_toe_rear_position_interpolated = self.interpolation(t, t_contact[pointer], t_contact[pointer+1],
                                                        left_toe_rear_position[pointer,:], left_toe_rear_position[pointer+1,:])
            right_toe_front_force_interpolated = self.interpolation(t, t_contact[pointer], t_contact[pointer+1],
                                                        right_toe_front_force[pointer,:], right_toe_front_force[pointer+1,:])
            right_toe_front_position_interpolated = self.interpolation(t, t_contact[pointer], t_contact[pointer+1],
                                                        right_toe_front_position[pointer,:], right_toe_front_position[pointer+1,:])
            right_toe_rear_force_interpolated = self.interpolation(t, t_contact[pointer], t_contact[pointer+1],
                                                        right_toe_rear_force[pointer,:], right_toe_rear_force[pointer+1,:])
            right_toe_rear_position_interpolated = self.interpolation(t, t_contact[pointer], t_contact[pointer+1],
                                                        right_toe_rear_position[pointer,:], right_toe_rear_position[pointer+1,:])
            joined_positions = np.hstack((left_toe_front_position_interpolated, left_toe_rear_position_interpolated,
                                        right_toe_front_position_interpolated, right_toe_rear_position_interpolated))
            joined_forces = np.hstack((left_toe_front_force_interpolated, left_toe_rear_force_interpolated, right_toe_front_force_interpolated, right_toe_rear_force_interpolated))
            contact_force_processed.append(joined_forces)
            contact_position_processed.append(joined_positions)
        contact_force_processed = np.array(contact_force_processed)
        contact_position_processed = np.array(contact_position_processed)

        # Get the osc_output

        osc_output = raw_data["osc_output"]
        t_osc = osc_output["t_osc"][0][0][0]
        start_index = np.argwhere(t_osc > start_time - 0.01)[0][0]
        end_index = np.argwhere(t_osc < end_time + 0.01)[-1][0]
        t_osc = t_osc[start_index: end_index]
        u_osc = osc_output["u_sol"][0][0][start_index:end_index,:]
        u_osc_processed = []
        v_dot_osc = osc_output["dv_sol"][0][0][start_index:end_index,:]
        v_dot_osc_proccessed = []
        pointer = 0
        for t in t_robot_output:
            while not (t_osc[pointer] <= t and t_osc[pointer+1] >=t):
                pointer+=1
            u_osc_interpolated = self.interpolation(t, t_osc[pointer], t_osc[pointer+1],
                                                    u_osc[pointer,:], u_osc[pointer+1, :])
            v_dot_osc_interpolated = self.interpolation(t, t_osc[pointer], t_osc[pointer+1],
                                                    v_dot_osc[pointer,:], v_dot_osc[pointer+1, :])
            u_osc_processed.append(u_osc_interpolated)
            v_dot_osc_proccessed.append(v_dot_osc_interpolated)
        u_osc_processed = np.array(u_osc_processed)
        v_dot_osc_proccessed = np.array(v_dot_osc_proccessed)

        # Get damping ratio
        damping_ratio = raw_data['damping_ratio']

        # Get spring stiffness
        spring_stiffness = raw_data['spring_stiffness']

        processed_data = {
            't':t_robot_output,
            'q':q,
            'v':v,
            'v_dot':v_dot,
            'u':u,
            'lambda_c':contact_force_processed,
            'lambda_c_position':contact_position_processed,
            'u_osc': u_osc_processed,
            'v_dot_osc': v_dot_osc_proccessed,
            'damping_ratio':damping_ratio,
            'spring_stiffness':spring_stiffness
        }

        return processed_data

def main():
    simulationDataTester = CaaiseSystemTest(data_path="log/03_15_22/lcmlog-11.mat", start_time=31, end_time=35)
    simulationDataTester.hardware_test()

if __name__ == "__main__":
    main()