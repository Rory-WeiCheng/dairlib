import numpy as np
import matplotlib.pyplot as plt

from pydairlib.common import plot_styler, plotting_utils
from pydrake.multibody.tree import JacobianWrtVariable
from osc_debug import lcmt_osc_tracking_data_t, osc_tracking_cost
from pydairlib.multibody import makeNameToPositionsMap, \
    makeNameToVelocitiesMap, makeNameToActuatorsMap, \
    createStateNameVectorFromMap, createActuatorNameVectorFromMap


def make_name_to_mbp_maps(plant):
    return makeNameToPositionsMap(plant), \
           makeNameToVelocitiesMap(plant), \
           makeNameToActuatorsMap(plant)


def make_mbp_name_vectors(plant):
    x_names = createStateNameVectorFromMap(plant)
    u_names = createActuatorNameVectorFromMap(plant)
    q_names = x_names[:plant.num_positions()]
    v_names = x_names[plant.num_positions():]
    return q_names, v_names, u_names


def make_joint_order_permutation_matrix(names_in_old_order, names_in_new_order):
    n = len(names_in_new_order)
    perm = np.zeros((n, n), dtype=int)
    for i, name in enumerate(names_in_new_order):
        try:
            j = names_in_old_order.index(name)
        except ValueError:
            print(f"Error: {name} not found in old joint ordering")
            raise
        except BaseException as err:
            print(f"Unexpected {err}, {type(err)}")
            raise
        perm[i, j] = 1
    return perm


def make_joint_order_permutations(robot_output_message, plant):
    qnames, vnames, unames = make_mbp_name_vectors(plant)
    qperm = make_joint_order_permutation_matrix(
        robot_output_message.position_names, qnames)
    vperm = make_joint_order_permutation_matrix(
        robot_output_message.velocity_names, vnames)
    uperm = make_joint_order_permutation_matrix(
        robot_output_message.effort_names, unames)
    return qperm, vperm, uperm


def process_state_channel(state_data, plant):
    t_x = []
    q = []
    u = []
    v = []

    pos_map = makeNameToPositionsMap(plant)
    vel_map = makeNameToVelocitiesMap(plant)
    act_map = makeNameToActuatorsMap(plant)

    for msg in state_data:
        q_temp = [[] for i in range(len(msg.position))]
        v_temp = [[] for i in range(len(msg.velocity))]
        u_temp = [[] for i in range(len(msg.effort))]
        for i in range(len(q_temp)):
            q_temp[pos_map[msg.position_names[i]]] = msg.position[i]
        for i in range(len(v_temp)):
            v_temp[vel_map[msg.velocity_names[i]]] = msg.velocity[i]
        for i in range(len(u_temp)):
            u_temp[act_map[msg.effort_names[i]]] = msg.effort[i]
        q.append(q_temp)
        v.append(v_temp)
        u.append(u_temp)
        t_x.append(msg.utime / 1e6)

    return {'t_x': np.array(t_x),
            'q': np.array(q),
            'v': np.array(v),
            'u': np.array(u)}


def process_effort_channel(data, plant):
    u = []
    t = []

    act_map = makeNameToActuatorsMap(plant)
    for msg in data:
        u_temp = [[] for i in range(len(msg.efforts))]
        for i in range(len(u_temp)):
            u_temp[act_map[msg.effort_names[i]]] = msg.efforts[i]
        t.append(msg.utime / 1e6)
        u.append(u_temp)

    return {'t_u': np.array(t), 'u': np.array(u)}

def process_ball_position_channel(data):
  id = []
  num_cameras_used = []
  xyz = []
  cam_statuses = []
  t = []
  dt = []

  prev_t = 0.0
  prev_id = -1
  curr_dt = 0.0
  first_message = True
  for msg in data:
    if first_message:
      first_message = False
      prev_t = msg.utime /1e-6
      prev_id = msg.id
    elif msg.id != prev_id:
      curr_dt = msg.utime / 1e6 - prev_t
      prev_t = msg.utime / 1e6
      prev_id = msg.id
    dt.append(curr_dt)

    id.append(msg.id)
    num_cameras_used.append(msg.num_cameras_used)

    xyz_temp = [msg.xyz[0], msg.xyz[1], msg.xyz[2]]
    xyz.append(xyz_temp)

    cam_status_temp = []
    for i in range(3):
      cam_status_temp.append(msg.cam_statuses[i])
    cam_statuses.append(cam_status_temp)

    t.append(msg.utime / 1e6)
  
  return {'t': np.array(t),
          'xyz': np.array(xyz),
          'num_cameras_used': np.array(num_cameras_used).reshape(-1, 1),
          'cam_statuses': cam_statuses,
          'id': np.array(id),
          'dt': np.array(dt).reshape(-1, 1)}

def process_c3_channel(data):
  x_d = []
  xdot_d = []
  f_d = []
  t = []

  for msg in data:
    # extract desire EE pos info
    x_d_temp = []
    for i in range(3):
      x_d_temp.append(msg.data[i])
    x_d.append(x_d_temp)

    # extract desired EE vel info
    xdot_d_temp = []
    for i in range(10, 13):
      xdot_d_temp.append(msg.data[i])
    xdot_d.append(xdot_d_temp)

    # extract force info in n, t1, t2 directions
    f_d_temp = [msg.data[24], msg.data[25]-msg.data[26], msg.data[27]-msg.data[28]]
    f_d.append(f_d_temp)

    t.append(msg.utime / 1e6)
  
  solve_times = [0.0]
  for i in range(1, len(t)):
    solve_times.append(t[i]-t[i-1])
  
  return {'t': np.array(t), 'x_d': np.array(x_d), 
          'xdot_d': np.array(xdot_d), 'f_d': np.array(f_d),
          'solve_times': np.array(solve_times).reshape(-1, 1)}

# 2023.7.28 newly added for adaptive learning new lcm message visualization
def process_dataset_channel(data):
    t = []
    state = []
    input = []
    state_pred = []

    A_M = []
    B_M = []
    D_M = []
    d_M = []

    E_M = []
    F_M = []
    H_M = []
    c_M = []

    for msg in data:
        # get the state from learning dataset
        t.append(msg.utime / 1e6)

        state.append(msg.state)
        input.append(msg.input)
        state_pred.append(msg.state_pred) # only velocity part

        A_M.append(msg.A)
        B_M.append(msg.B)
        D_M.append(msg.D)
        d_M.append(msg.d)

        E_M.append(msg.E)
        F_M.append(msg.F)
        H_M.append(msg.H)
        c_M.append(msg.c)

    # not extracting orientation and angular velocity
    return {'t': np.array(t), 'ee_pos': np.array(state)[:,0:3], 'ee_vel': np.array(state)[:,10:13], 'ball_pos': np.array(state)[:,7:10],
            'ball_vel': np.array(state)[:,16:19], 'input': np.array(input),'ee_vel_pred': np.array(state_pred)[:,0:3],'ball_vel_pred': np.array(state_pred)[:,6:9],
            'A_M': np.array(A_M), 'B_M': np.array(B_M), 'D_M': np.array(D_M), 'd_M': np.array(d_M),
            'E_M': np.array(E_M), 'F_M': np.array(F_M), 'H_M': np.array(H_M), 'c_M': np.array(c_M)
            }
def process_residual_channel(data):
    t = []
    d_res = []
    c_res = []

    for msg in data:
        # get the state from residual dataset
        t.append(msg.utime / 1e6) # this time is the time for the initial data of the batch data

        d_res.append(msg.d)
        c_res.append(msg.c)

    return {'t': np.array(t), 'd_res': np.array(d_res)[:,6:9], 'c_res': np.array(c_res), 'c_res_eeb': np.array(c_res)[:,0:4], 'c_res_bg': np.array(c_res)[:,4:8]}

def process_learning_visual_channel(data):
    t = []
    total_loss = []
    dyn_loss = []
    lcp_loss = []
    c_grad = []
    d_grad = []
    lambda_n = []
    lambda_all = []
    residual = []

    for msg in data:
        # get the state from residual dataset
        t.append(msg.utime / 1e6) # this time is the time for the initial data of the batch data

        total_loss.append(msg.total_loss)
        dyn_loss.append(msg.dyn_loss)
        lcp_loss.append(msg.lcp_loss)
        c_grad.append(msg.c_grad)
        d_grad.append(msg.d_grad)
        lambda_n.append(msg.lambda_n)
        lambda_all.append(msg.lambda_check)
        residual.append(msg.res_check)

    return {'t': np.array(t), 'total_loss': np.array(total_loss), 'dyn_loss': np.array(dyn_loss), 'lcp_loss': np.array(lcp_loss),
            'loss_stack': np.column_stack((np.array(total_loss),np.array(dyn_loss),np.array(lcp_loss))),
            'c_grad': np.array(c_grad),'d_grad': np.array(d_grad), 'lambda_n': np.array(lambda_n), 'lambda_all': np.array(lambda_all),
            'c_grad_eeb':np.array(c_grad)[:,0:4], 'c_grad_bg':np.array(c_grad)[:,4:8], 'lambda_n_eeb': np.array(lambda_n)[:,0].reshape(-1, 1),
            'lambda_eeb': np.array(lambda_all)[:,0:4], 'residual': np.array(residual)[:,-3:]
            }


def make_point_positions_from_q(
        q, plant, context, frame, pt_on_frame, frame_to_calc_position_in=None):

    if frame_to_calc_position_in is None:
        frame_to_calc_position_in = plant.world_frame()

    pos = np.zeros((q.shape[0], 3))
    for i, generalized_pos in enumerate(q):
        plant.SetPositions(context, generalized_pos)
        pos[i] = plant.CalcPointsPositions(context, frame, pt_on_frame,
                                           frame_to_calc_position_in).ravel()

    return pos

def make_point_velocities(
        q, v, plant, context, frame, pt_on_frame, frame_to_calc_velocity_in=None):

    if frame_to_calc_velocity_in is None:
        frame_to_calc_velocity_in = plant.world_frame()

    vel = np.zeros((q.shape[0], 3))
    for i in range(q.shape[0]):
        generalized_pos = q[i]
        generalized_vel = v[i]
        plant.SetPositions(context, generalized_pos)
        plant.SetVelocities(context, generalized_vel)

        J_trans = plant.CalcJacobianTranslationalVelocity( \
            context, JacobianWrtVariable.kV, frame, pt_on_frame, \
            frame_to_calc_velocity_in, \
            frame_to_calc_velocity_in)
        vel[i] = np.matmul(J_trans, generalized_vel)

    return vel


def process_osc_channel(data):
    t_osc = []
    input_cost = []
    accel_cost = []
    soft_constraint_cost = []
    qp_solve_time = []
    u_sol = []
    lambda_c_sol = []
    lambda_h_sol = []
    dv_sol = []
    epsilon_sol = []
    osc_output = []
    fsm = []
    osc_debug_tracking_datas = {}

    for msg in data:
        t_osc.append(msg.utime / 1e6)
        input_cost.append(msg.input_cost)
        accel_cost.append(msg.acceleration_cost)
        soft_constraint_cost.append(msg.soft_constraint_cost)
        qp_solve_time.append(msg.qp_output.solve_time)
        u_sol.append(msg.qp_output.u_sol)
        lambda_c_sol.append(msg.qp_output.lambda_c_sol)
        lambda_h_sol.append(msg.qp_output.lambda_h_sol)
        dv_sol.append(msg.qp_output.dv_sol)
        epsilon_sol.append(msg.qp_output.epsilon_sol)

        osc_output.append(msg)
        for tracking_data in msg.tracking_data:
            if tracking_data.name not in osc_debug_tracking_datas:
                osc_debug_tracking_datas[tracking_data.name] = \
                    lcmt_osc_tracking_data_t()
            osc_debug_tracking_datas[tracking_data.name].append(
                tracking_data, msg.utime / 1e6)

        fsm.append(msg.fsm_state)

    tracking_cost_handler = osc_tracking_cost(osc_debug_tracking_datas.keys())
    for msg in data:
        tracking_cost_handler.append(msg.tracking_data_names, msg.tracking_cost)
    tracking_cost = tracking_cost_handler.convertToNP()

    for name in osc_debug_tracking_datas:
        osc_debug_tracking_datas[name].convertToNP()

    return {'t_osc': np.array(t_osc),
            'input_cost': np.array(input_cost),
            'acceleration_cost': np.array(accel_cost),
            'soft_constraint_cost': np.array(soft_constraint_cost),
            'qp_solve_time': np.array(qp_solve_time),
            'u_sol': np.array(u_sol),
            'lambda_c_sol': np.array(lambda_c_sol),
            'lambda_h_sol': np.array(lambda_h_sol),
            'dv_sol': np.array(dv_sol),
            'epsilon_sol': np.array(epsilon_sol),
            'tracking_cost': tracking_cost,
            'osc_debug_tracking_datas': osc_debug_tracking_datas,
            'fsm': np.array(fsm),
            'osc_output': osc_output}


def permute_osc_joint_ordering(osc_data, robot_output_msg, plant):
    _, vperm, uperm = make_joint_order_permutations(robot_output_msg, plant)
    osc_data['u_sol'] = (osc_data['u_sol'] @ uperm.T)
    osc_data['dv_sol'] = (osc_data['dv_sol'] @ vperm.T)
    return osc_data


def load_default_channels(data, plant, state_channel, input_channel,
                          osc_debug_channel):
    robot_output = process_state_channel(data[state_channel], plant)
    robot_input = process_effort_channel(data[input_channel], plant)
    osc_debug = process_osc_channel(data[osc_debug_channel])
    osc_debug = permute_osc_joint_ordering(
        osc_debug, data[state_channel][0], plant)

    return robot_output, robot_input, osc_debug

# def load_default_franka_channels(data, plant, state_channel, input_channel, c3_channel):
def load_default_franka_channels(data, plant, state_channel, input_channel, c3_channel,
    cam0_channel, cam1_channel, cam2_channel, vision_channel):
    
    print("\nDetected the following channels:")
    print(data.keys())
    print('')

    robot_output = process_state_channel(data[state_channel], plant)
    robot_input = process_effort_channel(data[input_channel], plant)
    c3_output = process_c3_channel(data[c3_channel])
    cam0 = process_ball_position_channel(data[cam0_channel])
    cam1 = process_ball_position_channel(data[cam1_channel])
    cam2 = process_ball_position_channel(data[cam2_channel])
    vision = process_ball_position_channel(data[vision_channel])

    return robot_output, robot_input, c3_output, cam0, cam1, cam2, vision

def load_default_franka_channels_reduction(data, plant, state_channel, input_channel, c3_channel):

    print("\nDetected the following channels:")
    print(data.keys())
    print('')

    robot_output = process_state_channel(data[state_channel], plant)
    robot_input = process_effort_channel(data[input_channel], plant)
    c3_output = process_c3_channel(data[c3_channel])

    return robot_output, robot_input, c3_output

def load_default_franka_channels_adaptive_learning(data, plant, state_channel, input_channel, c3_channel,\
                                                   dataset_channel, residual_channel, learning_visual):

    print("\nDetected the following channels:")
    print(data.keys())
    print('')

    robot_output = process_state_channel(data[state_channel], plant)
    robot_input = process_effort_channel(data[input_channel], plant)
    c3_output = process_c3_channel(data[c3_channel])
    learning_dataset = process_dataset_channel(data[dataset_channel])
    residual_lcs = process_residual_channel(data[residual_channel])
    learning_visual = process_learning_visual_channel(data[learning_visual])

    return robot_output, robot_input, c3_output, learning_dataset, residual_lcs, learning_visual

def load_franka_state_estimate_channel(data, plant, state_channel):
    return process_state_channel(data[state_channel], plant)


def plot_q_or_v_or_u(
        robot_output, key, x_names, x_slice, time_slice,
        ylabel=None, title=None):
    ps = plot_styler.PlotStyler()
    if ylabel is None:
        ylabel = key
    if title is None:
        title = key

    plotting_utils.make_plot(
        robot_output,                       # data dict
        't_x',                              # time channel
        time_slice,
        [key],                              # key to plot
        {key: x_slice},                     # slice of key to plot
        {key: x_names},                     # legend entries
        {'xlabel': 'Time',
         'ylabel': ylabel,
         'title': title}, ps)
    return ps

def plot_c3_plan(c3_output, key, time_slice, ylabel=None, title=None):
  titles = {'x_d': 'Desired End Effector Position',
            'xdot_d': 'Desired End Effector Velocity',
            'f_d': 'Desired End Effector Forces',
            'solve_times': 'C3 Solve Times'}
  ylabels = {'x_d': 'Position [m]',
            'xdot_d': 'Velocity [m/s]',
            'f_d': 'Force [N]',
            'solve_times': 'Time [s]'}
  if title is None:
    title = titles[key]
  if ylabel is None:
    ylabel = ylabels[key]
  
  if key == 'x_d':
    legend = ['x', 'y', 'z']
  elif key == 'xdot_d':
    legend = ['xdot', 'ydot', 'zdot']
  elif key == 'f_d':
    legend = ['n', 't1', 't2']
  elif key == 'solve_times':
    legend = ['Solve Time']
  else:
    raise Exception("Key must be one of x_d, xdot_d, or f_d")
  
  ps = plot_styler.PlotStyler()
  plotting_utils.make_plot(
    c3_output,
    't',
    time_slice,
    [key],
    {key: slice(len(legend))},
    {key: legend},
    {'xlabel': 'Time',
     'ylabel': ylabel,
     'title': title},
    ps)

  return ps

def plot_ball_position(output, key, time_slice, ylabel=None, title=None):
  titles = {'xyz': 'Vision Output',
            'num_cameras_used': 'Number of Valid Camera Measurements',
            'dt': 'Frame Period',
            'solve_times': 'C3 Solve Times'}
  ylabels = {'xyz': 'Position [m]',
            'num_cameras_used': 'Number of Valid Camera Measurements',
            'dt': 'Period [s]'}

  if title is None:
    title = titles[key]
  if ylabel is None:
    ylabel = ylabels[key]
  
  if key == 'xyz':
    legend = ['x', 'y', 'z']
  elif key == 'num_cameras_used':
    legend = ['Valid Cameras']
  elif key == 'dt':
    legend = ['period']
  else:
    raise Exception("Key must be one of x_d, xdot_d, or f_d")
  
  ps = plot_styler.PlotStyler()
  plotting_utils.make_plot(
    output,
    't',
    time_slice,
    [key],
    {key: slice(len(legend))},
    {key: legend},
    {'xlabel': 'Time',
     'ylabel': ylabel,
     'title': title},
    ps)

  return ps

def plot_multiple_ball_positions(outputs, time_slice, legend, ylabel='Position [m]', title='Raw Camera Outputs'):
  min_msgs = float('inf')
  for i in range(len(outputs)):
    min_msgs = min(min_msgs, outputs[i]['xyz'].shape[0])

  data = outputs[0]['xyz'][:min_msgs, :]
  for i in range(1, len(outputs)):
    data = np.concatenate((data, outputs[i]['xyz'][:min_msgs, :]), axis = 1)
  
  ps = plot_styler.PlotStyler()
  plotting_utils.make_plot(
    {'t': outputs[0]['t'][:min_msgs],
     'data': data},
    't',
    time_slice,
    ['data'],
    {'data': slice(3*len(outputs))},
    {'data': legend},
    {'xlabel': 'Time',
     'ylabel': ylabel,
     'title': title},
    ps)

  return ps



def plot_floating_base_positions(robot_output, q_names, fb_dim, time_slice):
    return plot_q_or_v_or_u(robot_output, 'q', q_names[:fb_dim], slice(fb_dim),
                            time_slice, ylabel='Position',
                            title='Floating Base Positions')


def plot_joint_positions(robot_output, q_names, fb_dim, time_slice):
    q_slice = slice(fb_dim, len(q_names))
    return plot_q_or_v_or_u(robot_output, 'q', q_names[q_slice], q_slice,
                            time_slice, ylabel='Joint Angle (rad)',
                            title='Joint Positions')


def plot_positions_by_name(robot_output, q_names, time_slice, pos_map, title='Select Positions'):
    q_slice = [pos_map[name] for name in q_names]
    return plot_q_or_v_or_u(robot_output, 'q', q_names, q_slice, time_slice,
                            ylabel='Position', title=title)


def plot_floating_base_velocities(robot_output, v_names, fb_dim, time_slice):
    return plot_q_or_v_or_u(robot_output, 'v', v_names[:fb_dim], slice(fb_dim),
                            time_slice, ylabel='Velocity',
                            title='Floating Base Velocities')


def plot_joint_velocities(robot_output, v_names, fb_dim, time_slice):
    q_slice = slice(fb_dim, len(v_names))
    return plot_q_or_v_or_u(robot_output, 'v', v_names[q_slice], q_slice,
                            time_slice, ylabel='Joint Vel (rad/s)',
                            title='Joint Velocities')


def plot_velocities_by_name(robot_output, v_names, time_slice, vel_map, title='Select Velocities'):
    v_slice = [vel_map[name] for name in v_names]
    return plot_q_or_v_or_u(robot_output, 'v', v_names, v_slice, time_slice,
                            ylabel='Velocity', title=title)


def plot_measured_efforts(robot_output, u_names, time_slice):
    return plot_q_or_v_or_u(robot_output, 'u', u_names, slice(len(u_names)),
                            time_slice, ylabel='Efforts (Nm)',
                            title='Joint Efforts')

def plot_desired_efforts(robot_input, u_names, time_slice):
    label = "Desired Joint Efforts"
    key = 'u'
    ps = plot_styler.PlotStyler()

    plotting_utils.make_plot(
      robot_input,
      't_u',
      time_slice,
      [key],
      {key: slice(len(u_names))},
      {key: u_names},
      {'xlabel': 'Time',
         'ylabel': label,
         'title': label}, ps)
    return ps


def plot_measured_efforts_by_name(robot_output, u_names, time_slice, u_map, title='Select Joint Efforts'):
    u_slice = [u_map[name] for name in u_names]
    return plot_q_or_v_or_u(robot_output, 'u', u_names, u_slice, time_slice,
                            ylabel='Efforts (Nm)', title=title)

############################ newly added for data set and learning plots ####################################
def plot_learning_dataset(learning_dataset, key, time_slice, ylabel=None, title = None):
    titles = {'ee_pos': 'Learning Data Set: End-Effector Position',
              'ee_vel': 'Learning Data Set: End-Effector Velocity',
              'ball_pos': 'Learning Data Set: Ball Position',
              'ball_vel': 'Learning Data Set: End-Effector Position',
              'input': 'Learning Data Set: C3 Input',
              'ee_pos_pred': 'Learning Data Set: End-Effector Position Prediction',
              'ee_vel_pred': 'Learning Data Set: End-Effector Velocity Prediction',
              'ball_pos_pred': 'Learning Data Set: Ball Position Prediction',
              'ball_vel_pred': 'Learning Data Set: End-Effector Velocity Prediction',
              }
    ylabels = {'ee_pos': 'Position [m]',
              'ee_vel': 'Velocity [m/s]',
              'ball_pos': 'Position [m]',
              'ball_vel': 'Velocity [m/s]',
              'input': 'Force [N]',
              'ee_pos_pred': 'Position [m]',
              'ee_vel_pred': 'Velocity [m/s]',
              'ball_pos_pred': 'Position [m]',
              'ball_vel_pred': 'Velocity [m/s]',
              }
    if title is None:
        title = titles[key]
    if ylabel is None:
        ylabel = ylabels[key]

    if key == 'ee_pos' or key == 'ball_pos' or key == 'ee_pos_pred' or key == 'ball_pos_pred':
        legend = ['x', 'y', 'z']
    elif key == 'ee_vel' or key == 'ball_vel' or key == 'ee_vel_pred' or key == 'ball_vel_pred':
        legend = ['xdot', 'ydot', 'zdot']
    elif key == 'input':
        legend = ['Fx', 'Fy', 'Fz']
    else:
        raise Exception("Please assign the correct key value")

    ps = plot_styler.PlotStyler()
    plotting_utils.make_plot(
        learning_dataset,
        't',
        time_slice,
        [key],
        {key: slice(len(legend))},
        {key: legend},
        {'xlabel': 'Time',
         'ylabel': ylabel,
         'title': title},
        ps)

    return ps

def plot_residual_lcs(residual_lcs, key, time_slice, ylabel=None, title = None):
    titles = {'c_res': 'LCP Offset Residual',
              'd_res': 'Dynamic Offset Residual',
              'c_res_eeb': 'LCP Offset Residual (EE and ball)',
              'c_res_bg': 'LCP Offset Residual (ball and ground)',
              }
    ylabels = {'c_res': 'c_res',
               'd_res': 'd_res',
               'c_res_eeb': 'c_res_eeb',
               'c_res_bg': 'c_res_bg',
               }
    if title is None:
        title = titles[key]
    if ylabel is None:
        ylabel = ylabels[key]

    if key == 'c_res':
        legend = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
    elif key == 'd_res':
        legend = ['d6', 'd7', 'd8']
    elif key == 'c_res_eeb':
        legend = ['c0', 'c1', 'c2', 'c3']
    elif key == 'c_res_bg':
        legend = ['c4', 'c5', 'c6', 'c7']
    else:
        raise Exception("Please assign the correct key value")

    ps = plot_styler.PlotStyler()
    plotting_utils.make_plot(
        residual_lcs,
        't',
        time_slice,
        [key],
        {key: slice(len(legend))},
        {key: legend},
        {'xlabel': 'Time',
         'ylabel': ylabel,
         'title': title},
        ps)

    return ps

def plot_learning_visual(learning_visual, key, time_slice, ylabel=None, title = None):
    titles = {'c_grad_eeb': 'LCP Offset Gradient (EE and ball)',
              'c_grad_bg': 'LCP Offset Gradient (ball and ground)',
              'd_grad': 'Dynamic Offset Gradient',
              'loss_stack': 'Learning loss',
              'lambda_n_eeb': 'normal contact (EE and ball)',
              'lambda_eeb': 'contact forces (EE and ball)',
              'residual': 'Ball Velocity Residual'
              }
    ylabels = {'c_grad_eeb': 'c_grad_eeb',
               'c_grad_bg': 'c_grad_bg',
               'd_grad': 'd_grad',
               'loss_stack': 'loss',
               'lambda_n_eeb': 'normal contact (EE and ball) (N)',
               'lambda_eeb': 'contact forces (EE and ball) (N)',
               'residual': 'Ball Velocity Residual (m/s)'
               }
    if title is None:
        title = titles[key]
    if ylabel is None:
        ylabel = ylabels[key]

    if key == 'c_grad_eeb':
        legend = ['c_grad_0', 'c_grad_1', 'c_grad_2', 'c_grad_3']
    elif key == 'c_grad_bg':
        legend = ['c_grad_4', 'c_grad_5', 'c_grad_6', 'c_grad_7']
    elif key == 'd_grad':
        legend = ['d_grad_6', 'd_grad_7', 'd_grad_8']
    elif key == 'loss_stack':
        legend = ['total_loss', 'dyn_loss', 'lcp_loss']
    elif key == 'lambda_n_eeb':
        legend = ['lambda_n_eeb']
    elif key == 'lambda_eeb':
        legend = ['lambda_eeb_0', 'lambda_eeb_1', 'lambda_eeb_2', 'lambda_eeb_3']
    elif key == 'residual':
        legend = ['v_res_x', 'v_res_y', 'v_res_z']
    else:
        raise Exception("Please assign the correct key value")

    ps = plot_styler.PlotStyler()
    plotting_utils.make_plot(
        learning_visual,
        't',
        time_slice,
        [key],
        {key: slice(len(legend))},
        {key: legend},
        {'xlabel': 'Time',
         'ylabel': ylabel,
         'title': title},
        ps)

    return ps

############################ newly added for data set and learning plots ####################################
def plot_points_positions(robot_output, time_slice, plant, context, frame_names,
                          pts, dims):

    dim_map = ['_x', '_y', '_z']
    data_dict = {'t': robot_output['t_x']}
    legend_entries = {}
    for name in frame_names:
        frame = plant.GetBodyByName(name).body_frame()
        pt = pts[name]
        data_dict[name] = make_point_positions_from_q(robot_output['q'],
                                                      plant, context, frame, pt)
        legend_entries[name] = [name + dim_map[dim] for dim in dims[name]]
    ps = plot_styler.PlotStyler()
    plotting_utils.make_plot(
        data_dict,
        't',
        time_slice,
        frame_names,
        dims,
        legend_entries,
        {'title': 'Point Positions',
         'xlabel': 'time (s)',
         'ylabel': 'pos (m)'}, ps)

    return ps

def plot_points_velocities(robot_output, time_slice, plant, context, frame_names, pts, dims):
    dim_map = ['_vx', '_vy', '_vz']
    data_dict = {'t': robot_output['t_x']}
    legend_entries = {}
    for name in frame_names:
        frame = plant.GetBodyByName(name).body_frame()
        pt = pts[name]
        data_dict[name] = make_point_velocities(robot_output['q'], robot_output['v'], \
                                                      plant, context, frame, pt)
        legend_entries[name] = [name + dim_map[dim] for dim in dims[name]]
    ps = plot_styler.PlotStyler()
    plotting_utils.make_plot(
        data_dict,
        't',
        time_slice,
        frame_names,
        dims,
        legend_entries,
        {'title': 'Point Velocity',
        'xlabel': 'time (s)',
        'ylabel': 'vel (m/s)'}, ps)

    return ps 

def plot_tracking_costs(osc_debug, time_slice):
    ps = plot_styler.PlotStyler()
    data_dict = \
        {key: val for key, val in osc_debug['tracking_cost'].items()}
    data_dict['t_osc'] = osc_debug['t_osc']

    plotting_utils.make_plot(
        data_dict,
        't_osc',
        time_slice,
        osc_debug['tracking_cost'].keys(),
        {},
        {key: [key] for key in osc_debug['tracking_cost'].keys()},
        {'xlabel': 'Time',
         'ylabel': 'Cost',
         'title': 'tracking_costs'}, ps)
    return ps


def plot_general_osc_tracking_data(traj_name, deriv, dim, data, time_slice):
    ps = plot_styler.PlotStyler()
    keys = [key for key in data.keys() if key != 't']
    plotting_utils.make_plot(
        data,
        't',
        time_slice,
        keys,
        {},
        {key: [key] for key in keys},
        {'xlabel': 'Time',
         'ylabel': '',
         'title': f'{traj_name} {deriv} tracking {dim}'}, ps)
    return ps


def plot_osc_tracking_data(osc_debug, traj, dim, deriv, time_slice):
    tracking_data = osc_debug['osc_debug_tracking_datas'][traj]
    data = {}
    if deriv == 'pos':
        data['y_des'] = tracking_data.y_des[:, dim]
        data['y'] = tracking_data.y[:, dim]
        data['error_y'] = tracking_data.error_y[:, dim]
    elif deriv == 'vel':
        data['ydot_des'] = tracking_data.ydot_des[:, dim]
        data['ydot'] = tracking_data.ydot[:, dim]
        data['error_ydot'] = tracking_data.error_ydot[:, dim]
    elif deriv == 'accel':
        data['yddot_des'] = tracking_data.yddot_des[:, dim]
        data['yddot_command'] = tracking_data.yddot_command[:, dim]
        data['yddot_command_sol'] = tracking_data.yddot_command_sol[:, dim]

    data['t'] = tracking_data.t
    return plot_general_osc_tracking_data(traj, deriv, dim, data, time_slice)


def plot_qp_costs(osc_debug, time_slice):
    cost_keys = ['input_cost', 'acceleration_cost',
                 'soft_constraint_cost']
    ps = plot_styler.PlotStyler()
    plotting_utils.make_plot(
        osc_debug,
        't_osc',
        time_slice,
        cost_keys,
        {},
        {key: [key] for key in cost_keys},
        {'xlabel': 'Time',
         'ylabel': 'Cost',
         'title': 'OSC QP Costs'}, ps)
    return ps


def plot_qp_solve_time(osc_debug, time_slice):
    ps = plot_styler.PlotStyler()
    plotting_utils.make_plot(
        osc_debug,
        't_osc',
        time_slice,
        ['qp_solve_time'],
        {},
        {},
        {'xlabel': 'Timestamp',
         'ylabel': 'Solve Time ',
         'title': 'OSC QP Solve Time'}, ps)
    return ps


def plot_lambda_c_sol(osc_debug, time_slice, lambda_slice):
    ps = plot_styler.PlotStyler()
    plotting_utils.make_plot(
        osc_debug,
        't_osc',
        time_slice,
        ['lambda_c_sol'],
        {'lambda_c_sol': lambda_slice},
        {'lambda_c_sol': ['lambda_c_' + i for i in
                          plotting_utils.slice_to_string_list(lambda_slice)]},
        {'xlabel': 'time',
         'ylabel': 'lambda',
         'title': 'OSC contact force solution'}, ps)
    return ps


def plot_epsilon_sol(osc_debug, time_slice, epsilon_slice):
    ps = plot_styler.PlotStyler()
    plotting_utils.make_plot(
        osc_debug,
        't_osc',
        time_slice,
        ['epsilon_sol'],
        {'epsilon_sol': epsilon_slice},
        {'epsilon_sol': ['epsilon_sol' + i for i in
                         plotting_utils.slice_to_string_list(epsilon_slice)]},
        {'xlabel': 'time',
         'ylabel': 'epsilon',
         'title': 'OSC soft constraint epsilon sol'}, ps)
    return ps


def add_fsm_to_plot(ps, fsm_time, fsm_signal):
    ax = ps.fig.axes[0]
    ymin, ymax = ax.get_ylim()

    # uses default color map
    for i in np.unique(fsm_signal):
        ax.fill_between(fsm_time, ymin, ymax, where=(fsm_signal == i), alpha=0.2)
    ax.relim()