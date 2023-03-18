import lcm
import matplotlib.pyplot as plt
import scipy.io
import dairlib.lcmt_robot_output
import drake
import numpy as np
from examples.franka_trajectory_following.scripts.franka_logging_utils_test import get_most_recent_logs
import os

def get_log_data(lcm_log, lcm_channels, end_time, data_processing_callback, *args,
                 **kwargs):
    """
    Parses an LCM log and returns data as specified by a callback function
    :param lcm_log: an lcm.EventLog object
    :param lcm_channels: dictionary with entries {channel : lcmtype} of channels
    to be read from the log
    :param data_processing_callback: function pointer which takes as arguments
     (data, args, kwargs) where data is a dictionary with
     entries {CHANNEL : [ msg for lcm msg in log with msg.channel == CHANNEL ] }
    :param args: positional arguments for data_processing_callback
    :param kwargs: keyword arguments for data_processing_callback
    :return: return args of data_processing_callback
    """

    data_to_process = {}
    print('Processing LCM log (this may take a while)...')
    t = lcm_log.read_next_event().timestamp
    lcm_log.seek(0)
    for event in lcm_log:
        if event.channel in lcm_channels:
            if event.channel in data_to_process:
                data_to_process[event.channel].append(
                    lcm_channels[event.channel].decode(event.data))
            else:
                data_to_process[event.channel] = \
                    [lcm_channels[event.channel].decode(event.data)]

        if event.eventnum % 50000 == 0:
            print(f'processed {(event.timestamp - t)*1e-6:.1f}'
                  f' seconds of log data')

        if 0 < end_time <= (event.timestamp - t)*1e-6:
            break
    return data_processing_callback(data_to_process, *args, *kwargs)


def get_log_summary(lcm_log):
    channel_names_and_msg_counts = {}
    for event in lcm_log:
        if event.channel not in channel_names_and_msg_counts:
            channel_names_and_msg_counts[event.channel] = 1
        else:
            channel_names_and_msg_counts[event.channel] = \
            channel_names_and_msg_counts[event.channel] + 1
    return channel_names_and_msg_counts


def print_log_summary(filename, log):
    print(f"Channels in {filename}:\n")
    summary = get_log_summary(log)
    for channel, count in summary.items():
        print(f"{channel}: {count:06} messages")


def passthrough_callback(data, *args, **kwargs):
    return data


channels = {
    "FRANKA_OUTPUT": dairlib.lcmt_robot_output,
    "FRANKA_STATE_ESTIMATE":dairlib.lcmt_robot_output,
    "CONTACT_RESULTS": drake.lcmt_contact_results_for_viz
}

'''
def processing_callback_old(data, channel):
    """
    Channel = FRANKA OUTPUT:
    position 14 (7 joint + 7 (4R+3T) ball, quaternion: qw, qx, qy, qz)
    velocity 13 (7 joint + 6 (3R+3T) ball)
    effort 7 (7 joints)
    acceleration (3)
    Channel = FRANKA_STATE_ESTIMATE:
    Channel = CONTACT_RESULTS:
    """
    if channel == "FRANKA_OUTPUT":
        # initialize the empty list to record the data
        for i in range(14):
            exec('p{} = []'.format(i), globals())
        for i in range(13):
            exec('v{} = []'.format(i), globals())
        for i in range(7):
            exec('u{} = []'.format(i), globals())
        for i in range(3):
            exec('a{} = []'.format(i), globals())
        print(len(data[channel]))
        for msg in data[channel]:
            for i in range(len(msg.position)):
                exec('p{}.append(msg.position[{}])'.format(i, i))
            for i in range(len(msg.velocity)):
                exec('v{}.append(msg.velocity[{}])'.format(i, i))
            for i in range(len(msg.effort)):
                exec('u{}.append(msg.effort[{}])'.format(i, i))
            for i in range(len(msg.imu_accel)):
                exec('a{}.append(msg.imu_accel[{}])'.format(i, i))

        return p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, \
            v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, \
            u0, u1, u2, u3, u4, u5, u6, \
            a0, a1, a2

    if channel == "CONTACT_RESULTS":
        contact_force_eeball = []
        contact_point_eeball = []
        contact_normal_eeball = []
        contact_force_ballg = []
        contact_point_ballg = []
        contact_normal_ballg = []
        print(len(data[channel]))
        for msg in data[channel]:
            contact_info = msg.point_pair_contact_info
            num_contact = len(contact_info)
            if num_contact == 0:
                contact_force_eeball.append((0.0, 0.0, 0.0))
                contact_point_eeball.append((0.0, 0.0, 0.0))
                contact_normal_eeball.append((0.0, 0.0, 0.0))
                contact_force_ballg.append((0.0, 0.0, 0.0))
                contact_point_ballg.append((0.0, 0.0, 0.0))
                contact_normal_ballg.append((0.0, 0.0, 0.0))
            if num_contact == 1:
                # print('Contact')
                # print(contact_info[0].body1_name)
                # print(contact_info[0].body2_name)
                contact_force_eeball.append((0.0, 0.0, 0.0))
                contact_point_eeball.append((0.0, 0.0, 0.0))
                contact_normal_eeball.append((0.0, 0.0, 0.0))
                contact_force_ballg.append(contact_info[0].contact_force)
                contact_point_ballg.append(contact_info[0].contact_point)
                contact_normal_ballg.append(contact_info[0].normal)
            if num_contact == 2:
                # print('Exist second contact')
                # for i in range(num_contact):
                #     print("Contact_{}".format(i))
                #     print(contact_info[i].body1_name)
                #     print(contact_info[i].body2_name)
                contact_force_eeball.append(contact_info[0].contact_force)
                contact_point_eeball.append(contact_info[0].contact_point)
                contact_normal_eeball.append(contact_info[0].normal)
                contact_force_ballg.append(contact_info[1].contact_force)
                contact_point_ballg.append(contact_info[1].contact_point)
                contact_normal_ballg.append(contact_info[1].normal)
    return contact_force_eeball,contact_point_eeball,contact_normal_eeball,\
           contact_force_ballg,contact_point_ballg,contact_normal_ballg
'''

def processing_callback(data, channel):
    '''
    Channel = FRANKA OUTPUT:
        position 14 (7 joint + 7 (4R+3T) ball, quaternion: qw, qx, qy, qz)
        velocity 13 (7 joint + 6 (3R+3T) ball)
        effort 7 (7 joints)
        acceleration (3)
    Channel = FRANKA_STATE_ESTIMATE:
    Channel = CONTACT_RESULTS:
        result at each time instant would differ from each other due to breaking and making contact

    '''
    if channel == "FRANKA_OUTPUT":
        # initialize the empty list to record the data
        position = []
        velocity = []
        effort = []
        acceleration = []
        t = []

        for msg in data[channel]:
            # for each time instant, stack up the corresponding data
            # t is in microseconds
            t.append(msg.utime)
            # append each time instant to form final list to be converted to numpy
            position.append(np.array(msg.position))
            velocity.append(np.array(msg.velocity))
            effort.append(np.array(msg.effort))
            acceleration.append(np.array(msg.imu_accel))

        position = np.vstack(position).T
        velocity = np.vstack(velocity).T
        effort = np.vstack(effort).T
        acceleration = np.vstack(acceleration).T
        timestamp_state = np.array(t)

        return position, velocity, effort ,acceleration, timestamp_state

    if channel == "CONTACT_RESULTS":
        contact_force_eeball = []
        contact_point_eeball = []
        contact_normal_eeball = []
        contact_force_ballg = []
        contact_point_ballg = []
        contact_normal_ballg = []
        t = []
        print(len(data[channel]))
        for msg in data[channel]:
            contact_info = msg.point_pair_contact_info
            num_contact = len(contact_info)
            # no contact, directly set as 0 for convenience of data processing
            if num_contact == 0:
                contact_force_eeball.append(np.array([0.0, 0.0, 0.0]))
                contact_point_eeball.append(np.array([0.0, 0.0, 0.0]))
                contact_normal_eeball.append(np.array([0.0, 0.0, 0.0]))
                contact_force_ballg.append(np.array([0.0, 0.0, 0.0]))
                contact_point_ballg.append(np.array([0.0, 0.0, 0.0]))
                contact_normal_ballg.append(np.array([0.0, 0.0, 0.0]))
            # 1 contact, record according to the name
            # though for this simulation, ground_ball always exists and ee_ball would
            # always be prior to ground_ball
            if num_contact == 1:
                if (contact_info[0].body1_name == 'panda_link10(2)') and (contact_info[0].body2_name == 'sphere(3)'):
                    contact_force_eeball.append(np.array(contact_info[0].contact_force))
                    contact_point_eeball.append(np.array(contact_info[0].contact_point))
                    contact_normal_eeball.append(np.array(contact_info[0].normal))
                    contact_force_ballg.append(np.array([0.0, 0.0, 0.0]))
                    contact_point_ballg.append(np.array([0.0, 0.0, 0.0]))
                    contact_normal_ballg.append(np.array([0.0, 0.0, 0.0]))
                else:
                    contact_force_eeball.append(np.array([0.0, 0.0, 0.0]))
                    contact_point_eeball.append(np.array([0.0, 0.0, 0.0]))
                    contact_normal_eeball.append(np.array([0.0, 0.0, 0.0]))
                    contact_force_ballg.append(np.array(contact_info[0].contact_force))
                    contact_point_ballg.append(np.array(contact_info[0].contact_point))
                    contact_normal_ballg.append(np.array(contact_info[0].normal))
            # 2 contact, record according to the name
            if num_contact == 2:
                if (contact_info[0].body1_name == 'panda_link10(2)') and (contact_info[0].body2_name == 'sphere(3)'):
                    contact_force_eeball.append(np.array(contact_info[0].contact_force))
                    contact_point_eeball.append(np.array(contact_info[0].contact_point))
                    contact_normal_eeball.append(np.array(contact_info[0].normal))
                    contact_force_ballg.append(np.array(contact_info[1].contact_force))
                    contact_point_ballg.append(np.array(contact_info[1].contact_point))
                    contact_normal_ballg.append(np.array(contact_info[1].normal))
                else:
                    print('2 contact yes')
                    contact_force_eeball.append(np.array(contact_info[1].contact_force))
                    contact_point_eeball.append(np.array(contact_info[1].contact_point))
                    contact_normal_eeball.append(np.array(contact_info[1].normal))
                    contact_force_ballg.append(np.array(contact_info[0].contact_force))
                    contact_point_ballg.append(np.array(contact_info[0].contact_point))
                    contact_normal_ballg.append(np.array(contact_info[0].normal))
            # t is in microseconds
            t.append(msg.timestamp)
        timestamp_contact = np.array(t)
        contact_force_eeball = np.vstack(contact_force_eeball).T
        contact_point_eeball = np.vstack(contact_point_eeball).T
        contact_normal_eeball = np.vstack(contact_normal_eeball).T
        contact_force_ballg = np.vstack(contact_force_ballg).T
        contact_point_ballg = np.vstack(contact_point_ballg).T
        contact_normal_ballg = np.vstack(contact_normal_ballg).T
    return contact_force_eeball,contact_point_eeball,contact_normal_eeball,\
           contact_force_ballg,contact_point_ballg,contact_normal_ballg, timestamp_contact



def main():

    # Wei-Cheng: 2023.1.31 modified local path to test the data recording
    logdir, log_num = get_most_recent_logs()
    if log_num is None:
        print("Did not find logs in {}".format(logdir))
        return
    os.chdir('{}/{}'.format(logdir, log_num))
    # logfile = "/usr/rory-workspace/data/test_log"
    # logfile = "/home/alpaydinoglu/workspace/dairlib/example_log"
    logfile = "lcmlog-{}".format(log_num)
    log = lcm.EventLog(logfile, "r")
    print_log_summary(logfile, log)

    log = lcm.EventLog(logfile, "r")
    # get the states and inputs
    position, velocity, effort ,acceleration, timestamp_state = get_log_data(log, channels, -1, processing_callback, "FRANKA_OUTPUT")

    # # briefly check the result
    # plt.plot(position[11], position[12])
    # circle2 = plt.Circle((0.55, 0), 0.1, color='b', fill=False)
    # plt.gca().add_patch(circle2)
    # plt.show()

    # get the contact result
    log = lcm.EventLog(logfile, "r")
    contact_force_eeball, contact_point_eeball, contact_normal_eeball, contact_force_ballg,\
    contact_point_ballg, contact_normal_ballg, timestamp_contact = get_log_data(log, channels, -1, processing_callback, "CONTACT_RESULTS")

    # # briefly check the result
    # plt.plot(contact_force_eeball[2,:])
    # plt.show()
    # plt.plot(contact_force_ballg[2,:])
    # plt.show()

    # Creating matfile old dictionary
    '''
    print("creating mat file")
    mdic_state_input = {"q0": p0, "q1": p1,"q2": p2, "q3": p3,"q4": p4, "q5": p5,"q6": p6, \
            "qw_b": p7,"qx_b": p8, "qy_b": p9, "qz_b": p10, "x_b":p11, "y_b":p12, "z_b":p13, \
            "qdot0": v0, "qdot1": v1, "qdot2": v2, "qdot3": v3, "qdot4": v4, "qdot5": v5, "qdot6": v6, \
            "wx_b": v7, "wy_b": v8, "wz_b": v9, "xdot_b": v10, "ydot_b": v11, "zdot_b": v12, \
            "u0": u0, "u1": u1, "u2": u2, "u3": u3, "u4": u4, "u5": u5, "u6": u6, \
            "ax": a0, "ay": a1, "az": a2,
            }
    mat_file = "State_Input-{}.mat".format(log_num)
    scipy.io.savemat(mat_file, mdic_state_input)
    '''

    # Creating save file for states and inputs
    print("creating state/input mat and npz files")
    mdic_state_input = {"q": position[0:7,:],"R_b": position[7:11,:],"p_b": position[11:,:],"q_dot":velocity[0:7,:],\
                             "w_b": velocity[7:10,:],"v_b": velocity[10:,:], "u":effort, "a":acceleration, "timestamp_state": timestamp_state}
    # save as npz file (numpy array)
    mat_file = "State_Input-{}.mat".format(log_num)
    scipy.io.savemat(mat_file, mdic_state_input)
    npz_file = "State_Input-{}.npz".format(log_num)
    np.savez(npz_file,**mdic_state_input)
    print("finished creating state/input file")

    # Creating save file for contact forces
    print("creating contact force mat and npz files")
    mdic_contact ={"f_eeball":contact_force_eeball,"p_eeball":contact_point_eeball,"n_eeball":contact_normal_eeball,\
                   "f_ballg":contact_force_ballg,"p_ballg":contact_point_ballg,"n_ballg":contact_normal_ballg, "timestamp_contact":timestamp_contact}
    mat_file = "Contact_Info-{}.mat".format(log_num)
    scipy.io.savemat(mat_file, mdic_contact)
    npz_file = "Contact_Info-{}.npz".format(log_num)
    np.savez(npz_file, **mdic_contact)

    print("finished creating contact force file")


if __name__ == "__main__":
    main()