import subprocess
import os
import codecs
import sys
from franka_logging_utils_test import *

# data logging script for simulating c3 on a single laptop, largely refer to adam wei's codes
# basically is to first create a new directory and call the lcm-logger to get the log file.
def main(argv):
    # start logging
    # dair = str(os.getenv('DAIR_PATH'))
    dair = "/usr/rory-workspace/dairlib"
    # make new directory: /usr/rory-workspace/data/experiment_logs/yy/mmdd/tt, see ranka_logging_utils_test
    logdir, log_num = create_new_log()
    os.chdir('{}/{}'.format(logdir, log_num))
    # call command to record the corresponding simulation parameters
    parameters_path = "{}/examples/franka_trajectory_following/parameters.yaml".format(dair)
    subprocess.call(['cp', parameters_path, 'parameters{}.yaml'.format(log_num)])
    # call lcm-logger command to record the data
    subprocess.call(['lcm-logger', '-f', 'lcmlog-%s' % log_num])

if __name__ == '__main__':
    main(sys.argv)