import os
import glob
from datetime import date

# Wei-Cheng 2023.1.29： Modify Adam's code for my laptop
default_log_root = "/usr/rory-workspace/data/experiment_logs".format(os.getenv('HOME'))

def create_new_log(log_root=default_log_root):
  ''' change this to change log location '''
  curr_date = date.today().strftime("%m_%d_%y")
  year = date.today().strftime("%Y")
  logdir = "{}/{}/{}".format(log_root, year, curr_date)
  
  if not os.path.isdir(logdir):
    os.makedirs(logdir)
  os.chdir(logdir)

  current_logs = sorted(glob.glob('*'))
  if len(current_logs) == 0:
    log_num = '00'
  elif current_logs[-1] == 'log_descriptions.txt':
    last_log = int(current_logs[-2])
    log_num = "{:02}".format(last_log+1)
  else:
    last_log = int(current_logs[-1])
    log_num = "{:02}".format(last_log+1)
  
  os.mkdir(log_num)

  return logdir, log_num

def get_most_recent_logs(log_root=default_log_root):
  ''' change this to change log location '''

  curr_date = date.today().strftime("%m_%d_%y")
  year = date.today().strftime("%Y")
  # Wei-Cheng 2023.1.29
  logdir = "{}/{}/{}".format(log_root, year, curr_date)

  os.chdir(logdir)
  current_logs = sorted(glob.glob('*'))

  if len(current_logs) == 0:
    log_num = None
  elif current_logs[-1] == 'log_descriptions.txt':
    last_log = int(current_logs[-2])
    log_num = "{:02}".format(last_log)
  else:
    last_log = int(current_logs[-1])
    log_num = "{:02}".format(last_log)

  return logdir, log_num