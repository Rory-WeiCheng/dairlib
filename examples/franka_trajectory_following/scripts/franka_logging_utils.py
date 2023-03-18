import os
import glob
from datetime import date

# Wei-Cheng 2023.1.29 adjust for local laptop
default_log_root = "/usr/rory-workspace/data/experiment_logs".format(os.getenv('HOME'))
#default_log_root = "/home/alpaydinoglu/Desktop/experiment_logs/08_30_22".format(os.getenv('HOME'))
#default_log_root = "/hdd_storage/c3-experiments/logs".format(os.getenv('HOME'))


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


  #logdir = "/home/alpaydinoglu/Desktop/experiment_logs/08_30_22".format(os.getenv('HOME'))

  #SQUARE
  log_num = "{:02}".format(7)

  #LINE
  #log_num = "{:02}".format(8)

  # #CIRCLE
  # log_num = "{:02}".format(11)

  #CIRCLE
  log_num = "{:02}".format(1)

  print("printing log_num")
  print(log_num)

  os.chdir(logdir)

  current_logs = sorted(glob.glob('*'))

  # if len(current_logs) == 0:
  #   log_num = None
  # elif current_logs[-1] == 'log_descriptions.txt':
  #   last_log = int(current_logs[-2])
  #   log_num = "{:02}".format(last_log)
  # else:
  #   last_log = int(current_logs[-1])
  #   log_num = "{:02}".format(last_log)




  return logdir, log_num