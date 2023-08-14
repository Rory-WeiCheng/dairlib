import sys
import ruamel.yaml
import os
from scipy import signal
import sys

def L(*l):
    # helper function to write compact list
    ret = ruamel.yaml.comments.CommentedSeq(l)
    ret.fa.set_flow_style()
    return ret

os.chdir(sys.path[0])
c3_param_path = 'parameters.yaml'
f = open(c3_param_path,"r")

yaml = ruamel.yaml.YAML()  # defaults to round-trip if no parameters given
c3_params = yaml.load(f.read())

# position
position_degree = c3_params['position_degree']
position_cutoff_freq = c3_params['position_cutoff_freq']
position_window = c3_params['position_window']

position_FIR_parameters = signal.firwin(position_degree + 1, position_cutoff_freq, window = position_window)
position_FIR_parameters = position_FIR_parameters.tolist()

# velocity
velocity_degree = c3_params['velocity_degree']
velocity_cutoff_freq = c3_params['velocity_cutoff_freq']
velocity_window = c3_params['velocity_window']

velocity_FIR_parameters = signal.firwin(velocity_degree + 1, velocity_cutoff_freq, window = velocity_window)
velocity_FIR_parameters = velocity_FIR_parameters.tolist()


c3_params['position_FIR_para'] = L()
for position_para in position_FIR_parameters:
    c3_params['position_FIR_para'].append(position_para)
c3_params['velocity_FIR_para'] = L()
for velocity_para in velocity_FIR_parameters:
    c3_params['velocity_FIR_para'].append(velocity_para)

# rewrite yaml
f = open(c3_param_path,"w")
yaml.dump(c3_params, f)