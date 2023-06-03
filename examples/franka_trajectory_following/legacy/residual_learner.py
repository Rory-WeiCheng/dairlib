# pydrake
from pydrake.systems.framework import Context, LeafSystem
from pydrake.common.value import AbstractValue

# pydairlib
from pydairlib.solvers.lcs import LCS
from pydairlib.systems.framework import OutputVector

# others
import numpy as np

class ResidualLearner(LeafSystem):
    def __init__(self):
        """
            Constructor for the ResidualLearner, currently fix the dimension of the problem,
            can be modified in the future to make it linked with external problem settings
        """
        LeafSystem.__init__(self)

        # problem dimensions
        self.num_state = 19
        self.num_velocity = 9
        self.num_control = 3
        self.num_lambda =12

        # port number definition
        # input port
        self.state_input_port = self.DeclareVectorInputPort("x,u,t",OutputVector(14,13,7)).get_index()
        # output port
        self.lcs_output_port = self.DeclareAbstractOutputPort("Residual LCS", lambda: AbstractValue.Make(LCS()),
                                                              self.CalcResidual).get_index()
        OutputPort = self.get_output_port()
        print(OutputPort.get_data_type())


    def CalcResidual(self, context:Context, residual_lcs:LCS):
        # get data from input port
        robot_output = self.EvalVectorInput(context, self.state_input_port)
        timestamp = robot_output.get_timestamp()

        N = 1
        A = np.zeros((self.num_velocity, self.num_state))
        B = np.zeros((self.num_velocity, self.num_control))
        D = np.zeros((self.num_velocity, self.num_lambda))
        d = np.zeros((self.num_velocity,1))

        E = np.zeros((self.num_lambda, self.num_state))
        F = np.zeros((self.num_lambda, self.num_lambda))
        H = np.zeros((self.num_lambda, self.num_control))
        c = np.zeros((self.num_lambda, 1))

        residual_lcs = LCS(A, B, D, d, E, F, H, c, N)





