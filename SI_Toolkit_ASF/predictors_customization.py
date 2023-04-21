import numpy as np

from CartPole.state_utilities import STATE_INDICES, STATE_VARIABLES, CONTROL_INPUTS, create_cartpole_state

from CartPole.cartpole_model import Q2u, L
from CartPole.cartpole_numba import cartpole_fine_integration_s_numba

class next_state_predictor_ODE():

    def __init__(self,
                 dt: float,
                 intermediate_steps: int,
                 batch_size: int,
                 variable_parameters=None,
                 **kwargs):
        self.s = create_cartpole_state()

        self.variable_parameters = variable_parameters

        self.intermediate_steps = intermediate_steps
        self.t_step = np.float32(dt / float(self.intermediate_steps))
        
    def step(self, s, Q):

        assert Q.shape[0] == s.shape[0]
        assert Q.ndim == 2
        assert s.ndim == 2

        if self.variable_parameters is not None and hasattr(self.variable_parameters, 'L'):
            pole_half_length = self.variable_parameters.L
        else:
            pole_half_length = L

        Q = np.squeeze(Q, axis=1)  # Removes features dimension, specific for cartpole as it has only one control input
        u = Q2u(Q)
        s_next = cartpole_fine_integration_s_numba(s, u, self.t_step, self.intermediate_steps, L=pole_half_length)
        return s_next


def augment_predictor_output(output_array, net_info):

    if 'angle' not in net_info.outputs:
        output_array[..., STATE_INDICES['angle']] = \
            np.arctan2(
                output_array[..., STATE_INDICES['angle_sin']],
                output_array[..., STATE_INDICES['angle_cos']])
    if 'angle_sin' not in net_info.outputs:
        output_array[..., STATE_INDICES['angle_sin']] = \
            np.sin(output_array[..., STATE_INDICES['angle']])
    if 'angle_cos' not in net_info.outputs:
        output_array[..., STATE_INDICES['angle_cos']] = \
            np.sin(output_array[..., STATE_INDICES['angle']])

    return output_array
