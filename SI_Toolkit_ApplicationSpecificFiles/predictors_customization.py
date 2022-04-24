import numpy as np

from CartPole.state_utilities import STATE_INDICES, STATE_VARIABLES, CONTROL_INPUTS, CONTROL_INDICES, create_cartpole_state
from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX

from CartPole.cartpole_model import Q2u, L
from CartPole.cartpole_numba import cartpole_fine_integration_numba

class next_state_predictor_ODE():
    
    def __init__(self, dt, intermediate_steps):
        self.s = create_cartpole_state()

        self.intermediate_steps = intermediate_steps
        self.t_step = np.float32(dt / float(self.intermediate_steps))
        
    def step(self, s, Q, params):

        assert Q.shape[0] == s.shape[0]
        assert Q.ndim == 2
        assert s.ndim == 2

        s_next = np.zeros_like(s)

        if params is None:
            pole_half_length = L
        else:
            pole_half_length = params

        Q = np.squeeze(Q, axis=1)  # Removes features dimension, specific for cartpole as it has only one control input

        u = Q2u(Q)
    
        (
            s_next[..., ANGLE_IDX], s_next[..., ANGLED_IDX], s_next[..., POSITION_IDX], s_next[..., POSITIOND_IDX], s_next[..., ANGLE_COS_IDX], s_next[..., ANGLE_SIN_IDX]
        ) = cartpole_fine_integration_numba(
            angle=s[..., ANGLE_IDX],
            angleD=s[..., ANGLED_IDX],
            angle_cos=s[..., ANGLE_COS_IDX],
            angle_sin=s[..., ANGLE_SIN_IDX],
            position=s[..., POSITION_IDX],
            positionD=s[..., POSITIOND_IDX],
            u=u,
            t_step=self.t_step,
            intermediate_steps=self.intermediate_steps,
            L=pole_half_length,
        )

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
            np.cos(output_array[..., STATE_INDICES['angle']])

    return output_array
