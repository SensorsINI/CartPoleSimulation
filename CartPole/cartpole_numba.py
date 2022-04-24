from numba import float32, jit
import numpy as np
from CartPole.cartpole_model import _cartpole_ode, euler_step, edge_bounce
from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state
from CartPole._CartPole_mathematical_helpers import wrap_angle_rad_inplace

from others.p_globals import (
    k, M, m, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength
)

_cartpole_ode_numba = jit(_cartpole_ode, nopython=True, cache=True, fastmath=True)


euler_step_numba = jit(euler_step, nopython=True, cache=True, fastmath=True)


edge_bounce_numba = jit(edge_bounce, nopython=True, cache=True, fastmath=True)


wrap_angle_rad_inplace_numba = jit(wrap_angle_rad_inplace, nopython=True, cache=True, fastmath=True)

@jit(nopython=True, cache=True, fastmath=True)
def cartpole_ode_numba(s: np.ndarray, u: float,
                 k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L):
    angleDD, positionDD = _cartpole_ode_numba(
        s[..., ANGLE_COS_IDX], s[..., ANGLE_SIN_IDX], s[..., ANGLED_IDX], s[..., POSITIOND_IDX], u,
        k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L
    )
    return angleDD, positionDD

@jit(nopython=True, cache=True, fastmath=True)
def edge_bounce_wrapper_numba(angle, angle_cos, angleD, position, positionD, t_step, L=L):
    for i in range(position.size):
        angle[i], angleD[i], position[i], positionD[i] = edge_bounce_numba(angle[i], angle_cos[i], angleD[i], position[i], positionD[i],
                                                                     t_step, L)
    return angle, angleD, position, positionD

@jit(nopython=True, cache=True, fastmath=True)
def cartpole_integration_numba(angle, angleD, angleDD, position, positionD, positionDD, t_step, ):
    angle_next = euler_step_numba(angle, angleD, t_step)
    angleD_next = euler_step_numba(angleD, angleDD, t_step)
    position_next = euler_step_numba(position, positionD, t_step)
    positionD_next = euler_step_numba(positionD, positionDD, t_step)

    return angle_next, angleD_next, position_next, positionD_next

# @jit(nopython=True, cache=True, fastmath=True)  # This seems to make the function slower, I don't know why.
def cartpole_fine_integration_numba(angle, angleD, angle_cos, angle_sin, position, positionD, u, t_step, intermediate_steps,
                              k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L):
    for _ in range(intermediate_steps):
        # Find second derivative for CURRENT "k" step (same as in input).
        # State and u in input are from the same timestep, output is belongs also to THE same timestep ("k")
        angleDD, positionDD = _cartpole_ode_numba(angle_cos, angle_sin, angleD, positionD, u,
                                                  k, M, m, g, J_fric, M_fric, L)

        # Find NEXT "k+1" state [angle, angleD, position, positionD]
        angle, angleD, position, positionD = cartpole_integration_numba(angle, angleD, angleDD, position, positionD,
                                                                  positionDD, t_step, )

        angle_cos = np.cos(angle)
        angle, angleD, position, positionD = edge_bounce_wrapper_numba(angle, angle_cos, angleD, position, positionD, t_step, L)

        wrap_angle_rad_inplace_numba(angle)

        angle_cos = np.cos(angle)
        angle_sin = np.sin(angle)

    return angle, angleD, position, positionD, angle_cos, angle_sin


def cartpole_fine_integration_s_numba(s, u, t_step, intermediate_steps,
                              k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric, L=L):

    if s.ndim == 1:
        s = s[np.newaxis, :]

    s_next = np.zeros_like(s)

    (
        s_next[..., ANGLE_IDX], s_next[..., ANGLED_IDX], s_next[..., POSITION_IDX], s_next[..., POSITIOND_IDX],
        s_next[..., ANGLE_COS_IDX], s_next[..., ANGLE_SIN_IDX]
    ) = cartpole_fine_integration_numba(
        angle=s[..., ANGLE_IDX],
        angleD=s[..., ANGLED_IDX],
        angle_cos=s[..., ANGLE_COS_IDX],
        angle_sin=s[..., ANGLE_SIN_IDX],
        position=s[..., POSITION_IDX],
        positionD=s[..., POSITIOND_IDX],
        u=u,
        t_step=t_step,
        intermediate_steps=intermediate_steps,
        L=L,
        k=k, M=M, m=m, g=g, J_fric=J_fric, M_fric=M_fric
    )

    return s_next

if __name__ == '__main__':
    import timeit

    s0 = create_cartpole_state()

    # Set non-zero input
    s = s0
    s[POSITION_IDX] = -30.2
    s[POSITIOND_IDX] = 2.87
    s[ANGLE_IDX] = -0.32
    s[ANGLED_IDX] = 0.237
    u = -0.24

    # Calculate time necessary to evaluate cartpole ODE:

    f_to_measure = 'angleDD, positionDD = cartpole_ode_numba(s, u)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 100000  # Gives how many times timeit should be repeated
    timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
    min_time = min(timings) / float(number)
    max_time = max(timings) / float(number)
    average_time = np.mean(timings) / float(number)
    print()
    print('----------------------------------------------------------------------------------')
    print('Min time to evaluate ODE is {} us'.format(min_time * 1.0e6))  # ca. 5 us
    print('Average time to evaluate ODE is {} us'.format(average_time * 1.0e6))  # ca 5 us
    # The max is of little relevance as it is heavily influenced by other processes running on the computer at the same time
    print('Max time to evaluate ODE is {} us'.format(max_time * 1.0e6))  # ca. 100 us
    print('----------------------------------------------------------------------------------')
    print()