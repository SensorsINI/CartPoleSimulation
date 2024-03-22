from numba import jit
import numpy as np

from CartPole.cartpole_equations import edge_bounce_numba, _cartpole_ode_numba, cartpole_integration_numba

from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX
from CartPole._CartPole_mathematical_helpers import wrap_angle_rad_inplace


def cartpole_fine_integration_numba_interface(s, u, t_step, intermediate_steps, params, **kwargs):
    k = kwargs.get('k', params.k)
    m_cart = kwargs.get('m_cart', params.m_cart)
    m_pole = kwargs.get('m_pole', params.m_pole)
    g = kwargs.get('g', params.g)
    J_fric = kwargs.get('J_fric', params.J_fric)
    M_fric = kwargs.get('M_fric', params.M_fric)
    L = kwargs.get('L', params.L)

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
        k=k, m_cart=m_cart, m_pole=m_pole, g=g, J_fric=J_fric, M_fric=M_fric
    )

    return s_next


wrap_angle_rad_inplace_numba = jit(wrap_angle_rad_inplace, nopython=True, cache=True, fastmath=True)


@jit(nopython=True, cache=True, fastmath=True)
def edge_bounce_wrapper_numba(angle, angle_cos, angleD, position, positionD, t_step, L):
    for i in range(position.size):
        angle[i], angleD[i], position[i], positionD[i] = edge_bounce_numba(
            angle[i], angle_cos[i], angleD[i], position[i], positionD[i], t_step, L)
    return angle, angleD, position, positionD


# @jit(nopython=True, cache=True, fastmath=True)  # This seems to make the function slower, I don't know why.
def cartpole_fine_integration_numba(angle, angleD, angle_cos, angle_sin, position, positionD,
                                    u, t_step, intermediate_steps,
                                    k, m_cart, m_pole, g, J_fric, M_fric, L):
    for _ in range(intermediate_steps):
        # Find second derivative for CURRENT "k" step (same as in input).
        # State and u in input are from the same timestep, output is belongs also to THE same timestep ("k")
        angleDD, positionDD = _cartpole_ode_numba(angle_cos, angle_sin, angleD, positionD, u,
                                                  k, m_cart, m_pole, g, J_fric, M_fric, L)

        # Find NEXT "k+1" state [angle, angleD, position, positionD]
        angle, angleD, position, positionD = cartpole_integration_numba(angle, angleD, angleDD, position, positionD,
                                                                        positionDD, t_step, )

        angle_cos = np.cos(angle)
        angle, angleD, position, positionD = edge_bounce_wrapper_numba(angle, angle_cos, angleD, position, positionD,
                                                                       t_step, L)

        wrap_angle_rad_inplace_numba(angle)

        angle_cos = np.cos(angle)
        angle_sin = np.sin(angle)

    return angle, angleD, position, positionD, angle_cos, angle_sin
