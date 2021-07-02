from types import SimpleNamespace
from typing import Union

import numpy as np

from CartPole.cartpole_model import _cartpole_ode
from CartPole.state_utilities import (
    create_cartpole_state, cartpole_state_varname_to_index,
    ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX
)
from others.p_globals import (
    k as param_k,
    M as param_M,
    m as param_m,
    g as param_g,
    J_fric as param_J_fric,
    M_fric as param_M_fric,
    L as param_L,
    CARTPOLE_EQUATIONS
)


import sympy as sym
from sympy.utilities.lambdify import lambdify, implemented_function


x, v, t, o, u = sym.symbols("x,v,t,o,u")
k, M, m, L, J_fric, M_fric, g = sym.symbols("k,M,m,L,J_fric,M_fric,g")

xD = v
tD = o
oD, vD, _, _ = _cartpole_ode(sym.cos(-t), sym.sin(-t), o, v, u)


xx = sym.diff(xD, x, 1)
xv = sym.diff(xD, v, 1)
xt = sym.diff(xD, t, 1)
xo = sym.diff(xD, o, 1)
xu = sym.diff(xD, u, 1)

vx = sym.diff(vD, x, 1)
vv = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(vD, v, 1), "numpy")
vt = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(vD, t, 1), "numpy")
vo = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(vD, o, 1), "numpy")
vu = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(vD, u, 1), "numpy")

tx = sym.diff(tD, x, 1)
tv = sym.diff(tD, v, 1)
tt = sym.diff(tD, t, 1)
to = sym.diff(tD, o, 1)
tu = sym.diff(tD, u, 1)

ox = sym.diff(oD, x, 1)
ov = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(oD, v, 1), "numpy")
ot = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(oD, t, 1), "numpy")
oo = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(oD, o, 1), "numpy")
ou = lambdify((x, v, t, o, u, k, M, m, L, J_fric, M_fric, g), sym.diff(oD, u, 1), "numpy")


def cartpole_jacobian(s: Union[np.ndarray, SimpleNamespace], u: float):
    """
    Jacobian of cartpole ode with the following structure:

        # ______________|    position     |   positionD    | angle | angleD |       u       |
        # position  (x) |   xx -> J[0,0]        xv            xt       xo      xu -> J[0,4]
        # positionD (v) |       vx              vv            vt       vo         vu
        # angle     (t) |       tx              tv            tt       to         tu
        # angleD    (o) |   ox -> J[3,0]        ov            ot       oo      ou -> J[3,4]
    
    :param s: State vector following the globally defined variable order
    :param u: Force applied on cart in unnormalized range

    The Jacobian is used to linearize the CartPole dynamics around the origin

    :returns: A 4x5 numpy.ndarray with all partial derivatives
    """
    if isinstance(s, np.ndarray):
        angle = s[cartpole_state_varname_to_index('angle')]
        angleD = s[cartpole_state_varname_to_index('angleD')]
        position = s[cartpole_state_varname_to_index('position')]
        positionD = s[cartpole_state_varname_to_index('positionD')]
    elif isinstance(s, SimpleNamespace):
        angle = s.angle
        angleD = s.angleD
        position = s.position
        positionD = s.positionD
    
    J = np.zeros(shape=(4, 5), dtype=np.float32)  # Array to keep Jacobian
    ca = np.cos(angle)
    sa = np.sin(angle)

    if CARTPOLE_EQUATIONS == 'Marcin-Sharpneat':
        # Jacobian entries
        J[0, 0] = 0.0  # xx

        J[0, 1] = 1.0  # xv

        J[0, 2] = 0.0  # xt

        J[0, 3] = 0.0  # xo

        J[0, 4] = 0.0  # xu

        J[1, 0] = 0.0  # vx

        J[1, 1] = vv(position, positionD, angle, angleD, u, param_k, param_M, param_m, param_L, param_J_fric, param_M_fric, param_g)

        J[1, 2] = vt(position, positionD, angle, angleD, u, param_k, param_M, param_m, param_L, param_J_fric, param_M_fric, param_g)

        J[1, 3] = vo(position, positionD, angle, angleD, u, param_k, param_M, param_m, param_L, param_J_fric, param_M_fric, param_g)

        J[1, 4] = vu(position, positionD, angle, angleD, u, param_k, param_M, param_m, param_L, param_J_fric, param_M_fric, param_g)

        J[2, 0] = 0.0  # tx

        J[2, 1] = 0.0  # tv

        J[2, 2] = 0.0  # tt

        J[2, 3] = 1.0  # to

        J[2, 4] = 0.0  # tu

        J[3, 0] = 0.0  # ox

        J[3, 1] = ov(position, positionD, angle, angleD, u, param_k, param_M, param_m, param_L, param_J_fric, param_M_fric, param_g)

        J[3, 2] = ot(position, positionD, angle, angleD, u, param_k, param_M, param_m, param_L, param_J_fric, param_M_fric, param_g)

        J[3, 3] = oo(position, positionD, angle, angleD, u, param_k, param_M, param_m, param_L, param_J_fric, param_M_fric, param_g)

        J[3, 4] = ou(position, positionD, angle, angleD, u, param_k, param_M, param_m, param_L, param_J_fric, param_M_fric, param_g)

        return J


s0 = create_cartpole_state()

if __name__ == '__main__':
    import timeit
    """
    On 9.02.2021 we saw a perfect coincidence (5 digits after coma) of Jacobian from Mathematica cartpole_model.nb
    with Jacobian calculated with this script for all non zero inputs, dtype=float32
    """
    # Set non-zero input
    s = s0
    s[cartpole_state_varname_to_index('position')] = -30.2
    s[cartpole_state_varname_to_index('positionD')] = 2.87
    s[cartpole_state_varname_to_index('angle')] = -0.32
    s[cartpole_state_varname_to_index('angleD')] = 0.237
    u = -0.24

    # Calculate time necessary for evaluation of a Jacobian:

    f_to_measure = 'Jacobian = cartpole_jacobian(s, u)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 100000 # Gives how many times timeit should be repeated
    timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
    min_time = min(timings)/float(number)
    max_time = max(timings)/float(number)
    average_time = np.mean(timings)/float(number)
    print('Min time to calculate Jacobian is {} us'.format(min_time * 1.0e6))  # ca. 14 us
    print('Average time to calculate Jacobian is {} us'.format(average_time*1.0e6))  # ca 16 us
    print('Max time to calculate Jacobian is {} us'.format(max_time * 1.0e6))          # ca. 150 us

    # Calculate once more to prrint the resulting matrix
    Jacobian = np.around(cartpole_jacobian(s, u), decimals=6)

    print()
    print(Jacobian.dtype)
    print(Jacobian)