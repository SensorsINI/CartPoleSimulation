"""
This is a linear-quadratic regulator
It assumes that the input relation is u = Q*p.u_max (no fancy motor model) !
"""

from CartPole.cartpole_model import cartpole_jacobian, cartpole_ode, p_globals, s0


class template_controller:

    def __init__(self):
        pass

    def step(self, s, target_position, time=None):
        Q = None  # This line is not obligatory. ;-) Just to indicate that Q must me defined and returned
        pass
        return Q  # normed control input in the range [-1,1]
