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


    # Optionally: A method called after an experiment.
    # May be used to print some statistics about controller performance (e.g. number of iter. to converge)
    def controller_report(self):
        pass

    # Optionally: reset the controller after an experiment
    # May be useful for stateful controllers, like these containing RNN,
    # To reload the hidden states e.g. if the controller went unstable in the previous run.
    # It is called after an experiment,
    # but only if the controller is supposed to be reused without reloading (e.g. in GUI)
    def controller_reset(self):
        pass
