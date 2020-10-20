"""do-mpc controller"""

from casadi.tools import *
import do_mpc
import time

import casadi

from controllers.do_mpc_files.template_mpc import template_mpc
from controllers.do_mpc_files.template_simulator import template_simulator
from controllers.do_mpc_files.template_model import template_model

from globals import *
from types import SimpleNamespace

from copy import deepcopy



class controller_do_mpc:
    def __init__(self,
                 store_results=False,
                 # Define obstacles to avoid (cycles)
                 obstacles=[
                     {},
                 ],
                 CartPosition_init=0.0,
                 CartPositionD_init=0.0,
                 angle_init=0.0,
                 angleD_init=0.0,
                 ):
        self.store_results = store_results
        self.obstacles = obstacles

        """
        Get configured do-mpc modules:
        """

        # Physical parameters of the cart
        p = SimpleNamespace()  # p like parameters
        p.m = m_globals  # mass of pend, kg
        p.M = M_globals  # mass of cart, kg
        p.L = L_globals  # half length of pend, m
        p.u_max = u_max_globals  # max cart force, N
        p.M_fric = M_fric_globals  # cart friction, N/m/s
        p.J_fric = J_fric_globals  # friction coefficient on angular velocity, Nm/rad/s
        p.v_max = v_max_globals  # max DC motor speed, m/s, in absence of friction, used for motor back EMF model
        p.controlDisturbance = controlDisturbance_globals  # disturbance, as factor of u_max
        p.sensorNoise = sensorNoise_globals  # noise, as factor of max values
        p.g = g_globals  # gravity, m/s^2
        p.k = k_globals  # Dimensionless factor, for moment of inertia of the pend (with L being half if the length)

        PositionTarget = -20.0

        # State of the cart
        s = SimpleNamespace()  # s like state
        s.CartPosition = 0.0
        s.CartPositionD = 0.0
        s.CartPositionDD = 0.0
        s.angle = 0.0
        s.angleD = 0.0
        s.angleDD = 0.0

        model_type = 'continuous'  # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)

        s.CartPosition = self.model.set_variable(var_type='_x', var_name='s.CartPosition', shape=(1, 1))
        s.CartPositionD = self.model.set_variable(var_type='_x', var_name='s.CartPositionD', shape=(1, 1))

        s.angle = self.model.set_variable(var_type='_x', var_name='s.angle', shape=(1, 1))
        s.angleD = self.model.set_variable(var_type='_x', var_name='s.angleD', shape=(1, 1))

        Q = self.model.set_variable(var_type='_u', var_name='Q')

        self.model.set_rhs('s.CartPosition', s.CartPositionD)
        self.model.set_rhs('s.angle', s.angleD)

        angleD_next, CartPositionD_next = cartpole_ode(p, s, Q2u(Q,p))

        self.model.set_rhs('s.CartPositionD', CartPositionD_next)
        self.model.set_rhs('s.angleD', angleD_next)


        # TODO resolve problem of u_eff vs. Q
        # Optimally I would use in the equations only u_eff Q beeing the input only for slider.

        # Expressions for kinetic and potential energy

        # Simplified, normalized expressions for E_kin and E_pot as a port of cost function
        E_kin_cart = (s.CartPositionD/p.v_max)**2
        E_kin_pol = (s.angleD/(2*np.pi))**2
        E_pot = np.cos(s.angle)

        # Positive means that during the procedure the distance to target increased, negative that it decreased
        # Idea good, but implementation does not work - need to implement s.CartPositionInitial as a time varying parameter
        # distance_difference = ((s.CartPosition-PositionTarget)**2) - ((CartPositionInitial-PositionTarget)**2)
        distance_difference = ((s.CartPosition - PositionTarget) ** 2)

        self.model.set_expression('E_kin_cart', E_kin_cart)
        self.model.set_expression('E_kin_pol', E_kin_pol)
        self.model.set_expression('E_pot', E_pot)
        self.model.set_expression('distance_difference', distance_difference)


        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)

        setup_mpc = {
            'n_horizon': 20,
            't_step': 0.02,
            'n_robust': 0,
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)
        self.mpc.set_param(nlpsol_opts = {'ipopt.linear_solver': 'mumps'})

        # mterm = 0.05*self.model.aux['E_kin'] - self.model.aux['E_pot']
        # lterm = -self.model.aux['E_pot'] + 10 * (
        #             (self.model.x['s.CartPosition'] - PositionTarget) / 100.0) ** 2  # stage cost

        lterm = - self.model.aux['E_pot']

        # lterm = 0.01*self.model.aux['E_kin'] - self.model.aux['E_pot'] + 0.1 * (
        #             (self.model.x['s.CartPosition'] - PositionTarget) / 100.0) ** 2
        mterm = 5*self.model.aux['E_kin_pol'] - 5*self.model.aux['E_pot'] + 0.01 * distance_difference

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(Q=0.1)

        self.mpc.bounds['lower', '_u', 'Q'] = -1.0
        self.mpc.bounds['upper', '_u', 'Q'] = 1.0

        self.mpc.setup()

        # Set initial state
        self.x0 = self.mpc.x0
        self.x0['s.CartPosition'] = CartPosition_init
        self.x0['s.CartPositionD'] = CartPositionD_init
        self.x0['s.angle'] = angle_init
        self.x0['s.angleD'] = angleD_init


        self.mpc.x0 = self.x0

        self.mpc.set_initial_guess()


    def step(self, state):

        s = deepcopy(state)

        self.x0['s.CartPosition'] = s.CartPosition
        self.x0['s.CartPositionD'] = s.CartPositionD

        self.x0['s.angle'] = s.angle
        self.x0['s.angleD'] = s.angleD


        Q = self.mpc.make_step(self.x0)

        return Q.item()


    # Store results
    def store_mpc_results(self):
        do_mpc.data.save_results([self.mpc, self.simulator], 'dip_mpc')


