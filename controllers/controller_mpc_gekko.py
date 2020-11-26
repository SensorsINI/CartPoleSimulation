"""

 Gekko based MPC controller
 Date created : 20/11/2020

 Copying the example to understand the working of GEKKO MPC
 TODO: Modify cartpole do_mpc to this format

"""

# Controler empty template. Pasted here by Marcin to prevent GUI crashing
class controller_mpc_gekko:
    def __init__(self):
        pass
    def step(self, s, target_position):
        Q = 0.0
        return Q








# from gekko import GEKKO
# import numpy as np
# import matplotlib.pyplot as plt
#
# m = GEKKO()
# m.time = np.linspace(0,20,41)
#
# # Parameters
# mass = 500
# b = m.Param(value=50)
# K = m.Param(value=0.8)
#
# # Manipulated variable
# p = m.MV(value=0, lb=0, ub=100)
# p.STATUS = 1  # allow optimizer to change
# p.DCOST = 0.1 # smooth out gas pedal movement
# p.DMAX = 20   # slow down change of gas pedal
#
# # Controlled Variable
# v = m.CV(value=0)
# v.STATUS = 1  # add the SP to the objective
# m.options.CV_TYPE = 2 # squared error
# v.SP = 40     # set point
# v.TR_INIT = 1 # set point trajectory
# v.TAU = 5     # time constant of trajectory
#
# # Process model
# m.Equation(mass*v.dt() == -v*b + K*b*p)
#
# m.options.IMODE = 6 # control
# m.solve(disp=False)
#
# # get additional solution information
# import json
# with open(m.path+'//results.json') as f:
#     results = json.load(f)
#
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(m.time,p.value,'b-',label='MV Optimized')
# plt.legend()
# plt.ylabel('Input')
# plt.subplot(2,1,2)
# plt.plot(m.time,results['v1.tr'],'k-',label='Reference Trajectory')
# plt.plot(m.time,v.value,'r--',label='CV Response')
# plt.ylabel('Output')
# plt.xlabel('Time')
# plt.legend(loc='best')
# plt.show()



# """do-mpc controller """


# import do_mpc

# from src.globals import *
# from types import SimpleNamespace

# from copy import deepcopy



# class controller_do_mpc:
#     def __init__(self,
#                  position_init=0.0,
#                  positionD_init=0.0,
#                  angle_init=0.0,
#                  angleD_init=0.0,
#                  ):

#         """
#         Get configured do-mpc modules:
#         """

#         # Physical parameters of the cart
#         p = SimpleNamespace()  # p like parameters
#         p.m = m_globals  # mass of pend, kg
#         p.M = M_globals  # mass of cart, kg
#         p.L = L_globals  # half length of pend, m
#         p.u_max = u_max_globals  # max cart force, N
#         p.M_fric = M_fric_globals  # cart friction, N/m/s
#         p.J_fric = J_fric_globals  # friction coefficient on angular velocity, Nm/rad/s
#         p.v_max = v_max_globals  # max DC motor speed, m/s, in absence of friction, used for motor back EMF model
#         p.controlDisturbance = controlDisturbance_globals  # disturbance, as factor of u_max
#         p.sensorNoise = sensorNoise_globals  # noise, as factor of max values
#         p.g = g_globals  # gravity, m/s^2
#         p.k = k_globals  # Dimensionless factor, for moment of inertia of the pend (with L being half if the length)

#         # State of the cart
#         s = SimpleNamespace()  # s like state
#         s.position = 0.0
#         s.positionD = 0.0
#         s.angle = 0.0
#         s.angleD = 0.0

#         model_type = 'continuous'  # either 'discrete' or 'continuous'
#         self.model = do_mpc.model.Model(model_type)

#         s.position = self.model.set_variable(var_type='_x', var_name='s.position', shape=(1, 1))
#         s.positionD = self.model.set_variable(var_type='_x', var_name='s.positionD', shape=(1, 1))

#         s.angle = self.model.set_variable(var_type='_x', var_name='s.angle', shape=(1, 1))
#         s.angleD = self.model.set_variable(var_type='_x', var_name='s.angleD', shape=(1, 1))

#         Q = self.model.set_variable(var_type='_u', var_name='Q')

#         target_position = self.model.set_variable('_tvp', 'target_position')

#         self.model.set_rhs('s.position', s.positionD)
#         self.model.set_rhs('s.angle', s.angleD)

#         angleD_next, positionD_next = cartpole_ode(p, s, Q2u(Q,p))

#         self.model.set_rhs('s.positionD', positionD_next)
#         self.model.set_rhs('s.angleD', angleD_next)

#         # Simplified, normalized expressions for E_kin and E_pot as a port of cost function
#         E_kin_cart = (s.positionD / p.v_max) ** 2
#         E_kin_pol = (s.angleD/(2*np.pi))**2
#         E_pot = np.cos(s.angle)

#         distance_difference = ((s.position - target_position) ** 2)

#         self.model.set_expression('E_kin_cart', E_kin_cart)
#         self.model.set_expression('E_kin_pol', E_kin_pol)
#         self.model.set_expression('E_pot', E_pot)
#         self.model.set_expression('distance_difference', distance_difference)


#         self.model.setup()

#         self.mpc = do_mpc.controller.MPC(self.model)

#         setup_mpc = {
#             'n_horizon': mpc_horizon_globals,
#             't_step': dt_mpc_simulation_globals,
#             'n_robust': 0,
#             'store_full_solution': False,
#             'store_lagr_multiplier': False,
#             'store_solver_stats': []
#         }
#         self.mpc.set_param(**setup_mpc)
#         self.mpc.set_param(nlpsol_opts={'ipopt.linear_solver': 'mumps'})
#         # Other possible linear solvers from hsl library
#         # The give better performance 2-3 times.
#         # However if simulating at max speedup the simulation blocks. Issue with memory leak?
#         # self.mpc.set_param(nlpsol_opts={'ipopt.linear_solver': 'MA27'})
#         # self.mpc.set_param(nlpsol_opts = {'ipopt.linear_solver': 'MA57'})

#         lterm = - self.model.aux['E_pot'] + 0.02 * distance_difference
#         mterm = 5 * self.model.aux['E_kin_pol'] - 5 * self.model.aux['E_pot']  + 5 * self.model.aux['E_kin_cart']

#         self.mpc.set_objective(mterm=mterm, lterm=lterm)
#         self.mpc.set_rterm(Q=0.1)

#         self.mpc.bounds['lower', '_u', 'Q'] = -1.0
#         self.mpc.bounds['upper', '_u', 'Q'] = 1.0

#         self.tvp_template = self.mpc.get_tvp_template()

#         self.mpc.set_tvp_fun(self.tvp_fun)

#         # Suppress IPOPT outputs
#         # suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
#         # self.mpc.set_param(nlpsol_opts=suppress_ipopt)

#         self.mpc.setup()

#         # Set initial state
#         self.x0 = self.mpc.x0
#         self.x0['s.position'] = position_init
#         self.x0['s.positionD'] = positionD_init
#         self.x0['s.angle'] = angle_init
#         self.x0['s.angleD'] = angleD_init


#         self.mpc.x0 = self.x0

#         self.mpc.set_initial_guess()

#     def tvp_fun(self, t_ind):
#         return self.tvp_template


#     def step(self, s, target_position):

#         self.x0['s.position'] = s.position
#         self.x0['s.positionD'] = s.positionD

#         self.x0['s.angle'] = s.angle
#         self.x0['s.angleD'] = s.angleD

#         self.tvp_template['_tvp', :, 'target_position'] = target_position

#         Q = self.mpc.make_step(self.x0)

#         return Q.item()