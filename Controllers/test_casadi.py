import casadi
from casadi import *
from CartPole.cartpole_model import _cartpole_ode, Q2u, v_max

import numpy as np
import matplotlib.pyplot as plt

from CartPole.state_utilities import ANGLE_IDX, ANGLED_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX, POSITION_IDX, POSITIOND_IDX

s0 = [0.01, 0.0, 0.0, 0.0]

number_of_features = 4
number_of_control_inputs = 1

T = 1.0  # Time horizon
N = 5  # Number of control intervals

angle = SX.sym('angle')
angleD = SX.sym('angleD')
position = SX.sym('position')
positionD = SX.sym('positionD')
Q = SX.sym('Q')

s = vertcat(angle, angleD, position, positionD)

ca = cos(angle)
sa = sin(angle)
angleDD, positionDD = _cartpole_ode(ca, sa, angleD, positionD, Q2u(Q))

ode = vertcat(angleD, angleDD, positionD, positionDD)

# cartpole_ode = Function('cartpole_ode', [ca, sa, angleD, positionD, Q], [angleDD, positionDD],
#                         ['ca', 'sa', 'angleD', 'positionD', 'Q'], ['angleDD', 'positionDD'])
# cartpole_ode(0.1, 0.2, 12.0, 5.4, -0.3)

dae = {'x': s, 'p': Q, 'ode': ode}

intg_options = {'tf': T / N, 'simplify': True, 'number_of_finite_elements': 4}

intg = integrator('intg', 'rk', dae, intg_options)
res = intg(x0=s, p=Q)
x_next = res['xf']

s_next = Function('s_next', [s, Q], [x_next], ['s', 'Q'], ['x_next'])

sim = s_next.mapaccum(int(N))

u_scenario = np.zeros(N)

res = sim(s0, u_scenario)
res_np = np.array(res).squeeze()
angle_predicted = res_np[0, :]

plt.figure()
plt.plot(angle_predicted)
plt.show()
#
# U = SX.sym('U', 1, N)
# X1 = sim(s0, U)[1, :]
# J = jacobian(X1, U)
#
# Jf = Function('Jf', [U], [J], ['U'], ['J'])
#
# plt.spy(Jf(0))
# plt.show()

# MPC implementation

opti = casadi.Opti()

s = opti.variable(number_of_features, N+1)
u = opti.variable(number_of_control_inputs, N)
s0 = opti.variable(number_of_features, 1)

# E_kin_cart = (positionD / v_max) ** 2
# E_kin_pol = (angleD / (2 * np.pi)) ** 2
# E_pot = cos(s.angle)
