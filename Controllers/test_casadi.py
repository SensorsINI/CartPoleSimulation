from casadi import *
from CartPole.cartpole_model import _cartpole_ode, Q2u

import numpy as np
import matplotlib.pyplot as plt

angle = SX.sym('ca')
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

T = 10.0  # Time horizon
N = 200  # Number of control intervals
intg_options = {'tf': T / N, 'simplify': True, 'number_of_finite_elements': 4}

intg = integrator('intg', 'rk', dae, intg_options)
res = intg(x0=s, p=Q)
x_next = res['xf']

s_next = Function('s_next', [s, Q], [x_next], ['s', 'Q'], ['x_next'])

sim = s_next.mapaccum(int(N))

u_scenario = np.zeros(N)
s0 = [0.01, 0.0, 0.0, 0.0]

res = sim(s0, u_scenario)
res_np = np.array(res).squeeze()
angle_predicted = res_np[0,:]

plt.figure()
plt.plot(angle_predicted)
plt.show()