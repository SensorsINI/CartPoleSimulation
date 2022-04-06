import numpy as np
from numba import jit, prange, cuda
from CartPole.cartpole_jacobian import cartpole_jacobian
from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, \
    create_cartpole_state
import yaml
from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE

config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

dd_weight = config["controller"]["mppi"]["dd_weight"]
cc_weight = config["controller"]["mppi"]["cc_weight"]
ep_weight = config["controller"]["mppi"]["ep_weight"]
R = config["controller"]["cem"]["cem_R"]
ccrc_weight = config["controller"]["cem"]["cem_ccrc_weight"]

from others.p_globals import (
    k as param_k,  # k
    M as param_M,  # m_c
    m as param_m,  # m_p
    g as param_g,  # g
    J_fric as param_J_fric,  # mue_p
    M_fric as param_M_fric,  # mue_c
    L as param_L,  # l
    u_max as u_max_param
)


def make_J(k=param_k, mc=param_M, mp=param_m, g=param_g, mue_p=param_J_fric, mue_c=param_M_fric, L=param_L,
           u_max=u_max_param):
    # for J[1,1]
    c1 = (1 + k) * mue_c
    c2 = (1 + k) * (mc + mp)

    # for J[1,2]
    c3 = 2 * (1 + k) * mp
    c4 = (1 + k) * L * mp
    c5 = mue_p / L
    gmp = g * mp

    # for J[1,3]
    c6 = c3 * L

    # for J[3,2]
    lmp = L * mp
    c7 = g * (mc + mp)
    c8 = (mc + mp) * mue_p / lmp

    @jit(nopython=True, cache=True, fastmath=True)
    def Jac(x, u, kk):  # x= [x,v,theta,omega]
        J = np.zeros((4, 5))
        xl = x[kk, 0]
        vel = x[kk, 1]
        the = x[kk, 2]
        ome = x[kk, 3]
        u = u_max * u[kk, 0]

        J[0, 1] = 1
        J[2, 3] = 1

        sin_th = np.sin(the)
        cos_th = np.cos(the)
        cos_th2 = cos_th ** 2
        sin_th2 = sin_th ** 2
        ome2 = ome ** 2
        temp1 = (-cos_th2 * mp + c2)
        temp2 = L * temp1

        J[1, 1] = -c1 / temp1
        J[1, 2] = (-c3 * u * cos_th * sin_th / (temp1 ** 2) - (2 * cos_th * sin_th * mp * (
                    -(c4 * sin_th * ome2) + gmp * cos_th * sin_th - c1 * vel + ome * cos_th * c5)) / temp1 ** 2
                   + (-(c4 * ome2 * cos_th) + gmp * (cos_th2 - sin_th2) - ome * sin_th * c5) / temp1)
        J[1, 3] = (-c6 * ome * sin_th + cos_th * c5) / temp1

        J[3, 1] = -cos_th * mue_c / temp2
        J[3, 2] = (-(2 * u * cos_th2 * sin_th * mp) / (L * temp1 ** 2) - u * sin_th / temp2 + (
                    -ome2 * cos_th2 * lmp + ome2 * sin_th2 * lmp + c7 * cos_th + vel * sin_th * mue_c) / temp2
                   - (2 * cos_th * sin_th * mp * (
                            -lmp * ome2 * sin_th * cos_th + c7 * sin_th - vel * mue_c * cos_th - ome * c8)) / (
                               L * temp1 ** 2))
        J[3, 3] = (-2 * lmp * ome * cos_th * sin_th - c8) / temp2

        J[1, 4] = (1 + k) / temp1
        J[3, 4] = cos_th / temp2

        return J

    return Jac


class ControlStateLengthMissmatchError(Exception):
    def __init__(self, message="The length of the state array and input arrays do not coincide."):
        self.message = message
        super().__init__(self.message)

    pass


@jit(nopython=True, cache=True, fastmath=True)
def dldu(x, u, k):
    return 2 * u[k, 0] * cc_weight * R


@jit(nopython=True, cache=True, fastmath=True)
def dldx(x, u, k):
    ret = np.zeros((1, 4))
    # ret[0,0] = dd_weight*2*((x[k,0])/(2.0*param_L))/(2.0*param_L)
    # ret[0,2] = ep_weight*0.25*2*(1.0-np.cos(x[k,2]))*(np.sin(x[k,2]))
    return ret


# @jit(nopython=True, cache=True, fastmath=True)
def dldxn(x):
    ret = np.zeros((1, 4))
    ret[0, 0] = 2 * x[-1, 0]
    # ret[0,0] = np.exp(10*(x[-1,0]-0.2))-np.exp(10*(-x[-1,0]-0.2))
    return ret


def make_cost_backprop(dldu, dldx, dldxn, J, dt):
    # @jit(nopython=True, cache=True, fastmath=True)
    def cost_backprop(s, u):
        Nx = s.shape[0]
        x = np.zeros((Nx, 4))
        x[:, 0] = s[:, POSITION_IDX]
        x[:, 1] = s[:, POSITIOND_IDX]
        x[:, 2] = s[:, ANGLE_IDX]
        x[:, 3] = s[:, ANGLED_IDX]
        Nu = u.shape[0]
        if Nx != Nu + 1:
            raise ControlStateLengthMissmatchError

        lossgrad = np.zeros((Nu, 1), dtype=np.float64)
        dldxk = dldxn(x)
        Jk = J(x, u, Nu - 1)
        dxdxk = np.eye(4)+Jk[0:4, 0:4]*dt
        dxduk = Jk[:, 4]*dt
        dxduk = dxduk[:, np.newaxis]
        dlduk = dldu(x, u, Nu - 1)
        lossgrad[Nu - 1, 0] = dlduk + dldxk @ dxduk
        for k in range(Nu - 2, -1, -1):
            dldx_part = dldx(x, u, k + 1)
            dldxk = dldx_part + dldxk @ dxdxk
            Jk = J(x, u, k)
            dxdxk = np.eye(4)+Jk[0:4, 0:4]*dt
            dxduk = Jk[:, 4]*dt
            dxduk = dxduk[:, np.newaxis]
            lossgrad[k, 0] = dldu(x, u, k) + dldxk @ dxduk
        return lossgrad

    return cost_backprop


def evalDyn(u, x0):
    Nu = u.shape[0]
    x = np.zeros((Nu + 1, 1), dtype=np.float64)
    x[0, 0] = x0
    for i in range(0, Nu):
        x[i + 1, 0] = -1 / 2 * (x[i, 0] ** 2) + 1 / 3 * (u[i, 0] ** 3)

    return x


def evalCost(u, x):
    Nx = x.shape[0]
    Nu = u.shape[0]
    Qb = np.identity(Nx)
    Rb = np.identity(Nu)
    return x.T @ Qb @ x + u.T @ Rb @ u


import timeit

s0 = create_cartpole_state()
# Set non-zero input
s = s0
s[POSITION_IDX] = -20.2
s[POSITIOND_IDX] = 1.87
s[ANGLE_IDX] = -0.42
s[ANGLED_IDX] = 0.537
u = np.array([-0.44])

jac = make_J()
Jo = cartpole_jacobian(s, u[0])
Ji = Jo[0:4, 0:4]
x = np.array([s[POSITION_IDX], s[POSITIOND_IDX], s[ANGLE_IDX], s[ANGLED_IDX]])
Jm = jac(x[np.newaxis, :], u[np.newaxis, :] / u_max_param, 0)

print(Jo)
print(Jm)
# delta = Ji-Jm
# print(delta)
pass

u = np.array([0.594623, 0.11093523, -0.32577565, 0.36339644, 0.19863953,
              -0.67005044, -0.00572653, 0.50473666, 0.82851535, 0.03227299,
              -0.89665616, -1., -0.15769833, -0.8742089, -0.00434032,
              -0.5908449, -0.8486508, 0.46566853, -0.26742178, -0.2585441,
              -1., 1., -1., 0.820513, 1.,
              0.65235853, 0.7771242, -0.834638, 0.9568739, 0.21720093,
              -0.18284637, 0.9694907, 0.68292177, -1., 1.,
              0.37337917, -0.46058115, -0.6156913, 0.52652395, 0.06510112,
              -0.13692386, 0.4193466, 0.08954383, -0.02065406, 0.7458399,
              -1., 0.83411133, -0.5809542, -0.5786972, -0.70775455],
             dtype=np.float32)
u = u[:, np.newaxis]

predictor = predictor_ODE(horizon=u.shape[0], intermediate_steps=1, dt=0.02)
predictor.predict(s[np.newaxis,:], u[np.newaxis,:,:])

s_test = s[-1,:]
u_test = u[-1]
x_test = np.array([s_test[POSITION_IDX], s_test[POSITIOND_IDX], s_test[ANGLE_IDX], s_test[ANGLED_IDX]])
J_test = jac(x_test[np.newaxis, :], u_test[np.newaxis, :] / u_max_param, 0)

cost_bp = make_cost_backprop(dldu, dldx, dldxn, jac,0.02)
lgd = cost_bp(s, u)


pass
# cost_bp = make_cost_backprop(dldu,dldx,dldx,dxdu,dxdx)
# u = np.array([-0.1,-0.1,-0.1,-0.1],dtype=np.float64)
# u = u[:,np.newaxis]
# u = np.ones((50,1))*0.1
# x0 = 1
# x = evalDyn(u,x0)
# print('u = {}'.format(u))
# print('x = {}'.format(x))
# lgd = np.zeros((u.shape[0],1))
# for n in range(0,1):
#     u = u-0.1*lgd
#     x = evalDyn(u, x0)
#     cost = evalCost(u,x)
#     print('cost = {}'.format(cost))
#     lgd = cost_bp(x,u)
#
# print('final u = {}'.format(u))
# print('final x = {}'.format(x))
# print('final lgd = {}'.format(lgd))
#
#
# f_to_measure = 'Jm = jac(x,u/u_max_param)'
# f_to_measure = 'Jm = jac(x,u/u_max_param)'
# number = 1  # Gives the number of times each timeit call executes the function which we want to measure
# repeat_timeit = 1000  # Gives how many times timeit should be repeated
# timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
# min_time = min(timings) / float(number)
# max_time = max(timings) / float(number)
# average_time = np.mean(timings) / float(number)
# print()
# print('----------------------------------------------------------------------------------')
# print('Min time to evaluate is {} us'.format(min_time * 1.0e6))  # ca. 5 us
# print('Average time to evaluate is {} us'.format(average_time * 1.0e6))  # ca 5 us
# # The max is of little relevance as it is heavily influenced by other processes running on the computer at the same time
# print('Max time to evaluate is {} us'.format(max_time * 1.0e6))  # ca. 100 us
# print('----------------------------------------------------------------------------------')
# print()
# # pass
