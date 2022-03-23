import numpy as np
from numba import jit, prange, cuda
from CartPole.cartpole_jacobian import cartpole_jacobian
from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state

from others.p_globals import (
    k as param_k, #k
    M as param_M, #m_c
    m as param_m, #m_p
    g as param_g, #g
    J_fric as param_J_fric, #mue_p
    M_fric as param_M_fric, #mue_c
    L as param_L, #l
    u_max as u_max_param
)
def make_J(k = param_k,mc=param_M,mp=param_m,g=param_g,mue_p = param_J_fric,mue_c=param_M_fric,L = param_L, u_max = u_max_param):
    #for J[1,1]
    c1 = (1+k)*mue_c
    c2 = (1+k)*(mc+mp)

    #for J[1,2]
    c3 = 2*(1+k)*mp
    c4 = (1+k)*L*mp
    c5 = mue_p/L
    gmp = g*mp

    # for J[1,3]
    c6 = c3*L

    def Jac(x,u): #x= [x,v,theta,omega]
        J = np.zeros((4,4))
        xl = x[0]
        vel = x[1]
        the = x[2]
        ome = x[3]
        u = u_max*u

        J[0,1] = 1
        J[2,3] = 1

        sin_th = np.sin(the)
        cos_th = np.cos(the)
        cos_th2 = cos_th**2
        sin_th2 = sin_th**2
        ome2 = ome**2
        temp1 = (-cos_th2 * mp + c2)

        J[1,1] = -c1/temp1
        J[1,2] = (-c3 * u * cos_th * sin_th / (temp1 ** 2) - (2 * cos_th * sin_th * mp * (-(c4 * sin_th * ome2) + gmp * cos_th * sin_th - c1 * vel + ome * cos_th * c5)) / temp1 ** 2
                  + (-(c4*ome2*cos_th)+gmp*(cos_th2-sin_th2)-ome*sin_th*c5) / temp1)
        J[1,3] = (-c6*ome*sin_th+cos_th*c5) / temp1

        #J[3,1] =
        return J

    return Jac





class ControlStateLengthMissmatchError(Exception):
    def __init__(self,message="The length of the state array and input arrays do not coincide."):
        self.message = message
        super().__init__(self.message)
    pass

@jit(nopython=True, cache=True, fastmath=True)
def dldu(x,u,k):
    return u[k,0]

@jit(nopython=True, cache=True, fastmath=True)
def dldx(x,u,k):
    return x[k,0]

@jit(nopython=True, cache=True, fastmath=True)
def dxdu(x,u,k):
    return u[k,0]**2

@jit(nopython=True, cache=True, fastmath=True)
def dxdx(x,u,k):
    return -x[k,0]




def make_cost_backprop(dldu,dldx,dldxn,dxdu,dxdx):

    @jit(nopython=True, cache=True, fastmath=True)
    def cost_backprop(x,u):
        Nx = x.shape[0]
        Nu = u.shape[0]
        if Nx != Nu+1:
            raise ControlStateLengthMissmatchError

        lossgrad = np.zeros((Nu,1),dtype = np.float64)
        dldxk = dldxn(x,u,Nu)
        dxkduk = dxdu(x,u,Nu-1)
        dlduk = dldu(x,u,Nu-1)
        lossgrad[Nu-1,0]=dlduk+dldxk*dxkduk
        for k in range(Nu-2, -1, -1):
            dldxk =dldx(x, u, k+1)+dldxk*dxdx(x, u, k+1)
            lossgrad[k,0]=dldu(x, u, k)+dldxk*dxdu(x, u, k)
        return lossgrad

    return cost_backprop

def evalDyn(u,x0):
    Nu = u.shape[0]
    x = np.zeros((Nu+1,1),dtype = np.float64)
    x[0,0]=x0
    for i in range(0,Nu):
        x[i+1,0] = -1/2*(x[i,0]**2)+1/3*(u[i,0]**3)

    return x

def evalCost(u,x):
    Nx = x.shape[0]
    Nu = u.shape[0]
    Qb = np.identity(Nx)
    Rb = np.identity(Nu)
    return x.T@Qb@x+u.T@Rb@u


import timeit

s0 = create_cartpole_state()
# Set non-zero input
s = s0
s[POSITION_IDX] = -30.2
s[POSITIOND_IDX] = 2.87
s[ANGLE_IDX] = -0.32
s[ANGLED_IDX] = 0.237
u = -0.24

jac = make_J()
Jo = cartpole_jacobian(s,u)
Ji = Jo[0:4,0:4]
x = np.array([s[POSITION_IDX],s[POSITIOND_IDX],s[ANGLE_IDX],s[ANGLED_IDX]])
Jm = jac(x,u/u_max_param)

print(Ji)
print(Jm)
delta = Ji-Jm
print(delta)
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
# f_to_measure = 'lgd = cost_bp(x,u)'
# number = 1  # Gives the number of times each timeit call executes the function which we want to measure
# repeat_timeit = 10000  # Gives how many times timeit should be repeated
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
# pass