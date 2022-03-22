import numpy as np
from numba import jit, prange, cuda

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
        dldxk = dldx(x,u,Nu)
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

cost_bp = make_cost_backprop(dldu,dldx,dldx,dxdu,dxdx)
u = np.array([-0.1,-0.1,-0.1,-0.1],dtype=np.float64)
u = u[:,np.newaxis]
u = np.ones((50,1))*0.1
x0 = 1
x = evalDyn(u,x0)
print('u = {}'.format(u))
print('x = {}'.format(x))
lgd = np.zeros((u.shape[0],1))
for n in range(0,1):
    u = u-0.1*lgd
    x = evalDyn(u, x0)
    cost = evalCost(u,x)
    print('cost = {}'.format(cost))
    lgd = cost_bp(x,u)

print('final u = {}'.format(u))
print('final x = {}'.format(x))
print('final lgd = {}'.format(lgd))


f_to_measure = 'lgd = cost_bp(x,u)'
number = 1  # Gives the number of times each timeit call executes the function which we want to measure
repeat_timeit = 10000  # Gives how many times timeit should be repeated
timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
min_time = min(timings) / float(number)
max_time = max(timings) / float(number)
average_time = np.mean(timings) / float(number)
print()
print('----------------------------------------------------------------------------------')
print('Min time to evaluate is {} us'.format(min_time * 1.0e6))  # ca. 5 us
print('Average time to evaluate is {} us'.format(average_time * 1.0e6))  # ca 5 us
# The max is of little relevance as it is heavily influenced by other processes running on the computer at the same time
print('Max time to evaluate is {} us'.format(max_time * 1.0e6))  # ca. 100 us
print('----------------------------------------------------------------------------------')
print()
pass