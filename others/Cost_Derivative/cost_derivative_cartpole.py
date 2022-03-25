import numpy as np
from numba import jit, prange, cuda
from CartPole.cartpole_jacobian import cartpole_jacobian
from CartPole.state_utilities import ANGLE_IDX, ANGLE_SIN_IDX, ANGLE_COS_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, create_cartpole_state
import yaml

config = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)

dd_weight = config["controller"]["mppi"]["dd_weight"]
cc_weight = config["controller"]["mppi"]["cc_weight"]
ep_weight = config["controller"]["mppi"]["ep_weight"]
R = config["controller"]["cem"]["cem_R"]
ccrc_weight = config["controller"]["cem"]["cem_ccrc_weight"]

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

    # for J[3,2]
    lmp = L * mp
    c7 = g*(mc+mp)
    c8 = (mc+mp)*mue_p/lmp

    @jit(nopython=True, cache=True, fastmath=True)
    def Jac(x,u,k): #x= [x,v,theta,omega]
        J = np.zeros((4,5))
        xl = x[k,0]
        vel = x[k,1]
        the = x[k,2]
        ome = x[k,3]
        u = u_max*u[k,0]

        J[0,1] = 1
        J[2,3] = 1

        sin_th = np.sin(the)
        cos_th = np.cos(the)
        cos_th2 = cos_th**2
        sin_th2 = sin_th**2
        ome2 = ome**2
        temp1 = (-cos_th2 * mp + c2)
        temp2 = L*temp1

        J[1,1] = -c1/temp1
        J[1,2] = (-c3 * u * cos_th * sin_th / (temp1 ** 2) - (2 * cos_th * sin_th * mp * (-(c4 * sin_th * ome2) + gmp * cos_th * sin_th - c1 * vel + ome * cos_th * c5)) / temp1 ** 2
                  + (-(c4*ome2*cos_th)+gmp*(cos_th2-sin_th2)-ome*sin_th*c5) / temp1)
        J[1,3] = (-c6*ome*sin_th+cos_th*c5) / temp1

        J[3,1] = cos_th*mue_c/temp2
        J[3,2] = (-(2*u*cos_th2*sin_th*mp) / (L*temp1**2) - u * sin_th / temp2 + (-ome2 * cos_th2 * lmp + ome2 * sin_th2 * lmp + c7 * cos_th + vel * sin_th * mue_c) / temp2
                  - (2*cos_th*sin_th*mp*(-lmp*ome2*sin_th*cos_th+c7*sin_th-vel*mue_c*cos_th-ome*c8)) / (L*temp1**2))
        J[3,3] = (-2*lmp*ome*cos_th*sin_th-c8)/temp2

        J[1,4] = (1+k) / temp1
        J[3,4] = cos_th/temp2

        return J

    return Jac





class ControlStateLengthMissmatchError(Exception):
    def __init__(self,message="The length of the state array and input arrays do not coincide."):
        self.message = message
        super().__init__(self.message)
    pass

@jit(nopython=True, cache=True, fastmath=True)
def dldu(x,u,k):
    return 2*u[k,0]*cc_weight*R

@jit(nopython=True, cache=True, fastmath=True)
def dldx(x,u,k):
    ret = np.zeros((1,4))
    ret[0,0] = dd_weight*2*((x[k,0])/(2.0*param_L))/(2.0*param_L)
    ret[0,2] = ep_weight*0.25*2*(1.0-np.cos(x[k,2]))*(-np.sin(x[k,2]))
    return ret

# @jit(nopython=True, cache=True, fastmath=True)
def dldxn(x):
    ret = np.zeros((1,4))
    ret[0,0] = np.exp(10*(x[-1,0]-0.2))-np.exp(10*(-x[-1,0]-0.2))
    return ret

def make_cost_backprop(dldu,dldx,dldxn,J):

    # @jit(nopython=True, cache=True, fastmath=True)
    def cost_backprop(s,u):
        Nx = s.shape[0]
        x = np.zeros((Nx,4))
        x[:,0] = s[:,POSITION_IDX]
        x[:,1] = s[:,POSITIOND_IDX]
        x[:,2]= s[:,ANGLE_IDX]
        x[:,3] = s[:,ANGLED_IDX]
        Nu = u.shape[0]
        if Nx != Nu+1:
            raise ControlStateLengthMissmatchError

        lossgrad = np.zeros((Nu,1),dtype = np.float64)
        dldxk = dldxn(x)
        Jk = J(x,u,Nu-1)
        dxdxk = Jk[0:4,0:4]
        dxduk = Jk[:,4]
        dxduk = dxduk[:,np.newaxis]
        dlduk = dldu(x,u,Nu-1)
        lossgrad[Nu-1,0]=dlduk+dldxk@dxduk
        for k in range(Nu-2, -1, -1):
            dldxk =dldx(x, u, k+1)+dldxk@dxdxk
            Jk = J(x, u, k)
            dxdxk = Jk[0:4, 0:4]
            dxduk = Jk[:, 4]
            dxduk = dxduk[:, np.newaxis]
            lossgrad[k,0]=dldu(x, u, k)+dldxk@dxduk
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
u = np.array([-0.24])

jac = make_J()
Jo = cartpole_jacobian(s,u[0])
Ji = Jo[0:4,0:4]
x = np.array([s[POSITION_IDX],s[POSITIOND_IDX],s[ANGLE_IDX],s[ANGLED_IDX]])
Jm = jac(s[np.newaxis,:],u[np.newaxis,:],0)

#print(Ji)
print(Jm)
# delta = Ji-Jm
#print(delta)
pass

u = np.array([ 0.594623  ,  0.11093523, -0.32577565,  0.36339644,  0.19863953,
       -0.67005044, -0.00572653,  0.50473666,  0.82851535,  0.03227299,
       -0.89665616, -1.        , -0.15769833, -0.8742089 , -0.00434032,
       -0.5908449 , -0.8486508 ,  0.46566853, -0.26742178, -0.2585441 ,
       -1.        ,  1.        , -1.        ,  0.820513  ,  1.        ,
        0.65235853,  0.7771242 , -0.834638  ,  0.9568739 ,  0.21720093,
       -0.18284637,  0.9694907 ,  0.68292177, -1.        ,  1.        ,
        0.37337917, -0.46058115, -0.6156913 ,  0.52652395,  0.06510112,
       -0.13692386,  0.4193466 ,  0.08954383, -0.02065406,  0.7458399 ,
       -1.        ,  0.83411133, -0.5809542 , -0.5786972 , -0.70775455],
      dtype=np.float32)
u = u[:,np.newaxis]

s = np.array([[ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
         0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
       [ 3.83169856e-03,  3.98153991e-01,  9.99992669e-01,
         3.83168925e-03,  1.00858125e-03,  1.04695752e-01],
       [ 1.13258809e-02,  3.51048648e-01,  9.99935865e-01,
         1.13256387e-02,  2.96960003e-03,  9.09709483e-02],
       [ 1.52857695e-02,  3.47912908e-02,  9.99883175e-01,
         1.52851744e-02,  3.96082783e-03,  5.06368279e-03],
       [ 1.83911826e-02,  2.86046237e-01,  9.99830902e-01,
         1.83901452e-02,  4.66920249e-03,  6.80904239e-02],
       [ 2.47545894e-02,  3.54734600e-01,  9.99693632e-01,
         2.47520618e-02,  6.16573635e-03,  8.21266100e-02],
       [ 2.67870054e-02, -1.68795094e-01,  9.99641240e-01,
         2.67838016e-02,  6.42919168e-03, -6.09649271e-02],
       [ 2.43054722e-02, -7.51480162e-02,  9.99704659e-01,
         2.43030787e-02,  5.39922621e-03, -4.13394161e-02],
       [ 2.67372094e-02,  3.34630400e-01,  9.99642611e-01,
         2.67340243e-02,  5.56602282e-03,  6.17874227e-02],
       [ 3.82728130e-02,  8.40327621e-01,  9.99267697e-01,
         3.82634699e-02,  8.02758057e-03,  1.89085796e-01],
       [ 5.34143448e-02,  6.72238171e-01,  9.98573780e-01,
         5.33889495e-02,  1.12973014e-02,  1.36090174e-01],
       [ 5.99645078e-02, -3.90898511e-02,  9.98202682e-01,
         5.99285774e-02,  1.21024577e-02, -6.27443790e-02],
       [ 5.39842062e-02, -5.76561213e-01,  9.98543203e-01,
         5.39579876e-02,  9.37106926e-03, -2.16034204e-01],
       [ 4.43959236e-02, -3.74045074e-01,  9.99014676e-01,
         4.43813428e-02,  5.47099812e-03, -1.72465190e-01],
       [ 3.36648114e-02, -7.10997403e-01,  9.99433398e-01,
         3.36584523e-02,  1.08951121e-03, -2.69303590e-01],
       [ 2.28295885e-02, -3.60141069e-01,  9.99739408e-01,
         2.28276048e-02, -3.45984474e-03, -1.82565346e-01],
       [ 1.41391242e-02, -5.14903665e-01,  9.99900043e-01,
         1.41386529e-02, -7.54068512e-03, -2.27217063e-01],
       [ 1.13876071e-03, -7.96861708e-01,  9.99999344e-01,
         1.13876048e-03, -1.28178392e-02, -3.03377748e-01],
       [-8.25479440e-03, -1.19714566e-01,  9.99965906e-01,
        -8.25470034e-03, -1.71595495e-02, -1.24367870e-01],
       [-1.09854648e-02, -1.55503273e-01,  9.99939680e-01,
        -1.09852441e-02, -1.97200626e-02, -1.31983340e-01],
       [-1.43091232e-02, -1.78795680e-01,  9.99897599e-01,
        -1.43086351e-02, -2.23955028e-02, -1.35724500e-01],
       [-2.28599384e-02, -6.97061896e-01,  9.99738693e-01,
        -2.28579473e-02, -2.63931695e-02, -2.68953562e-01],
       [-2.74180137e-02,  2.74289042e-01,  9.99624133e-01,
        -2.74145789e-02, -2.92564891e-02, -7.93282967e-03],
       [-2.84979492e-02, -4.08326656e-01,  9.99593973e-01,
        -2.84940917e-02, -3.10974326e-02, -1.82516888e-01],
       [-2.94897798e-02,  3.34335625e-01,  9.99565184e-01,
        -2.94855051e-02, -3.28052938e-02,  1.90505162e-02],
       [-1.68191809e-02,  9.55945134e-01,  9.99858558e-01,
        -1.68183874e-02, -3.07988003e-02,  1.87851191e-01],
       [ 4.19130223e-03,  1.15509832e+00,  9.99991238e-01,
         4.19129012e-03, -2.65206303e-02,  2.42113546e-01],
       [ 2.95314863e-02,  1.39196026e+00,  9.99563992e-01,
         2.95271948e-02, -2.11044271e-02,  3.01881731e-01],
       [ 4.87465449e-02,  5.02466917e-01,  9.98812139e-01,
         4.87272404e-02, -1.73999369e-02,  5.99581264e-02],
       [ 6.46660402e-02,  1.11647642e+00,  9.97909904e-01,
         6.46209791e-02, -1.47431130e-02,  2.11341888e-01],
       [ 8.64864737e-02,  1.07075655e+00,  9.96262372e-01,
         8.63786936e-02, -1.07725775e-02,  1.84931621e-01],
       [ 1.05318867e-01,  8.10573399e-01,  9.94459093e-01,
         1.05124272e-01, -7.91707076e-03,  9.76084992e-02],
       [ 1.27476677e-01,  1.43610036e+00,  9.91885841e-01,
         1.27131701e-01, -4.58242558e-03,  2.41215244e-01],
       [ 1.58829451e-01,  1.72044420e+00,  9.87413108e-01,
         1.58162504e-01,  7.04449485e-04,  2.89429963e-01],
       [ 1.84938386e-01,  8.72782171e-01,  9.82947588e-01,
         1.83885977e-01,  3.98278702e-03,  2.92026512e-02],
       [ 2.09863350e-01,  1.66020656e+00,  9.78059411e-01,
         2.08326250e-01,  6.23245491e-03,  2.02187032e-01],
       [ 2.44831920e-01,  1.85919654e+00,  9.70178068e-01,
         2.42393255e-01,  1.03639625e-02,  2.11507618e-01],
       [ 2.78821588e-01,  1.54623091e+00,  9.61380422e-01,
         2.75222927e-01,  1.32708000e-02,  7.44285434e-02],
       [ 3.07447314e-01,  1.32647622e+00,  9.53109205e-01,
         3.02626640e-01,  1.36116249e-02, -4.45265584e-02],
       [ 3.40079665e-01,  1.97870636e+00,  9.42728102e-01,
         3.33562195e-01,  1.38397124e-02,  7.16986060e-02],
       [ 3.81957471e-01,  2.24028754e+00,  9.27936792e-01,
         3.72737586e-01,  1.52791170e-02,  7.24014416e-02],
       [ 4.28187788e-01,  2.41335869e+00,  9.09719706e-01,
         4.15222883e-01,  1.64058283e-02,  3.91655304e-02],
       [ 4.81658101e-01,  2.97944975e+00,  8.86228025e-01,
         4.63249266e-01,  1.78771354e-02,  1.10554948e-01],
       [ 5.44204891e-01,  3.31683564e+00,  8.55539203e-01,
         5.17737985e-01,  2.00242363e-02,  1.03823148e-01],
       [ 6.13360465e-01,  3.64294457e+00,  8.17718267e-01,
         5.75618625e-01,  2.18763594e-02,  8.02988037e-02],
       [ 6.93304956e-01,  4.41054535e+00,  7.69138098e-01,
         6.39082611e-01,  2.45121270e-02,  1.86590061e-01],
       [ 7.80126035e-01,  4.31276846e+00,  7.10824907e-01,
         7.03369021e-01,  2.62380224e-02, -2.23523322e-02],
       [ 8.74743521e-01,  5.21533155e+00,  6.41193688e-01,
         7.67379045e-01,  2.71472149e-02,  1.17353752e-01],
       [ 9.81392384e-01,  5.50824404e+00,  5.55865645e-01,
         8.31272185e-01,  2.82772016e-02, -1.07419221e-02],
       [ 1.09534526e+00,  5.95206070e+00,  4.57739532e-01,
         8.89086306e-01,  2.71528456e-02, -1.07428588e-01],
       [ 1.21889699e+00,  6.47339296e+00,  3.44681382e-01,
         9.38719749e-01,  2.40949821e-02, -2.04719409e-01]], dtype=np.float32)

cost_bp = make_cost_backprop(dldu,dldx,dldxn,jac)
lgd = cost_bp(s,u)
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
#f_to_measure = 'Jm = jac(x,u/u_max_param)'
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