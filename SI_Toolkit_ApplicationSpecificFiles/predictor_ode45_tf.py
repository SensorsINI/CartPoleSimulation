
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from numpy.random import SFC64, Generator

from CartPole.state_utilities import (
    create_cartpole_state,
    ANGLE_IDX, ANGLED_IDX, POSITION_IDX, POSITIOND_IDX, ANGLE_COS_IDX, ANGLE_SIN_IDX, STATE_VARIABLES
)

from others.p_globals import (
    k, M, m, g, J_fric, M_fric, L, v_max, u_max, controlDisturbance, controlBias, TrackHalfLength
)

from CartPole.cartpole_model import _cartpole_ode

k = tf.convert_to_tensor(k)
M = tf.convert_to_tensor(M)
m = tf.convert_to_tensor(m)
g = tf.convert_to_tensor(g)
J_fric = tf.convert_to_tensor(J_fric)
M_fric = tf.convert_to_tensor(M_fric)
L = tf.convert_to_tensor(L)
v_max = tf.convert_to_tensor(v_max)
u_max = tf.convert_to_tensor(u_max)
controlDisturbance = tf.convert_to_tensor(controlDisturbance)
controlBias = tf.convert_to_tensor(controlBias)
TrackHalfLength = tf.convert_to_tensor(TrackHalfLength)

def Q2u_tf(Q):
    """
    Converts dimensionless motor power [-1,1] to a physical force acting on a cart.

    In future there might be implemented here a more sophisticated model of a motor driving CartPole
    """
    u = tf.convert_to_tensor(u_max, dtype=tf.float32) * (
            Q +
            tf.convert_to_tensor(controlDisturbance, dtype=tf.float32) * tf.random.normal(shape=tf.shape(Q),
                                                                                          dtype=tf.float32) + tf.convert_to_tensor(
        controlBias, dtype=tf.float32)
    )  # Q is drive -1:1 range, add noise on control

    return u

class predictor_ode45_tf():
    def __init__(self, horizon=50, dt=0.02, max_steps = 10, batch_size = 1):
        self.horizon = tf.convert_to_tensor(horizon)
        self.batch_size = batch_size
          # Will be adjusted the control input size

        self.initial_state = tf.zeros(shape=(1, len(STATE_VARIABLES)))


        self.dt = dt
        self.solver = tfp.math.ode.DormandPrince(max_num_steps = max_steps)
        out = tf.zeros([self.batch_size, self.horizon, 6])
        self.out = tf.Variable(out)



    def ode_fun(self, t, y, **const):
        angle = y[0]
        angleD = y[1]

        position = y[2]
        positionD = y[3]

        u = y[4]

        angle_cos = tf.cos(angle)
        angle_sin = tf.sin(angle)
        angleDD, positionDD = _cartpole_ode(angle_cos, angle_sin, angleD, positionD, u,
                                               k, M, m, g, J_fric, M_fric, L)

        return tf.stack([angleD, angleDD, positionD, positionDD, 0])

    # @tf.function(jit_compile=True)
    def predict(self, s, Q):
        s = tf.convert_to_tensor(s)
        Q = tf.convert_to_tensor(Q)
        U = Q2u_tf(Q)

        if tf.rank(s) == 1:
            s = s[tf.newaxis,:]

        # self.batch_size = s.shape[0]

        ys = tf.gather(s,[ANGLE_IDX,ANGLED_IDX,POSITION_IDX,POSITIOND_IDX],axis=1)
        y = tf.concat([ys,U[:,0,:]],1)


        tinit = 0.0


        for k in tf.range(self.horizon):
            for j in tf.range(self.batch_size):
                results = self.solver.solve(self.ode_fun, tinit, y[j], solution_times=[self.dt])
                yo = results.states[0]
                self.out[j,k,ANGLE_IDX].assign(yo[0])
                self.out[j,k,ANGLED_IDX].assign(yo[1])
                self.out[j,k,POSITION_IDX].assign(yo[2])
                self.out[j,k,POSITIOND_IDX].assign(yo[3])

            self.out[:, :, ANGLE_SIN_IDX].assign(tf.sin(self.out[:, :, ANGLE_IDX]))
            self.out[:, :, ANGLE_COS_IDX].assign(tf.cos(self.out[:, :, ANGLE_IDX]))
        return self.out


@tf.function(jit_compile=True)
def predict_wrap(predictor,s,u):
    return predictor.predict(s,u)

if __name__ == '__main__':
    s0 = create_cartpole_state()
    s_org = s0
    s_org[POSITION_IDX] = -30.2
    s_org[POSITIOND_IDX] = 2.87
    s_org[ANGLE_IDX] = -0.32
    s_org[ANGLED_IDX] = 0.237



    # u_org = tf.Variable([0.594623, 0.11093523, -0.32577565, 0.36339644, 0.19863953,
    #                  -0.67005044, -0.00572653, 0.50473666, 0.82851535, 0.03227299,
    #                  -0.89665616, -1., -0.15769833, -0.8742089, -0.00434032,
    #                  -0.5908449, -0.8486508, 0.46566853, -0.26742178, -0.2585441,
    #                  -1., 1., -1., 0.820513, 1.,
    #                  0.65235853, 0.7771242, -0.834638, 0.9568739, 0.21720093,
    #                  -0.18284637, 0.9694907, 0.68292177, -1., 1.,
    #                  0.37337917, -0.46058115, -0.6156913, 0.52652395, 0.06510112,
    #                  -0.13692386, 0.4193466, 0.08954383, -0.02065406, 0.7458399,
    #                  -1., 0.83411133, -0.5809542, -0.5786972, -0.70775455],
    #                 dtype=tf.float32)
    num_rollouts = 1
    predictor = predictor_ode45_tf(batch_size=num_rollouts)

    SEED = 5876
    rng_gen = Generator(SFC64(SEED))
    dist_var = 0.5 * np.ones([1, 50])
    stdev = np.sqrt(dist_var)

    dist_mue = np.zeros([1, 50])
    Q = np.tile(dist_mue, (num_rollouts, 1)) + np.multiply(rng_gen.standard_normal(
        size=(num_rollouts, 50), dtype=np.float32), stdev)
    Q = np.clip(Q, -1.0, 1.0, dtype=np.float32)

    s = s_org
    s = np.tile(s, tf.constant([num_rollouts, 1]))
    s_org = tf.convert_to_tensor(s)

    u = tf.convert_to_tensor(Q)
    u = u[:, :, tf.newaxis]
    rollout_trajectory = predict_wrap(predictor, s, u)

    import timeit

    f_to_measure = 'predictor.predict(s,u)'
    number = 10  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 1
    timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
    print(timings)