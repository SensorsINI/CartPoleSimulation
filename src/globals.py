import numpy as np


# Variables controlling flow of the program
mode_globals = 2
save_history_globals = True
load_recording_globals = False

# Variables used for physical simulation
dt_main_simulation_globals = 0.020
speedup_globals = 1.0

# MPC
dt_mpc_simulation_globals = 0.20
mpc_horizon_globals = 20

# Parameters of the CartPole
m_globals = 2.0  # mass of pend, kg
M_globals = 1.0  # mass of cart, kg
L_globals = 1.0  # half length of pend, m
u_max_globals = 200.0  # max cart force, N
M_fric_globals = 0.0  # 1.0, # cart friction, N/m/s
J_fric_globals = 0.0  # 10.0, # friction coefficient on angular velocity, Nm/rad/s
v_max_globals = 10.0  # max DC motor speed, m/s, in absense of friction, used for motor back EMF model
controlDisturbance_globals = 0.0  # 0.01, # disturbance, as factor of u_max
sensorNoise_globals = 0.0  # 0.01, # noise, as factor of max values

g_globals = 9.81  # gravity, m/s^2
k_globals = 4.0 / 3.0  # Dimensionless factor, for moment of inertia of the pend (with L being half if the length)

# Variables for random trace generation
N_globals = 10  # Complexity of the random trace, number of random points used for interpolation
random_length_globals = 1e2  # Number of points in the random length trece

# def cartpole_dyncamic(position,):

def cartpole_integration(s, dt):

    position = s.position + s.positionD * dt
    positionD = s.positionD + s.positionDD * dt

    angle = s.angle + s.angleD * dt
    angleD = s.angleD + s.angleDD * dt

    return position, positionD, angle, angleD

def cartpole_ode(p, s, u):
    ca = np.cos(s.angle)
    sa = np.sin(s.angle)
    A = (p.k + 1) * (p.M + p.m) - p.m * (ca ** 2)

    angleDD = (p.g * (p.m + p.M) * sa -
               ((p.J_fric * (p.m + p.M) * s.angleD) / (p.L * p.m)) -
               ca * (p.m * p.L * (s.angleD ** 2) * sa + p.M_fric * s.positionD) +
               ca * u) / (A * p.L)
    positionDD = (
                             p.m * p.g * sa * ca -
                             ((p.J_fric * s.angleD * ca) / (p.L)) -
                             (p.k + 1) * (p.m * p.L * (s.angleD ** 2) * sa + p.M_fric * s.positionD) +
                             (p.k + 1) * u
                        ) / A

    return angleDD, positionDD


def Q2u(Q, p):
    u = p.u_max * Q + p.controlDisturbance * np.random.normal() * p.u_max  # Q is drive -1:1 range, add noise on control
    return u


