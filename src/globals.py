"""Parameters of CartPole system and simulation, Ground Truth equations of CartPole"""

from src.cartpole_model import *



# Variables settings parameters CartPole GUI starts with
# This is useful if you need to many times restart the GUI to some particular settings
# e.g. while testing new controller
# TODO: Give controller name instead
mode_globals = 5  # Defines which controller is loaded

controller_interval_threshold_globals = 0.1
"""Sets the controller of the CartPole

Possible choices - out of data
manual stabilization
linear-quadratic regulator (LQR)
MPC based on Ground Truth CartPole ODE (continuous model) implemented with do-mpc library
MPC based on Ground Truth CartPole ODE wrapped (integrated) to predict the next state of the system (discrete model)
    implemented in do-mpc library
    RNN trained to mimic the MPC controller, RNN implemented in Pytorch
like as 4, just different RNN implemented in TensorFlow
like 3, just CartPole dynamics model provided not with true equation,
        but rather with RNN (Pytorch) trained to predict future state of the CartPole
like  6, just different RNN implemented in TensorFlow

"""
PATH_TO_CONTROLLERS = './controllers/'
PATH_TO_DATA = './data/'
save_history_globals = True  # Save experiment history as CSV
stop_at_90_globals = False  # Block the pole if it reaches +/-90 deg (horizontal position)
load_recording_globals = False  # Start/Stop button: True: load and replay a recording; False: start new experiment
slider_on_click_globals = True  # True: update slider only on click, False: update slider while hoovering over it
speedup_globals = 1.0  # Multiplicative factor by which the simulation seen by the user differs from real time
# E.g. 2.0 means that you watch simulation double speed
# WARNING: This is the target value, max speedup is limited by speed of performing CartPole simulation
# True instantaneous speedup is displayed in CartPole GUI as "Speed-up(measured)"
# BUG: It seems that if speedup is set the way it cannot be reached,
#   it may lead to unstable MPC instead of just making simulation run on the fastest achievable speed

# Variables used for physical simulation
dt_main_simulation_globals = 0.0020  # Time step of CartPole simulation

# MPC
dt_mpc_simulation_globals = 0.2  # Time step used by MPC controller
# WARNING: if using RNN to provide CartPole model to MPC
# make sure that it is trained to predict future states with this timestep
# TODO: Add dt information to .txt file associated with and describing each RNN
mpc_horizon_globals = 10  # Number of steps into future MPC controller simulates at each evaluation

# Variables for random trace generation
# Complexity of the random trace, number of turning points used for interpolation
track_relative_complexity_globals = 0.05  # 0.5 is normal default
random_length_globals = 100.0e1  # Number of seconds in the random length trace
interpolation_type_globals = 'previous'  # Sets how to interpolate between turning points of random trace
# Possible choices: '0-derivative-smooth', 'linear', 'previous'
turning_points_period_globals = 'regular'  # How turning points should be distributed
# Possible choices: 'regular', 'random'
# Where the target position of the random experiment starts and end
start_random_target_position_at_globals = 10.0
end_random_target_position_at_globals = 10.0
# Alternatively you can provide a list of target positions.
# If not None this variable has precedence -
# track_relative_complexity, start/end_random_target_position_at_globals have no effect.
turning_points_globals = None


# turning_points_globals = [10.0, 0.0, 0.0]


def cartpole_integration(s, dt):
    """Simple single step integration of CartPole state by dt

    Takes state as SimpleNamespace, but returns as separate variables
    # TODO: Consider changing it to return a SimpleNamepece for consistency

    :param s: state of the CartPole (contains: s.position, s.positionD, s.angle and s.angleD)
    :param dt: time step by which the CartPole state should be integrated
    """
    s_next = SimpleNamespace()

    s_next.position = s.position + s.positionD * dt
    s_next.positionD = s.positionD + s.positionDD * dt

    s_next.angle = s.angle + s.angleD * dt
    s_next.angleD = s.angleD + s.angleDD * dt

    return s_next


def mpc_next_state(s, p, u, dt):
    """Wrapper for CartPole ODE. Given a current state (without second derivatives), returns a state after time dt

    TODO: This might be combined with cartpole_integration,
        although the order of cartpole_ode and cartpole_integration is different than in CartClass
    """

    s_next = s

    s_next.angleDD, s_next.positionDD = cartpole_ode(p, s_next, u)  # Calculates CURRENT second derivatives

    # Calculate NEXT state:
    s_next = cartpole_integration(s_next, dt)

    return s_next
