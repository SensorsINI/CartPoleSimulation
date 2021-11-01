# Parameters CartPole GUI starts with
# This is useful if you need to many times restart the GUI to some particular setting
# e.g. while testing new controller

# TIME SCALES - timescales cannot be changed after starting the program
dt_simulation = 0.002  # s, time step of CartPole simulation if run through GUI
controller_update_interval = 0.02  # s, sets how often the control input should be calculated
# The last variable must be in practice the multiple of dt_simulation
# Hence the provided value is translated to interval given in the number of time steps of main simulation
# controller_update_interval_steps =  np.rint(controller_update_interval/dt_simulation)
# 0 is treated the same as 1 with controller updated at every simulation time step
save_interval = 0.02  # s, How often to save the results of simulation
# The same considerations as for controller_update_interval apply
# We recommend to set it to the same value as controller_update_interval.
# In such a case saving is done at the same time step as the controller is updated


# Starting setting of the CartPole GUI (can be changed in GUI)
# controller_init = 'custom-mpc-scipy'  # Defines which controller is loaded at the start of the program
controller_init = 'lqr'
"""
Possible choices for CartPole controller:

'manual-stabilization' -> GUI slider sets directly motor power
'lqr' -> linear-quadratic regulator (LQR)
'do-mpc' -> we provide do-mpc library with true equations, it internally integrates it with cvodes
'do-mpc-discrete' -> we provide do-mpc library with function returning the next state with single step eurler stepping
'rnn-as-mpc-pytorch' -> RNN trained to mimic the MPC controller, RNN implemented in Pytorch
'rnn-as-mpc-tf' -> like above, just different RNN implemented in TensorFlow
'custom-mpc-scipy' -> we optimize the cost function by calling scipy.optimize.minimize()
'mpc-opti' -> mpc implementation based on casadi opti library
"""

save_history_init = True  # Save experiment history as CSV (after experiment finished, online saving for Data Generator only)
show_experiment_summary_init = True  # If true a window plotting the experiment history will be displayed after experiment ended
stop_at_90_init = False  # Block the pole if it reaches +/-90 deg (horizontal position)
slider_on_click_init = False  # True: update slider only on click, False: update slider while hoovering over it
simulator_mode_init = 'Slider-Controlled Experiment'  # Effects Start/Stop button:
# 'Slider-Controlled Experiment': Run experiment controlled by user (through slider)
# 'Random Experiment': Run experiment with random target position
# 'Replay': load and replay a recording; False: start new experiment
speedup_init = 1.0  # Multiplicative factor by which the simulation seen by the user differs from real time
# E.g. 2.0 means that you watch simulation double speed
# WARNING: This is the target value, max speedup is limited by speed of performing CartPole simulation
# True instantaneous speedup is displayed in CartPole GUI as "Speed-up(measured)"


# Variables for random trace generation - GUI only
# Data Generator sets these parameters independently
track_relative_complexity_init = 0.5  # randomly placed target points/s, 0.5 is normal default
length_of_experiment_init = 100.0e1  # seconds, length of the random length trace
interpolation_type_init = 'previous'  # Sets how to interpolate between turning points of random trace
# Possible choices: '0-derivative-smooth', 'linear', 'previous'
turning_points_period_init = 'regular'  # How turning points should be distributed
# Possible choices: 'regular', 'random'
# Regular means that they are equidistant from each other
# Random means we pick randomly points at time axis at which we place turning points
# Where the target position of the random experiment starts and end:
start_random_target_position_at_init = 10.0
end_random_target_position_at_init = 10.0
# Alternatively you can provide a list of target positions.
# e.g. turning_points_globals = [10.0, 0.0, 0.0]
# If not None this variable has precedence -
# track_relative_complexity, start/end_random_target_position_at_globals have no effect.
turning_points_init = None