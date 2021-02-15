
from tqdm import trange

import numpy as np
from CartPole.cartpole_model import Q2u, cartpole_ode

from memory_profiler import profile

# @profile(precision=4)
def generate_cartpole_experiment(MyCart,
                        initial_state,
                        exp_len,
                        dt,
                        track_relative_complexity,
                        interpolation_type,
                        turning_points_period,
                        start_random_target_position_at,
                        end_random_target_position_at,
                        csv=None,
                        save_data_online = True,
                        controller='manual-stabilization'):
    """
    This function runs a random CartPole experiment
    and returns the history of CartPole states, control inputs and desired cart position
    :param MyCart: instance of CartPole containing CartPole dynamics
    :param exp_len: How many time steps should the experiment have
                (default: 64+640+1 this format is used as it can )
    """



    # Set CartPole in the right (automatic control) mode
    try:
        mode = MyCart.controller_names.index(controller)
    except ValueError:
        raise ValueError('{} is not in list. \n In list are: {}'.format(controller, MyCart.controller_names))
    MyCart.set_mode(mode)

    MyCart.turning_points = None
    MyCart.interpolation_type = interpolation_type
    MyCart.turning_points_period = turning_points_period
    MyCart.start_random_target_position_at = start_random_target_position_at
    MyCart.end_random_target_position_at = end_random_target_position_at

    if save_data_online:
        MyCart.save_history = False
    else:
        MyCart.save_history = True
    MyCart.use_pregenerated_target_position = 1
    MyCart.dt = dt

    # Generate new random function returning desired target position of the cart
    MyCart.random_length = exp_len
    MyCart.track_relative_complexity = track_relative_complexity  # Complexity of generated target position track
    MyCart.Generate_Random_Trace_Function()
    number_of_timesteps = int(np.ceil(MyCart.random_length / MyCart.dt))

    # Randomly set the initial state

    MyCart.time = 0.0
    if initial_state[0] is None:
        # CartPoleInstance.s.position = np.random.uniform(low=-CartPoleInstance.HalfLength / 2.0,
        #                                       high=CartPoleInstance.HalfLength / 2.0)
        MyCart.s.position = np.random.uniform(low=-MyCart.HalfLength / 4.0,
                                              high=MyCart.HalfLength / 4.0)
    else:
        MyCart.s.position = initial_state[0]

    if initial_state[1] is None:
        # CartPoleInstance.s.positionD = np.random.uniform(low=-10.0,
        #                                        high=10.0)
        MyCart.s.positionD = np.random.uniform(low=-1.0,
                                               high=1.0)
    else:
        MyCart.s.positionD = initial_state[1]

    if initial_state[2] is None:
        # CartPoleInstance.s.angle = np.random.uniform(low=-17.5 * (np.pi / 180.0),
        #                                    high=17.5 * (np.pi / 180.0))
        MyCart.s.angle = np.random.uniform(low=-3.5 * (np.pi / 180.0),
                                           high=3.5 * (np.pi / 180.0))
    else:
        MyCart.s.angle = initial_state[2]

    if initial_state[3] is None:
        # CartPoleInstance.s.angleD = np.random.uniform(low=-15.5 * (np.pi / 180.0),
        #                                     high=15.5 * (np.pi / 180.0))
        MyCart.s.angleD = np.random.uniform(low=-3.0 * (np.pi / 180.0),
                                            high=3.0 * (np.pi / 180.0))
    else:
        MyCart.s.angleD = initial_state[3]


    # Target position at time 0
    target_position = MyCart.random_track_f(MyCart.time)

    # Make already in the first timestep Q appropriate to the initial state, target position and controller

    MyCart.Update_Q()

    MyCart.set_cartpole_state_at_t0(reset_mode=2, s=s, Q=MyCart.Q, target_position=target_position)










    if save_data_online and csv is not None:
        MyCart.save_history_csv(csv_name=csv, init=True, iter=False)
        MyCart.save_history_csv(csv_name=csv, init=False, iter=True)

    # Run the CartPole experiment for number of time
    for _ in trange(number_of_timesteps):

        # Print an error message if it runs already to long (should stop before)
        if MyCart.time > MyCart.t_max_pre:
            raise Exception('ERROR: It seems the experiment is running too long...')

        MyCart.update_state()
        if abs(MyCart.s.position)>45.0:
            break
            print('Cart went out of safety boundaries')

        # if abs(CartPoleInstance.s.angle) > 0.8*np.pi:
        #     raise ValueError('Cart went unstable')

        if save_data_online and csv is not None:
            MyCart.save_history_csv(csv_name=csv, init=False, iter=True)

    data = pd.DataFrame(MyCart.dict_history)

    if csv is not None and not save_data_online:
        MyCart.save_history_csv(csv_name=csv)

    if not save_data_online:
        MyCart.summary_plots()

    # Set CartPole state - the only use is to make sure that experiment history is discared
    # Maybe you can delete this line
    MyCart.set_cartpole_state_at_t0(reset_mode=0)

    return data