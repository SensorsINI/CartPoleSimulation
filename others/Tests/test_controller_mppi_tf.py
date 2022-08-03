import numpy as np
from Control_Toolkit.Controllers.scontroller_mppi_tf import controller_mppi_tf

if __name__ == '__main__':
    ctrl = controller_mppi_tf()
    import timeit

    from CartPole.cartpole_model import TrackHalfLength, s0, u_max
    from CartPole.state_utilities import (ANGLE_COS_IDX, ANGLE_IDX,
                                          ANGLE_SIN_IDX, ANGLED_IDX,
                                          POSITION_IDX, POSITIOND_IDX,
                                          create_cartpole_state)

    s0 = create_cartpole_state()
    # Set non-zero input
    s = s0
    s[POSITION_IDX] = -30.2
    s[POSITIOND_IDX] = 2.87
    s[ANGLE_IDX] = -0.32
    s[ANGLED_IDX] = 0.237
    u = -0.24

    ctrl.step(s0)
    f_to_measure = 'ctrl.step(s0)'
    number = 1  # Gives the number of times each timeit call executes the function which we want to measure
    repeat_timeit = 1000  # Gives how many times timeit should be repeated
    timings = timeit.Timer(f_to_measure, globals=globals()).repeat(repeat_timeit, number)
    min_time = min(timings) / float(number)
    max_time = max(timings) / float(number)
    average_time = np.mean(timings) / float(number)
    print()
    print('----------------------------------------------------------------------------------')
    print('Min time to evaluate is {} ms'.format(min_time * 1.0e3))  # ca. 5 us
    print('Average time to evaluate is {} ms'.format(average_time * 1.0e3))  # ca 5 us
    # The max is of little relevance as it is heavily influenced by other processes running on the computer at the same time
    print('Max time to evaluate is {} ms'.format(max_time * 1.0e3))  # ca. 100 us
    print('----------------------------------------------------------------------------------')
    print()
