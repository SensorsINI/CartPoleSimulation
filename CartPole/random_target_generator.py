import numpy as np
from scipy.interpolate import BPoly, interp1d

from CartPole.cartpole_parameters import TrackHalfLength


# Generates a random target position
# in a form of a function interpolating between turning points
def Generate_Random_Trace_Function(

        length_of_experiment,
        rtf_rng,

        track_relative_complexity,
        interpolation_type,
        turning_points,
        turning_points_period,

        start_random_target_position_at,
        end_random_target_position_at,

        used_track_fraction,
):
    if (turning_points is None) or (turning_points == []):

        number_of_turning_points = int(np.floor(length_of_experiment * track_relative_complexity))

        y = rtf_rng.uniform(-1.0, 1.0, number_of_turning_points)
        y = y * used_track_fraction * TrackHalfLength

        if number_of_turning_points == 0:
            y = np.append(y, 0.0)
            y = np.append(y, 0.0)
        elif number_of_turning_points == 1:
            if start_random_target_position_at is not None:
                y[0] = start_random_target_position_at
            elif end_random_target_position_at is not None:
                y[0] = end_random_target_position_at
            else:
                pass
            y = np.append(y, y[0])
        else:
            if start_random_target_position_at is not None:
                y[0] = start_random_target_position_at
            if end_random_target_position_at is not None:
                y[-1] = end_random_target_position_at

    else:
        number_of_turning_points = len(turning_points)
        if number_of_turning_points == 0:
            raise ValueError('You should not be here!')
        elif number_of_turning_points == 1:
            y = np.array([turning_points[0], turning_points[0]])
        else:
            y = np.array(turning_points)

    random_samples = number_of_turning_points - 2 if number_of_turning_points - 2 >= 0 else 0

    if turning_points_period == 'random':
        t_init = np.sort(rtf_rng.uniform(0.0, 1.0, random_samples))
        t_init = np.insert(t_init, 0, 0.0)
        t_init = np.append(t_init, 1.0)
    elif turning_points_period == 'regular':
        t_init = np.linspace(0, 1.0, num=random_samples + 2, endpoint=True)
    else:
        raise NotImplementedError('There is no mode corresponding to this value of turning_points_period variable')

    t_init = t_init * length_of_experiment

    # Try algorithm setting derivative to 0 a each point
    if interpolation_type == '0-derivative-smooth':
        yder = [[y[i], 0] for i in range(len(y))]
        random_track_f = BPoly.from_derivatives(t_init, yder, extrapolate='periodic')
    elif interpolation_type == 'linear':
        random_track_f = interp1d(t_init, y, kind='linear', fill_value='extrapolate')
    elif interpolation_type == 'previous':
        random_track_f = interp1d(t_init, y, kind='previous', fill_value='extrapolate')
    else:
        raise ValueError('Unknown interpolation type.')

    # Truncate the target position to be not grater than 80% of track length
    def random_track_f_truncated(time):

        target_position = random_track_f(time)
        target_position = np.clip(target_position, -used_track_fraction * TrackHalfLength, used_track_fraction * TrackHalfLength)

        return target_position

    return random_track_f_truncated
