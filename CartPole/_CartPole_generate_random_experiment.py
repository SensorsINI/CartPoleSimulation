import numpy as np

# Interpolate function to create smooth random track
from scipy.interpolate import interp1d, BPoly

# Generates a random target position in a form of a function
def Generate_Random_Trace_Function(self):
    if (self.turning_points is None) or (self.turning_points == []):

        number_of_turning_points = int(np.floor(self.random_length * self.track_relative_complexity))

        y = 2.0 * (np.random.random(number_of_turning_points) - 0.5)
        y = y * 0.5 * self.HalfLength

        if number_of_turning_points == 0:
            y = np.append(y, 0.0)
            y = np.append(y, 0.0)
        elif number_of_turning_points == 1:
            if self.start_random_target_position_at is not None:
                y[0] = self.start_random_target_position_at
            elif self.end_random_target_position_at is not None:
                y[0] = self.end_random_target_position_at
            else:
                pass
            y = np.append(y, y[0])
        else:
            if self.start_random_target_position_at is not None:
                y[0] = self.start_random_target_position_at
            if self.end_random_target_position_at is not None:
                y[-1] = self.end_random_target_position_at

    else:
        number_of_turning_points = len(self.turning_points)
        if number_of_turning_points == 0:
            raise ValueError('You should not be here!')
        elif number_of_turning_points == 1:
            y = np.array([self.turning_points[0], self.turning_points[0]])
        else:
            y = np.array(self.turning_points)

    number_of_timesteps = np.ceil(self.random_length / self.dt_simulation)
    self.t_max_pre = number_of_timesteps * self.dt_simulation

    random_samples = number_of_turning_points - 2 if number_of_turning_points - 2 >= 0 else 0

    # t_init = linspace(0, self.t_max_pre, num=self.track_relative_complexity, endpoint=True)
    if self.turning_points_period == 'random':
        t_init = np.sort(np.random.uniform(self.dt_simulation, self.t_max_pre - self.dt_simulation, random_samples))
        t_init = np.insert(t_init, 0, 0.0)
        t_init = np.append(t_init, self.t_max_pre)
    elif self.turning_points_period == 'regular':
        t_init = np.linspace(0, self.t_max_pre, num=random_samples + 2, endpoint=True)
    else:
        raise NotImplementedError('There is no mode corresponding to this value of turning_points_period variable')

    # Try algorithm setting derivative to 0 a each point
    if self.interpolation_type == '0-derivative-smooth':
        yder = [[y[i], 0] for i in range(len(y))]
        random_track_f = BPoly.from_derivatives(t_init, yder)
    elif self.interpolation_type == 'linear':
        random_track_f = interp1d(t_init, y, kind='linear')
    elif self.interpolation_type == 'previous':
        random_track_f = interp1d(t_init, y, kind='previous')
    else:
        raise ValueError('Unknown interpolation type.')

    # Truncate the target position to be not grater than 80% of track length
    def random_track_f_truncated(time):

        target_position = random_track_f(time)
        if target_position > 0.8 * self.HalfLength:
            target_position = 0.8 * self.HalfLength
        elif target_position < -0.8 * self.HalfLength:
            target_position = -0.8 * self.HalfLength

        return target_position

    self.random_track_f = random_track_f_truncated

    self.new_track_generated = True