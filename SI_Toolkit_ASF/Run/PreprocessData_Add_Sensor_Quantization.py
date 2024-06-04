from SI_Toolkit.data_preprocessing import transform_dataset
import numpy as np

from CartPole.cartpole_parameters import TrackHalfLength

ANGLE_360_DEG_IN_ADC_UNITS = 4068.67
POSITION_ENCODER_RANGE = 4705

get_files_from = './SI_Toolkit_ASF/Experiments/CPS-17-02-2023-UpDown-Imitation/Recordings/Train/'
save_files_to = './SI_Toolkit_ASF/Experiments/CPS-17-02-2023-UpDown-Imitation-quant/Recordings/Train/'

# You are responsible to make sure that the units make sense and you have right dt! - this is NOT taken automatically from the dataset
angle_precision = (1.0/ANGLE_360_DEG_IN_ADC_UNITS)*2*np.pi  # rad, 12 bit ADC
position_precision = (2*TrackHalfLength/POSITION_ENCODER_RANGE)  # m
measurement_interval = 0.001  # s
angleD_precision = angle_precision/measurement_interval  # rad/s
positionD_precision = position_precision/measurement_interval  # m/s


variables_quantization_dict = {
    'angle': angle_precision,
    'position': position_precision,
    'angleD': angleD_precision,
    'positionD': positionD_precision,
}


if __name__ == '__main__':
    transform_dataset(get_files_from, save_files_to, transformation='apply_sensors_quantization',
                      variables_quantization_dict=variables_quantization_dict)