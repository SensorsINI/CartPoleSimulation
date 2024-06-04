"""
Generate new target position along timeline.
It is an intermediate step to calculate a new control signal along the prerecorded trajectories.
"""

from SI_Toolkit.data_preprocessing import transform_dataset
from others.globals_and_utils import create_rng, load_config

get_files_from = 'SI_Toolkit_ASF/Experiments/Experiment-24-quant-10-shiftD-1/Recordings'
save_files_to = 'SI_Toolkit_ASF/Experiments/Experiment-24-quant-10-shiftD-1/RecordingsAdaptiveTarget'
new_target_position_variable_name = 'target_position_offline'
path_to_config = "config_data_gen.yml"


# Load yaml
config = load_config(path_to_config)

# Create the random number generator
rtf_rng = create_rng('rtf_rng', config['seed'])

# fill the config
target_position_config = {
    'rtf_rng': rtf_rng,

    'track_relative_complexity': config['turning_points']['track_relative_complexity'],
    'interpolation_type': config['turning_points']['interpolation_type'],
    'turning_points': config['turning_points']['turning_points'],
    'turning_points_period': config['turning_points']['turning_points_period'],

    'start_random_target_position_at': config['random_initial_state']['target_position'],
    'end_random_target_position_at': config['target_position_end'],

    'used_track_fraction': config['track_fraction_usable_for_target_position'],
}

transform_dataset(get_files_from, save_files_to,
                  transformation='add_new_target_position',
                  target_position_config=target_position_config,
                  new_target_position_variable_name=new_target_position_variable_name,
                  )
