"""
Generate new target position along timeline.
It is an intermediate step to calculate a new control signal along the prerecorded trajectories.
"""

from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = 'SI_Toolkit_ASF/Experiments/Experiment-24-quant-10-shiftD-1/Recordings'
save_files_to = 'SI_Toolkit_ASF/Experiments/Experiment-24-quant-10-shiftD-1/RecordingsAdaptiveTarget'
new_target_equilibrium_variable_name = 'target_equilibrium_new'

transform_dataset(get_files_from, save_files_to,
                  transformation='flip_target_equilibrium',
                  new_target_equilibrium_variable_name=new_target_equilibrium_variable_name,
                  )
