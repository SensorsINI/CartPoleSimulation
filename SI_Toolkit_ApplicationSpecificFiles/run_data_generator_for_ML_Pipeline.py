from run_data_generator import run_data_generator

# Automatically create new path to save everything in

import yaml, os
config = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml')), Loader=yaml.FullLoader)

experiment_index = 1
while True:
    record_path = "Experiment-" + str(experiment_index)
    if os.path.exists(config['paths']['PATH_TO_EXPERIMENT_RECORDINGS'] + record_path):
        experiment_index += 1
    else:
        record_path += "/Recordings"
        break

    record_path = config['paths']['PATH_TO_EXPERIMENT_RECORDINGS'] + record_path

run_data_generator(run_for_ML_Pipeline=True, record_path=record_path)