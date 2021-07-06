from run_data_generator import run_data_generator

# Automatically create new path to save everything in

import yaml, os
config_SI = yaml.load(open(os.path.join('SI_Toolkit_ApplicationSpecificFiles', 'config.yml')), Loader=yaml.FullLoader)
config_cartpole = yaml.load(open('config.yml'), Loader=yaml.FullLoader)

def get_record_path():
    experiment_index = 1
    while True:
        record_path = "Experiment-" + str(experiment_index)
        if os.path.exists(config_SI['paths']['PATH_TO_EXPERIMENT_RECORDINGS'] + record_path):
            experiment_index += 1
        else:
            record_path += "/Recordings"
            break

    record_path = config_SI['paths']['PATH_TO_EXPERIMENT_RECORDINGS'] + record_path
    return record_path

if __name__ == '__main__':
    record_path = get_record_path()

    # Save copy of configs in experiment folder
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    yaml.dump(config_SI, open(record_path + "/SI_Toolkit_config_savefile.yml", "w"), default_flow_style=False)
    yaml.dump(config_cartpole, open(record_path + "/CartPole_config_savefile.yml", "w"), default_flow_style=False)

    # Run data generator
    run_data_generator(run_for_ML_Pipeline=True, record_path=record_path)
