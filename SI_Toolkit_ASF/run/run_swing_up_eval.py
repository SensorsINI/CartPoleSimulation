from others.globals_and_utils import load_config
from run_data_generator import run_data_generator

# Automatically create new path to save everything in

import yaml
import os
import shutil
import numpy as np

config_SI = load_config(os.path.join("SI_Toolkit_ASF", "config_training.yml"))
config_cartpole = load_config(os.path.join("config.yml"))


def get_record_path():
    experiment_index = 1
    while True:
        record_path = "Experiment-" + str(experiment_index)
        if os.path.exists(config_SI['paths']['PATH_TO_EXPERIMENT_FOLDERS'] + record_path):
            experiment_index += 1
        else:
            record_path += "/Recordings"
            break

    record_path = config_SI['paths']['PATH_TO_EXPERIMENT_FOLDERS'] + record_path
    return record_path


if __name__ == '__main__':
    MODEL_DIR = './SI_Toolkit_ASF/Experiments/CPS-Tutorial/Models'
    MODELS = os.listdir(MODEL_DIR)
    results_dict = {}
    for model in sorted(MODELS, key=lambda x: int(x[1:].split('-')[0])):
        print(model)
        tempconf = yaml.load(open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml"),
                                  "r"), Loader=yaml.FullLoader)
        tempconf['mpc']['predictor_specification'] = model
        yaml.dump(tempconf, open(os.path.join("Control_Toolkit_ASF", "config_controllers.yml"),'w'))

        record_path = get_record_path()
        print(record_path)

        config_SI['modeling']['NET_NAME'] = model

        # Save copy of configs in experiment folder
        if not os.path.exists(record_path):
            os.makedirs(record_path)

        # Copy configs at the moment of creation of dataset

        yaml.dump(config_SI, open(record_path + "/SI_Toolkit_config_savefile.yml", "w"),
                  default_flow_style=False)

        shutil.copy2(
            "config_data_gen.yml",
            os.path.join(record_path, "config_data_gen.yml"))

        shutil.copy2(
            os.path.join("SI_Toolkit_ASF", "config_training.yml"),
            os.path.join(record_path, "config_training.yml"))

        shutil.copy2(
            os.path.join("SI_Toolkit_ASF", "config_predictors.yml"),
            os.path.join(record_path, "config_predictors.yml"))

        shutil.copy2(
            os.path.join("Control_Toolkit_ASF", "config_controllers.yml"),
            os.path.join(record_path, "config_controllers.yml"))

        shutil.copy2(
            os.path.join("Control_Toolkit_ASF", "config_cost_function.yml"),
            os.path.join(record_path, "config_cost_function.yml"))

        shutil.copy2(
            os.path.join("Control_Toolkit_ASF", "config_optimizers.yml"),
            os.path.join(record_path, "config_optimizers.yml"))

        # Run data generator
        swing_up_times = run_data_generator(run_for_ML_Pipeline=True, record_path=record_path)
        results_dict[model] = swing_up_times

    np.save(os.path.join(MODEL_DIR, 'swing_up_times.npy'), results_dict)

