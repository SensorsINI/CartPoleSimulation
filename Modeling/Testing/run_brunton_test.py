# "Command line" parameters
from Modeling.Testing.Parameters_for_testing import args
from Modeling.Testing.preprocess_for_brunton import preprocess_for_brunton

from Modeling.load_and_normalize import \
    load_data, get_sampling_interval_from_datafile, load_cartpole_parameters


# Custom functions
# from Modeling.Testing.get_prediction_TF import get_data_for_gui_TF
from Modeling.Testing.get_prediction_TF_predictor import get_data_for_gui_TF
from get_prediction_from_euler import get_prediction_from_euler
from get_prediction_from_euler_predictor import get_prediction_from_euler_predictor

from Modeling.Testing.Brunton_GUI import run_test_gui

print('')
a = args()  # 'a' like arguments
print(a.__dict__)
print('')

def run_brunton_test():

    dataset, time_axis, dataset_sampling_dt, ground_truth = preprocess_for_brunton(a)

    predictions_list = []
    for test_idx in range(len(a.tests)):
        if a.tests[test_idx] == 'Euler':
            predictions = get_prediction_from_euler(a, dataset, dataset_sampling_dt, dt_sampling_by_dt_fine=10)
        elif a.tests[test_idx] == 'Euler-predictor':
            predictions = get_prediction_from_euler_predictor(a, dataset, dataset_sampling_dt, dt_sampling_by_dt_fine=10)
        else: # Assume this is a neural_network test:
            predictions = get_data_for_gui_TF(a, dataset, net_name=a.tests[test_idx])

        predictions_list.append(predictions)

    run_test_gui(a.features, a.titles,
                 ground_truth, predictions_list, time_axis,
                 )


if __name__=='__main__':
    run_brunton_test()