import csv, os, yaml
import numpy as np
from CartPole.state_utilities import STATE_VARIABLES, CONTROL_INPUTS
#from SI_Toolkit.Predictors.predictor_noisy import predictor_noisy
from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
import re
import timeit

dt = 0.02
intermediate_steps = 10

config = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_training.yml'), 'r'),
                   Loader=yaml.FullLoader)
parent_dir = config['paths']['PATH_TO_EXPERIMENT_FOLDERS']
experiment_folder = config['paths']['path_to_experiment']

training_experiments = os.listdir(os.path.join(parent_dir, experiment_folder, 'Recordings/Train'))
validation_experiments = os.listdir(os.path.join(parent_dir, experiment_folder, 'Recordings/Validate'))
test_experiments = os.listdir(os.path.join(parent_dir, experiment_folder, 'Recordings/Test'))

if experiment_folder[-1] == '/':
    new_folder = experiment_folder[0:-1] + '-Augmented/'
else:
    print('No \'/\' after experiment folder')
    new_folder = ''
    exit()
os.mkdir(os.path.join(parent_dir, new_folder))
os.mkdir(os.path.join(parent_dir, new_folder, 'Recordings'))
os.mkdir(os.path.join(parent_dir, new_folder, 'Recordings/Train'))
os.mkdir(os.path.join(parent_dir, new_folder, 'Recordings/Validate'))
os.mkdir(os.path.join(parent_dir, new_folder, 'Recordings/Test'))

################################# Select Predictor #################################

predictor = predictor_ODE(horizon=1, dt=dt, intermediate_steps=intermediate_steps)

####################################################################################
irrelevant_columns = [11,11,12,12,12,14,15]  # choose the columns you want to remove. Note: The index stays at the same spot after removal

for experiment in test_experiments:
    data = []
    counter = 0
    t0 = 0
    with open(os.path.join(parent_dir, experiment_folder, 'Recordings/Test/', str(experiment))) as file:
        reader = csv.reader(file, delimiter=',')
        for x in reader:
            if bool(x):  # Sometimes lines are empty in csv
                if x[0][0] != '#':
                    counter += 1
                    if counter == 2:  # We need to normalize the time to 0 for the test data
                        t0 = float(x[0])
                        x[0] = '0'
                    elif counter > 2:
                        x[0] = str(float(x[0]) - t0)
                    if bool(irrelevant_columns):
                        for i in irrelevant_columns:
                            del x[i]
                data.append(x)
    start = 0
    while data[start][0][0] == '#':
        start += 1
    start += 1  # start is the row in which the actual numerical data starts. In row start-1 are the column titles
    i = 0
    relevant_columns = []
    Q_index = []
    for x in data[start - 1]:
        if x in STATE_VARIABLES:
            relevant_columns.append(i)
            data[start - 1].append(data[start - 1][i] + '_pred')
        if x in CONTROL_INPUTS:
            Q_index.append(i)
        i += 1
    if len(relevant_columns) != len(STATE_VARIABLES) or len(Q_index) != len(CONTROL_INPUTS):
        print('Column names not matching state variable names')
        exit()
    initial_state = np.array(data[start:])
    Q = initial_state[:, Q_index].astype(float)
    initial_state = initial_state[:, relevant_columns].astype(float)
    for i in range(len(data) - start):
        if start + i + 1 < len(data):  # Don't do prediction if it's the last entry
            prediction = predictor.predict(initial_state[i, :], Q[i, :])
            for x in prediction[1]:
                data[start + i + 1].append(x)
    del data[start]

    with open(os.path.join(parent_dir, new_folder, 'Recordings/Test/', str(experiment)), 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for x in data:
            writer.writerow(x)

for experiment in training_experiments:
    data = []
    with open(os.path.join(parent_dir, experiment_folder, 'Recordings/Train/', str(experiment))) as file:
        reader = csv.reader(file, delimiter=',')
        for x in reader:
            if bool(x):
                if bool(irrelevant_columns):
                    if x[0][0] != '#':
                        i = 0
                        consumable_irrelevant_columns = irrelevant_columns.copy()
                        while i < len(x):
                            if i in consumable_irrelevant_columns:
                                del x[i]
                                consumable_irrelevant_columns.remove(i)
                                i -= 1
                            i += 1
                data.append(x)

    start = 0
    while data[start][0][0] == '#':
        start += 1
    start += 1  # start is the row in which the actual numerical data starts. In row start-1 are the column titles
    i = 0
    relevant_columns = []
    Q_index = []
    for x in data[start - 1]:
        if x in STATE_VARIABLES:
            relevant_columns.append(i)
            data[start - 1].append(data[start - 1][i] + '_pred')
        if x in CONTROL_INPUTS:
            Q_index.append(i)
        i += 1
    if len(relevant_columns) != len(STATE_VARIABLES) or len(Q_index) != len(CONTROL_INPUTS):
        print('Column names not matching state variable names')
        exit()
    initial_state = np.array(data[start:])
    Q = initial_state[:, Q_index].astype(float)
    initial_state = initial_state[:, relevant_columns].astype(float)
    for i in range(len(data) - start):
        if start + i + 1 < len(data):  # Don't do prediction if it's the last entry
            prediction = predictor.predict(initial_state[i, :], Q[i, :])
            for x in prediction[1]:
                data[start + i + 1].append(x)
    del data[start]

    with open(os.path.join(parent_dir, new_folder, 'Recordings/Train/', str(experiment)), 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for x in data:
            writer.writerow(x)

for experiment in validation_experiments:
    data = []
    with open(os.path.join(parent_dir, experiment_folder, 'Recordings/Validate/', str(experiment))) as file:
        reader = csv.reader(file, delimiter=',')
        for x in reader:
            if bool(x):
                if bool(irrelevant_columns):
                    if x[0][0] != '#':
                        i = 0
                        consumable_irrelevant_columns = irrelevant_columns.copy()
                        while i < len(x):
                            if i in consumable_irrelevant_columns:
                                del x[i]
                                consumable_irrelevant_columns.remove(i)
                                i -= 1
                            i += 1
                data.append(x)

    start = 0
    while data[start][0][0] == '#':
        start += 1
    start += 1
    i = 0
    relevant_columns = []
    Q_index = []
    for x in data[start - 1]:
        if x in STATE_VARIABLES:
            relevant_columns.append(i)
            data[start - 1].append(data[start - 1][i] + '_pred')
        if x in CONTROL_INPUTS:
            Q_index.append(i)
        i += 1
    if len(relevant_columns) != len(STATE_VARIABLES) or len(Q_index) != len(CONTROL_INPUTS):
        print('Column names not matching state variable names')
        exit()
    initial_state = np.array(data[start:])
    Q = initial_state[:, Q_index].astype(float)
    initial_state = initial_state[:, relevant_columns].astype(float)
    for i in range(len(data) - start):
        if start + i + 1 < len(data):  # Don't do prediction if it's the last entry
            prediction = predictor.predict(initial_state[i, :], Q[i, :])
            for x in prediction[1]:
                data[start + i + 1].append(x)
    del data[start]

    with open(os.path.join(parent_dir, new_folder, 'Recordings/Validate/', str(experiment)), 'w') as file:
        writer = csv.writer(file, delimiter=',')
        for x in data:
            writer.writerow(x)

print("Execution Successful")
