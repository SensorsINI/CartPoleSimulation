
import scipy
import numpy as np
from modeling.rnn.utilis_rnn import *


RNN_FULL_NAME = 'GRU-5IN-64H1-64H2-1OUT-0'
INPUTS_LIST = ['s.position', 's.angle']
OUTPUTS_LIST = ['Q']
PATH_SAVE = './save/'

from copy import deepcopy

class controller_rnn_as_mpc:
    def __init__(self):

        self.rnn_full_name = RNN_FULL_NAME
        self.path_save = PATH_SAVE

        self.device = get_device()

        # Create rnn instance and update lists of input, outputs and its name (if pretraind net loaded)
        self.net, self.rnn_name, self.inputs_list, self.outputs_list \
            = create_rnn_instance(load_rnn=self.rnn_full_name, path_save=self.path_save, device=self.device)

        self.normalization_info = load_normalization_info(self.path_save, self.rnn_full_name)

        self.net.reset()
        self.net.eval()

        self.rnn_input = pd.DataFrame(columns=self.inputs_list)
        self.rnn_output = pd.DataFrame(columns=self.outputs_list)


    def step(self, s, target_position):

        # Copy state and target_position into rnn_input

        if 's.position' in self.rnn_input:
            self.rnn_input['s.position'] = [s.position]
        if 's.angle' in self.rnn_input:
            self.rnn_input['s.angle'] = [s.angle]
        if 's.positionD' in self.rnn_input:
            self.rnn_input['s.positionD'] = [s.positionD]
        if 's.angleD' in self.rnn_input:
            self.rnn_input['s.angleD'] = [s.angleD]
        if 'target_position' in self.rnn_input:
            self.rnn_input['target_position'] = [target_position]

        rnn_input_normed = normalize_df(self.rnn_input, self.normalization_info)

        rnn_input_np = np.squeeze(rnn_input_normed.to_numpy())
        rnn_input_torch = torch.from_numpy(rnn_input_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        normalized_rnn_output = self.net(rnn_input=rnn_input_torch)
        normalized_rnn_output = np.squeeze(normalized_rnn_output.detach().cpu().numpy()).tolist()
        normalized_rnn_output = pd.DataFrame(data=[normalized_rnn_output], columns=self.outputs_list)

        denormalized_rnn_output = denormalize_df(normalized_rnn_output, self.normalization_info)

        Q = float(denormalized_rnn_output['Q'])

        return Q
